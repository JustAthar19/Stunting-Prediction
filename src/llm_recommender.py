from __future__ import annotations
import os
from typing import Any
from pydantic import BaseModel, Field
from src.rag.guideline_rag import maybe_auto_build_index, rag_answer
from dotenv import load_dotenv
load_dotenv()

class LLMRecommendation(BaseModel):
    diagnosis: dict[str, str]
    recommendation_markdown: str = Field(..., description="Recommendations in Markdown.")
    model_used: str
    used_fallback: bool


def _fallback_recommendation(diagnosis: dict[str, str], patient: dict[str, Any]) -> str:
    age = patient.get("age_months")
    haz = diagnosis.get("Height per Age", "")
    waz = diagnosis.get("Weight per Age", "")
    whz = diagnosis.get("Weight per Height", "")

    age_aware_feeding_hints: list[str] = ["", "### Tips makanan sesuai usia"]
    if age <= 6:
        age_aware_feeding_hints += [ "- **0–5 bulan**: ASI eksklusif. Jangan beri air, madu, atau makanan lain kecuali atas anjuran tenaga kesehatan.",]

    elif 6 <= age <= 8:
        age_aware_feeding_hints += [
            "- **6–8 bulan**: Mulai MP-ASI **lumat/kental** (bukan bubur encer).",
            "- Frekuensi: 2–3x makan utama + ASI.",
            "- Tambahkan **protein hewani setiap hari** (telur/ikan/ayam, porsi kecil)."]


    elif 9 <= age <= 11:
        age_aware_feeding_hints += [
            "- **9–11 bulan**: Tekstur lebih padat, mulai **finger food** bertahap.",
            "- Frekuensi: 3x makan utama + 1–2 selingan + ASI.",
            "- Variasikan protein hewani dan sayur.",
        ]

    elif 12 <= age <= 23:
        age_aware_feeding_hints += [
            "- **12–23 bulan**: Makanan keluarga (dipotong kecil, empuk).",
            "- Frekuensi: 3x makan utama + 2 selingan.",
            "- Pastikan **protein hewani harian** dan energi cukup.",
        ]

    else:  # ≥24 bulan
        age_aware_feeding_hints += [
            "- **≥24 bulan**: Makanan keluarga seimbang.",
            "- Biasakan makan teratur, lauk hewani, sayur, buah.",
        ]

    base = [
        "### Langkah berikutnya",
        "- **Pantau pertumbuhan**: ukur berat & tinggi tiap bulan, catat di KMS/Buku KIA.",
        "- **Cek asupan**: pastikan makan utama + selingan sesuai usia, dengan protein hewani rutin.",
        "- **Cek kesehatan**: bila ada diare berulang, nafsu makan turun, atau demam berkepanjangan, periksa ke puskesmas/dokter.",
        "",
        "### Ide makanan (pilih sesuai usia & ketersediaan)",
        "- **Protein hewani**: telur, ikan, ayam, hati (porsi kecil tapi sering).",
        "- **Protein nabati**: tempe, tahu, kacang-kacangan (tekstur disesuaikan).",
        "- **Sumber energi**: nasi, kentang, ubi, roti.",
        "- **Sayur & buah**: bayam, wortel, brokoli, pepaya, pisang.",
        "- **Lemak sehat**: santan secukupnya, minyak, alpukat.",
    ]

    focus: list[str] = ["", "### Tindak lanjut berdasarkan hasil"]
    if "sangat" in haz or haz.lower().startswith("stunting") or "stunting" in haz:
        focus += [
            "- **Tinggi berdasarkan umur dibawah rata-rata**: prioritaskan **protein hewani harian** + energi cukup (jangan hanya bubur encer).",
            "- **Target**: makan padat gizi, tambah 1 porsi protein hewani per hari bila memungkinkan.",
        ]
    if "buruk" in waz.lower() or "kurang" in waz.lower():
        focus += [
            "- **Berat menurut umur dibawah rata-rata**: tambah **frekuensi makan** (makan utama + 2 selingan) dan tingkatkan porsi bertahap.",
            "- **Contoh selingan**: pisang + susu/yogurt (jika cocok), bubur kacang hijau, roti + telur.",
        ]
    if "buruk" in whz or "kurang" in whz:
        focus += [
            "- **Tanda gizi buruk**: diperlukan **evaluasi tenaga kesehatan** lebih cepat (risiko gizi buruk).",
            "- **Segera**: konsultasi puskesmas/dokter untuk rencana terapi gizi yang aman.",
        ]

    
    if age is not None:
        age_aware_feeding_hints.insert(0, f"**Usia**: {age} bulan")

    return "\n".join(age_aware_feeding_hints + base + focus)



def generate_recommendation(diagnosis: dict[str, str], patient: dict[str, Any]) -> LLMRecommendation:
    model = "gemini-2.5-flash-lite"
    api_key = os.getenv("GEMINI_API_KEY")

    prompt = f"""
        Anda adalah asisten edukasi gizi anak. Buat rekomendasi langkah berikutnya untuk mencegah/mengatasi stunting.
        Gunakan bahasa indonesia yang sederhana, praktis.

        DATA PASIEN:
        - usia (bulan): {patient.get("age_months")}
        - jenis kelamin (0=perempuan, 1=laki-laki): {patient.get("sex")}
        - berat (kg): {patient.get("weight_kg")}
        - tinggi/panjang (cm): {patient.get("height_cm")}
        
        HASIL (WHO):
        - Height per Age: {diagnosis.get("Height per Age")}
        - Weight per Age: {diagnosis.get("Weight per Age")}
        - Weight per Height: {diagnosis.get("Weight per Height")}

        OUTPUT YANG DIMINTA:
        1) Ringkasan hasil (1-2 kalimat).
        2) Langkah aksi 7 hari ke depan (berikan dalam bentuk bullet points).
        3) Menu/ide makanan (berikan dalam bentuk bullet points), fokus protein hewani & variasi lokal.
        4) Tanda bahaya kapan harus ke tenaga kesehatan (berikan dalam bentuk bullet points).
        Jangan berikan dosis obat. Jika wasting/SAM, tekankan perlu evaluasi tenaga kesehatan.
        """.strip()

    try:
        maybe_auto_build_index()
        rag_text = rag_answer(question=prompt, gemini_api_key=api_key, model=model)
        if rag_text:
            return LLMRecommendation(
                diagnosis=diagnosis,
                recommendation_markdown=rag_text.strip(),
                model_used=f"rag+{model}",
                used_fallback=False,
            )
        raise RuntimeError("RAG unavailable or returned empty answer")
    except Exception as e:
        print(f"error: {e}")
        return LLMRecommendation(
            diagnosis=diagnosis,
            recommendation_markdown=_fallback_recommendation(diagnosis, patient),
            model_used="fallback",
            used_fallback=True,
        )
