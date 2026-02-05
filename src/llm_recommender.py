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

    focus: list[str] = ["", "### Fokus berdasarkan hasil"]
    if "Severly" in haz or haz.lower().startswith("stunted") or "Stunted" in haz:
        focus += [
            "- **Tinggi menurut umur rendah**: prioritaskan **protein hewani harian** + energi cukup (jangan hanya bubur encer).",
            "- **Target**: makan padat gizi, tambah 1 porsi protein hewani per hari bila memungkinkan.",
        ]
    if "underweight" in waz.lower() or "severly underweight" in waz.lower():
        focus += [
            "- **Berat menurut umur rendah**: tambah **frekuensi makan** (makan utama + 2 selingan) dan tingkatkan porsi bertahap.",
            "- **Contoh selingan**: pisang + susu/yogurt (jika cocok), bubur kacang hijau, roti + telur.",
        ]
    if "Wasting" in whz or "SAM" in whz:
        focus += [
            "- **Tanda wasting**: ini perlu **evaluasi tenaga kesehatan** lebih cepat (risiko gizi buruk).",
            "- **Segera**: konsultasi puskesmas/dokter untuk rencana terapi gizi yang aman.",
        ]

    
    if age is not None:
        base.insert(0, f"**Usia**: {age} bulan")

    return "\n".join(base + focus)


def generate_recommendation(diagnosis: dict[str, str], patient: dict[str, Any]) -> LLMRecommendation:
    """
    Turns diagnose() output into a practical recommendation.
    - Uses RAG (Chroma + LangChain) over guideline docs.
    - Falls back to deterministic rule-based text if RAG/LLM is unavailable.
    """
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
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

    if not api_key:
        return LLMRecommendation(
            diagnosis=diagnosis,
            recommendation_markdown=_fallback_recommendation(diagnosis, patient),
            model_used="fallback",
            used_fallback=True,
        )

    try:
        maybe_auto_build_index(gemini_api_key=api_key)
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

