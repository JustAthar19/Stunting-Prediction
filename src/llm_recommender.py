from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

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
        "### Langkah berikutnya (umum)",
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

    safety = [
        "",
        "### Catatan penting",
        "- **Ini bukan diagnosis medis final**. Jika anak tampak lemas, tidak mau minum, muntah terus, atau ada tanda bahaya lain, segera cari pertolongan medis.",
    ]

    if age is not None:
        base.insert(0, f"**Usia**: {age} bulan")

    return "\n".join(base + focus + safety)


def _gemini_generate_content(prompt: str, *, model: str, api_key: str, timeout_s: int = 30) -> str:
    """
    Minimal HTTP client for Google Gemini (Generative Language API) without extra dependencies.
    Expects GEMINI_API_KEY in env.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 700,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "x-goog-api-key" : api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    j = json.loads(raw)

    # Typical shape:
    # { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
    candidates = j.get("candidates") or []
    if candidates:
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        texts: list[str] = []
        for p in parts:
            t = (p or {}).get("text")
            if isinstance(t, str):
                texts.append(t)
        return "\n".join(texts).strip()

    return ""


def generate_recommendation(diagnosis: dict[str, str], patient: dict[str, Any]) -> LLMRecommendation:
    """
    Turns diagnose() output into a practical recommendation.
    - Uses Gemini if GEMINI_API_KEY is set.
    - Falls back to a deterministic rule-based recommendation otherwise.
    """
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    api_key = os.getenv("GEMINI_API_KEY")

    language = (patient.get("language") or "id").lower()
    allergies = patient.get("allergies") or []
    preferences = patient.get("preferences") or []
    notes = patient.get("notes") or ""

    # Keep prompt compact and grounded in dx + age/anthro.
    prompt = f"""
Anda adalah asisten edukasi gizi anak. Buat rekomendasi langkah berikutnya untuk mencegah/mengatasi stunting.
Gunakan bahasa {"Indonesia" if language.startswith("id") else "English"} yang sederhana, praktis, dan empatik.

DATA PASIEN:
- usia (bulan): {patient.get("age_months")}
- jenis kelamin (0=perempuan, 1=laki-laki): {patient.get("sex")}
- berat (kg): {patient.get("weight_kg")}
- tinggi/panjang (cm): {patient.get("height_cm")}
- alergi: {allergies}
- preferensi: {preferences}
- catatan: {notes}

HASIL (WHO):
- Height per Age: {diagnosis.get("Height per Age")}
- Weight per Age: {diagnosis.get("Weight per Age")}
- Weight per Height: {diagnosis.get("Weight per Height")}

OUTPUT YANG DIMINTA (format Markdown):
1) Ringkasan hasil (1-2 kalimat).
2) Langkah aksi 7 hari ke depan (bullet).
3) Menu/ide makanan (bullet), fokus protein hewani & variasi lokal.
4) Tanda bahaya kapan harus ke tenaga kesehatan (bullet).
Jangan berikan dosis obat. Jika wasting/SAM, tekankan perlu evaluasi tenaga kesehatan.
""".strip()
    if not api_key:
        print("print key not available")
        return LLMRecommendation(
            diagnosis=diagnosis,
            recommendation_markdown=_fallback_recommendation(diagnosis, patient),
            model_used="fallback",
            used_fallback=True,
        )
    # start = time.time()
    # text = _gemini_generate_content(prompt, model=model, api_key=api_key)
    # if not text:
    #     raise RuntimeError("Empty LLM response")
    # _ = time.time() - start
    # return LLMRecommendation(
    #         diagnosis=diagnosis,
    #         recommendation_markdown=text.strip(),
    #         model_used=model,
    #         used_fallback=False,
    # )

    try:
        start = time.time()
        text = _gemini_generate_content(prompt, model=model, api_key=api_key)
        if not text:
            raise RuntimeError("Empty LLM response")
        _ = time.time() - start
        return LLMRecommendation(
            diagnosis=diagnosis,
            recommendation_markdown=text.strip(),
            model_used=model,
            used_fallback=False,
        )
    except Exception as e:
        print(f"error: {e}")
        return LLMRecommendation(
            diagnosis=diagnosis,
            recommendation_markdown=_fallback_recommendation(diagnosis, patient),
            model_used=model,
            used_fallback=True,
        )

