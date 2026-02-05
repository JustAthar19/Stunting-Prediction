from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from src.child_growth_standards import who_haz, who_waz, who_whz
from src.llm_recommender import LLMRecommendation, generate_recommendation

app = FastAPI(title="Stunting Prediction and Guideline-based Recommendation API")


class RecommendationRequest(BaseModel):
    sex: int
    age: int
    weight: float
    height: float


def diagnose(sex: int, age: int, weight: float, height: float):
    return {
        "Height per Age": who_haz(sex, age, height),
        "Weight per Age": who_waz(sex, age, weight),
        "Weight per Height": who_whz(sex, weight, height)
    }


@app.post("/rekomendasi", response_model=LLMRecommendation)
def recommendation(req: RecommendationRequest):
    dx = diagnose(sex=req.sex, age=req.age, weight=req.weight, height=req.height)
    return generate_recommendation(
        diagnosis=dx,
        patient={
            "sex": req.sex,
            "age_months": req.age,
            "weight_kg": req.weight,
            "height_cm": req.height,
        },
    )
