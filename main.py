from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd 
from src.pipeline.predict_pipeline import InputData, PredictPipeline
import math
from src.llm_recommender import generate_recommendation, LLMRecommendation
from typing import List, Optional

class PatientData(BaseModel):
    umur: int 
    jenis_kelamin:int
    tinggi_badan:float
    

class OutputPred(BaseModel):
    stunting_prediction: str


class RecommendationRequest(BaseModel):
    sex: int
    age: int
    weight: float
    height: float
    # Optional context for better recommendations
    language: Optional[str] = "id"
    allergies: Optional[List[str]] = None
    preferences: Optional[List[str]] = None
    notes: Optional[str] = None


# Initialize the FastAPI app.
app = FastAPI()

# --Flag
def who_z_score(height, L, M, S):
    if L != 0:
        return(height/M ** L - 1) / (L * S)
    else:
        return math.log(height/M)/S

def who_haz(sex:int, age:int, height:float):
    if sex == 0:
        data = pd.read_csv("data/height-Age/Monthly-girls-height-z-score.csv")
    else:
        data = pd.read_csv("data/height-Age/Monthly-boys-height-z-score.csv")
    row = data[data['Month']==age]
    z = who_z_score(height,
                    row["L"].iloc[0],
                    row["M"].iloc[0], 
                    row["S"].iloc[0])
    
    if z < -3:
        return "Severly Stunted"
    elif z >= -3 and z < -2:
        return "Stunted"
    elif z >= 2 and z <= 3:
        return "Normal"
    else:
        return "Vary Tall"
    

def who_waz(sex:int, age:int, weight:float):
    if sex == 0:
        data = pd.read_csv("data/weight-Age/Monthly-girls-weight-z-score.csv")
    else: 
        data = pd.read_csv("data/weight-Age/Monthly-boys-weight-z-score.csv")
    row = data[data['Month']==age]
    z = who_z_score(weight,
                    row["L"].iloc[0],
                    row["M"].iloc[0], 
                    row["S"].iloc[0])
    
    if z < -3:
        return "Severly underweight"
    elif z >= -3 and z < -2:
        return "underweight"
    elif z >= 2 and z <= 3:
        return "Normal"
    else:
        return "Very Tall"
    
    return z

def who_whz(sex: int, weight:float, length: float):
    if sex == 0:
        data = pd.read_excel("data/weight-Height/girls-zscore-weight-height.xlsx")
    else: 
        data = pd.read_excel("data/weight-Height/boys-zscore-weight-height-table.xlsx")
    row = data[data['Length']==length]
    z = who_z_score(weight,
                    row["L"].iloc[0],
                    row["M"].iloc[0], 
                    row["S"].iloc[0])
    
    if z < -3:
        return "Severly Wasting (SAM)"
    elif z >= -3 and z < -2:
        return "Moderate Wasting"
    elif z >= 2 and z <= 1:
        return "Normal"
    elif z > 1 and  z <= 2:
        return "Risk of Overweight"
    elif z > 2 and z <= 3:
        return "Overweight"
    else: return "Obesity"


@app.post("/diagnose")
def diagnose(sex: int, age: int, weight: float, height: float):
    return {
        "Height per Age" : who_haz(sex, age, height),
        "Weight per Age" : who_waz(sex, age, weight),
        "Weight per Height" : who_whz(sex, weight, height)
    }


@app.post("/recommendation", response_model=LLMRecommendation)
def recommendation(req: RecommendationRequest):
    dx = diagnose(sex=req.sex, age=req.age, weight=req.weight, height=req.height)
    return generate_recommendation(
        diagnosis=dx,
        patient={
            "sex": req.sex,
            "age_months": req.age,
            "weight_kg": req.weight,
            "height_cm": req.height,
            "language": req.language,
            "allergies": req.allergies,
            "preferences": req.preferences,
            "notes": req.notes,
        },
    )


@app.post("/predict/")
def predict(patient_data: PatientData):
    
    input_data = InputData(
        umur=patient_data.umur,
        jenis_kelamin=patient_data.jenis_kelamin,
        tinggi_badan=patient_data.tinggi_badan
    )


    input_data = input_data.get_input_data_df()
    pipeline = PredictPipeline()

    pred = pipeline.predict(input_data)

    return OutputPred(stunting_prediction=pred)


if __name__ == "__main__":
    print(who_haz(0, 0, 49))
    
