from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd 
from src.pipeline.predict_pipeline import InputData, PredictPipeline
import math

class PatientData(BaseModel):
    umur: int 
    jenis_kelamin:int
    tinggi_badan:float
    

class OutputPred(BaseModel):
    stunting_prediction: str





# Initialize the FastAPI app.
app = FastAPI()

# --Flag
def who_z_score(height, L, M, S):
    if L != 0:
        return(height/M ** L - 1) / (L * S)
    else:
        return math.log(height/M)/S

@app.post("/count_z_score/")
def who_haz(sex, age, height):
    if sex == 1:
        data = pd.read_excel("data/height-Age/boys-zscore-high-tables.xlsx")
    else: 
        data = pd.read_excel("data/height-Age/girls-zscore-high-tables.xlsx")
    row = data[data['Day']==age]
    z = who_z_score(height,
                    row["L"].iloc[0],
                    row["M"].iloc[0], 
                    row["S"].iloc[0])
    
    if z < -3:
        print("Severly Stunted")
    elif z >= -3 and z < -2:
        print("Stunted")
    elif z >= 2 and z <= 3:
        print("Normal")
    else:
        print("Vary Tall")
    
    return z

def who_waz(sex, age, height):
    if sex == 1:
        data = pd.read_excel("data/weight-Age/boys-zscore-high-tables.xlsx")
    else: 
        data = pd.read_excel("data/weight-Age/girls-zscore-high-tables.xlsx")
    row = data[data['Day']==age]
    z = who_z_score(height,
                    row["L"].iloc[0],
                    row["M"].iloc[0], 
                    row["S"].iloc[0])
    
    if z < -3:
        print("Severly underweight")
    elif z >= -3 and z < -2:
        print("underweight")
    elif z >= 2 and z <= 3:
        print("Normal")
    else:
        print("Very Tall")
    
    return z
    
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
    print(who_haz(0, 12, 40))
    
