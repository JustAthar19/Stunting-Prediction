from fastapi import FastAPI, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates 
from pydantic import BaseModel
from src.pipeline.predict_pipeline import InputData, PredictPipeline


class PatientData(BaseModel):
    umur: int 
    jenis_kelamin:int
    tinggi_badan:float
    

class OutputPred(BaseModel):
    stunting_prediction: str

# Initialize the FastAPI app.
app = FastAPI()

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
    
    
