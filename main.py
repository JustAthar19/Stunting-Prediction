from fastapi import FastAPI, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates 
from pydantic import BaseModel
from src.pipeline.predict_pipeline import InputData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline

class CustomerData(BaseModel):
    customer_age: int 
    gender: str 
    dependent_count: int 
    education_level: str  
    marital_status: str 
    income_category: str
    card_category: str
    months_on_book: int
    total_relationship_count: int
    months_inactive_12_mon: int
    contacts_count_12_mon: int
    credit_limit: int
    total_revolving_bal: int
    total_amt_chng_q4_q1: float
    total_trans_amt: int
    total_trans_ct: int
    total_ct_chng_q4_q1:float
    avg_utilization_ratio: float

class ChurnPrediction(BaseModel):
    churn_prediction: str

# Initialize the FastAPI app.
app = FastAPI()

app.mount("/templates", StaticFiles(directory='templates'), name='templates')
app.mount("/static", StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory="templates")


@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )


@app.post("/training/")
def train_model():
    print("Training model...")
    train_pipeline = TrainPipeline()
    train_pipeline.train()
    print("Model training completed successfully.")
    return Response(content="Model training completed successfully.", status_code=200)

@app.post("/predict/")
def predict(customer_data: CustomerData):
    
    input_data = InputData(
        customer_age=customer_data.customer_age,
        gender=customer_data.gender,
        dependent_count=customer_data.dependent_count,
        education_level=customer_data.education_level,
        marital_status=customer_data.marital_status,
        income_category=customer_data.income_category,
        card_category=customer_data.card_category,
        months_on_book=customer_data.months_on_book,
        total_relationship_count=customer_data.total_relationship_count,
        months_inactive_12_mon=customer_data.months_inactive_12_mon,
        contacts_count_12_mon=customer_data.contacts_count_12_mon,
        credit_limit=customer_data.credit_limit,
        total_revolving_bal=customer_data.total_revolving_bal,
        total_amt_chng_q4_q1=customer_data.total_amt_chng_q4_q1,
        total_trans_ct=customer_data.total_trans_ct,
        total_trans_amt=customer_data.total_trans_amt,
        total_ct_chng_q4_q1=customer_data.total_ct_chng_q4_q1,
        avg_utilization_ratio=customer_data.avg_utilization_ratio
    )
    
    input_df = input_data.get_input_data_df()
    predict_pipeline = PredictPipeline()

    prediction = predict_pipeline.predict(input_df)
    
    return ChurnPrediction(churn_prediction=prediction)

