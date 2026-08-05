from fastapi import FastAPI
from api.schemas import CreditInput, CreditOutput
from api.model import predict_default

app = FastAPI(title="Credit Risk API")

@app.get("/")
def root():
    return {'message': 'Credit Risk API is running'}

@app.post("/predict", response_model=CreditOutput)
def predict(input: CreditInput):
    result = predict_default(input.model_dump())
    return result