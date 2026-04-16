from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

#ham = 0 spam =1 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "spam_model_nb_best.joblib")
model = joblib.load(model_path)

# 1. Define the data structure (Validation)
class EmailRequest(BaseModel):
    email: str

@app.post("/api/predict")
def predict(request: EmailRequest):
    # Get probabilities     
    prediction = model.predict(pd.Series([request.email]))[0] 
    proba = model.predict_proba(pd.Series([request.email]))[0]    
    
    return {
        "classification": "SPAM" if int(prediction) == 1 else "LEGIT",
        "confidence": round(float(max(proba)) * 100, 2),
        "indicators":["Keyword analysis", "Pattern matching"], 
        "summary":"AI-driven content analysis complete."
    }

@app.get("/api/health")
def health(): 
    return {"status":"alive"}

# Run with: uvicorn app:app --port 5001 --reload (locally)