from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("clv_model.pkl")

app = FastAPI(
    title="CLV Prediction API",
    description="Customer Lifetime Value Prediction using XGBoost",
    version="1.0.0"
)

# Input schema
class CustomerData(BaseModel):
    Recency       : float  # Days since last purchase
    Frequency     : float  # Number of orders
    AvgOrderValue : float  # Average order value
    TotalItems    : float  # Total items purchased
    UniqueProducts: float  # Unique products bought
    Tenure        : float  # Days as a customer
    PurchaseGap   : float  # Avg days between orders

def assign_tier(clv):
    if   clv >= 5000: return "Platinum"
    elif clv >= 1000: return "Gold"
    elif clv >= 300 : return "Silver"
    else            : return "Bronze"

@app.get("/")
def root():
    return {
        "message"  : "CLV Prediction API is running!",
        "docs"     : "Visit /docs for Swagger UI",
        "endpoints": ["/predict", "/health"]
    }

@app.post("/predict")
def predict_clv(customer: CustomerData):
    features = pd.DataFrame([customer.dict()])
    log_clv  = model.predict(features)[0]
    clv      = round(float(np.expm1(log_clv)), 2)
    tier     = assign_tier(clv)

    if   clv >= 5000: recommendation = "VIP customer — offer exclusive deals and personal account manager"
    elif clv >= 1000: recommendation = "High value — priority support and loyalty rewards"
    elif clv >= 300 : recommendation = "Mid value — nurture with targeted email campaigns"
    else            : recommendation = "Low value — re-engage with discount offers"

    return {
        "predicted_clv_12months": clv,
        "currency"              : "GBP",
        "customer_tier"         : tier,
        "recommendation"        : recommendation
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model" : "XGBoost CLV Predictor v1.0"
    }
