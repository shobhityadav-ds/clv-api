# Customer Lifetime Value (CLV) Prediction API

A Machine Learning API that predicts the 12-month Customer Lifetime Value for any customer using XGBoost, deployed with FastAPI on Render.com.

---

## Live API

```
https://clv-api.onrender.com
```

Swagger UI (Test the API):
```
https://clv-api.onrender.com/docs
```

---

## Project Overview

This project predicts how much revenue a customer will generate over the next 12 months. It helps businesses identify high-value customers, reduce churn, and allocate marketing budgets more effectively.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | Preprocessing pipeline |
| XGBoost | ML model |
| Lifetimes | BG/NBD + Gamma-Gamma statistical CLV model |
| MLflow | Experiment tracking |
| FastAPI | REST API |
| Uvicorn | ASGI server |
| Render.com | Free cloud deployment |
| Google Colab | Model training environment |

---

## Dataset

**UCI Online Retail Dataset**
- Real transactional data from a UK-based online store
- Period: 2010 to 2011
- Customers: 4,300+
- Transactions: 500,000+
- Source: https://archive.ics.uci.edu/ml/datasets/Online+Retail

---

## Features Used (RFM + Extended)

| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of orders placed |
| AvgOrderValue | Average spend per order |
| TotalItems | Total items purchased |
| UniqueProducts | Number of unique products bought |
| Tenure | Days since first purchase |
| PurchaseGap | Average days between orders |

---

## Customer Tiers

| Tier | Predicted CLV |
|---|---|
| Platinum | Above 5000 |
| Gold | 1000 to 4999 |
| Silver | 300 to 999 |
| Bronze | Below 300 |

---

## API Endpoints

### GET /
Returns API status and available endpoints.

**Response:**
```json
{
  "message": "CLV Prediction API is running!",
  "docs": "Visit /docs for Swagger UI",
  "endpoints": ["/predict", "/health"]
}
```

---

### POST /predict
Predicts 12-month CLV for a customer.

**Request Body:**
```json
{
  "Recency": 30,
  "Frequency": 12,
  "AvgOrderValue": 85.5,
  "TotalItems": 200,
  "UniqueProducts": 45,
  "Tenure": 365,
  "PurchaseGap": 30.4
}
```

**Response:**
```json
{
  "predicted_clv_12months": 1250.75,
  "currency": "GBP",
  "customer_tier": "Gold",
  "recommendation": "High value — priority support and loyalty rewards"
}
```

---

### GET /health
Returns API health status.

**Response:**
```json
{
  "status": "healthy",
  "model": "XGBoost CLV Predictor v1.0"
}
```

---

## Project Structure

```
clv-api/
├── app.py              # FastAPI application
├── clv_model.pkl       # Trained XGBoost model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Model Performance

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R2 | Explained Variance Score |

Model was trained and evaluated in Google Colab with MLflow experiment tracking.

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/clv-api.git
cd clv-api
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the API**
```bash
uvicorn app:app --reload
```

**4. Visit in browser**
```
http://localhost:8000/docs
```

---

## How to Test the API (Python)

```python
import requests

response = requests.post(
    "https://clv-api.onrender.com/predict",
    json={
        "Recency"       : 30,
        "Frequency"     : 12,
        "AvgOrderValue" : 85.5,
        "TotalItems"    : 200,
        "UniqueProducts": 45,
        "Tenure"        : 365,
        "PurchaseGap"   : 30.4
    }
)

print(response.json())
```

---

## Deployment

This API is deployed for free on **Render.com**.

| Field | Value |
|---|---|
| Build Command | pip install -r requirements.txt |
| Start Command | uvicorn app:app --host 0.0.0.0 --port 10000 |
| Instance Type | Free |

---

## Training Notebook

The full training pipeline was built in Google Colab and includes:

- Data cleaning and EDA
- RFM feature engineering
- BG/NBD and Gamma-Gamma statistical CLV model
- XGBoost ML pipeline
- Model evaluation (MAE, RMSE, R2)
- MLflow experiment tracking
- Customer tier segmentation
- Model export as .pkl file

---

## Business Use Cases

- Identify VIP customers for exclusive offers
- Reduce churn by targeting low CLV customers early
- Allocate marketing budget based on predicted value
- Segment customers for personalized campaigns
- Forecast future revenue by customer cohort

---

## Author

Built with Python, XGBoost, Lifetimes, FastAPI, and MLflow.

Trained on Google Colab. Deployed on Render.com.
