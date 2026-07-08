# Models

This directory contains the trained machine learning model used by the Customer Lifetime Value (CLV) Prediction API.

---

# Directory Structure

```text
models/
├── clv_model.pkl
└── README.md
```

---

# Files

## clv_model.pkl

This file contains the trained machine learning pipeline used by the API.

The pipeline consists of:

```text
Pipeline
├── StandardScaler
└── XGBRegressor
```

### Pipeline Components

| Component | Description |
|-----------|-------------|
| StandardScaler | Standardizes numerical features before prediction |
| XGBRegressor | Predicts the 12-month Customer Lifetime Value (CLV) |

The preprocessing step is included inside the pipeline, so no additional scaler file is required.

---

# Model Information

| Property | Value |
|----------|-------|
| Model Type | Scikit-learn Pipeline |
| Algorithm | XGBoost Regressor |
| Preprocessing | StandardScaler |
| Task | Regression |
| Prediction | Customer Lifetime Value (CLV) |

---

# Loading the Model

```python
import joblib

model = joblib.load("models/clv_model.pkl")
```

---

# Prediction Example

```python
import pandas as pd
import joblib

model = joblib.load("models/clv_model.pkl")

customer = pd.DataFrame({
    "Recency":[20],
    "Frequency":[8],
    "Monetary":[550],
    "AvgOrderValue":[68.75],
    "PurchaseFrequency":[8]
})

prediction = model.predict(customer)

print(prediction)
```

---

# Workflow

```text
Customer Data
      │
      ▼
StandardScaler
      │
      ▼
XGBoost Regressor
      │
      ▼
Predicted Customer Lifetime Value
```

---

# Notes

- This model is loaded by the FastAPI application during server startup.
- The input features must match the same format used during training.
- The model is stored as a serialized binary file (`.pkl`) and should not be edited manually.
- Use `joblib.load()` to load the model.

---

# Purpose

The trained model predicts the estimated **12-month Customer Lifetime Value (CLV)** of a customer based on historical purchasing behavior. It is optimized for deployment through FastAPI to provide real-time predictions.
