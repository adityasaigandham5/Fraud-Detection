from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib, json, time
import numpy as np
import pandas as pd

app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time payment fraud detection",
    version     = "1.0.0"
)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model artifacts at startup
model     = joblib.load("../models/fraud_model.pkl")
features  = json.load(open("../models/feature_names.json"))
threshold = json.load(open("../models/optimal_threshold.json"))["threshold"]
metrics   = json.load(open("../models/metrics.json"))

class Transaction(BaseModel):
    Time:   float = Field(..., json_schema_extra={"example": 43200})
    V1:     float = Field(..., json_schema_extra={"example": 0.5})
    V2:     float = Field(..., json_schema_extra={"example": 0.3})
    V3:     float = Field(0.0, json_schema_extra={"example": 0.1})
    V4:     float = Field(0.0, json_schema_extra={"example": 0.2})
    V5:     float = Field(0.0, json_schema_extra={"example": 0.0})
    V6:     float = Field(0.0, json_schema_extra={"example": 0.1})
    V7:     float = Field(0.0, json_schema_extra={"example": 0.0})
    V8:     float = Field(0.0, json_schema_extra={"example": 0.0})
    V9:     float = Field(0.0, json_schema_extra={"example": 0.1})
    V10:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V11:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V12:    float = Field(0.0, json_schema_extra={"example": 0.1})
    V13:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V14:    float = Field(0.0, json_schema_extra={"example": 0.2})
    V15:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V16:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V17:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V18:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V19:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V20:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V21:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V22:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V23:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V24:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V25:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V26:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V27:    float = Field(0.0, json_schema_extra={"example": 0.0})
    V28:    float = Field(0.0, json_schema_extra={"example": 0.0})
    Amount: float = Field(..., json_schema_extra={"example": 100.0})

def build_features(txn: Transaction) -> pd.DataFrame:
    """Build feature vector matching training features exactly."""
    d = txn.dict()
    # Add engineered features (same as training)
    import numpy as np
    d['Amount_Log']  = float(np.log1p(d['Amount']))
    d['Is_Night']    = int((d['Time'] // 3600) % 24 >= 22 or (d['Time'] // 3600) % 24 <= 6)
    d['Is_Morning']  = int(6 <= (d['Time'] // 3600) % 24 < 12)
    d['Small_Txn']   = int(d['Amount'] < 50)
    d['Large_Txn']   = int(d['Amount'] > 500)
    d['Zero_Amount'] = int(d['Amount'] == 0)
    # Build DataFrame with exact feature order
    feat_df = pd.DataFrame([{f: d.get(f, 0.0) for f in features}])
    return feat_df

@app.get("/")
def root():
    return {
        "status":    "running",
        "model":     "Fraud Detection v1.0",
        "auc_roc":   metrics["auc_roc"],
        "threshold": threshold
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_fraud(txn: Transaction):
    start = time.time()
    try:
        feat_df = build_features(txn)
        prob    = float(model.predict_proba(feat_df)[0][1])
        is_fraud = bool(prob >= threshold)

        # Risk level logic
        if prob < 0.1:    risk, action = "LOW",    "APPROVE"
        elif prob < 0.4:  risk, action = "MEDIUM", "REVIEW"
        else:             risk, action = "HIGH",   "BLOCK"

        return {
            "fraud_probability":  round(prob, 4),
            "is_fraud":           is_fraud,
            "risk_level":         risk,
            "recommended_action": action,
            "threshold_used":     round(threshold, 4),
            "confidence":         round(max(prob, 1 - prob), 4),
            "latency_ms":         round((time.time() - start) * 1000, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))