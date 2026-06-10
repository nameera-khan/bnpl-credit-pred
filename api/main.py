import time
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager

from monitoring.drift_detector import check_drift
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

model        = None
explainer    = None
feature_cols = None
threshold    = 0.35

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, feature_cols, threshold
    logger.info("Loading model artifacts...")
    model        = joblib.load("model/lgbm_calibrated.pkl")
    explainer    = joblib.load("model/shap_explainer.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    threshold    = float(joblib.load("model/threshold.pkl"))
    logger.info(f"Model loaded. Threshold: {threshold:.4f}")
    yield

app = FastAPI(title="BNPL Credit Risk Scorer", version="1.0.0", lifespan=lifespan)
prediction_log = []

class CreditApplication(BaseModel):
    amt_credit:              float = Field(..., gt=0)
    amt_annuity:             float = Field(..., gt=0)
    amt_income_total:        float = Field(..., gt=0)
    amt_goods_price:         float = Field(0.0)
    age_years:               float = Field(..., gt=18, lt=100)
    years_employed:          float = Field(0.0)
    ext_source_1:            Optional[float] = None
    ext_source_2:            Optional[float] = None
    ext_source_3:            Optional[float] = None
    is_male:                 int = Field(0)
    is_cash_loan:            int = Field(0)
    education_encoded:       int = Field(0)
    bureau_loan_count:       int = Field(0)
    bureau_active_loans:     int = Field(0)
    bureau_total_debt:       float = Field(0.0)
    bureau_overdue_count:    int = Field(0)
    prev_app_count:          int = Field(0)
    prev_approval_rate:      float = Field(0.0)
    install_late_count:      int = Field(0)
    install_avg_delay_days:  float = Field(0.0)
    class Config:
        extra = "allow"

class ScoreResponse(BaseModel):
    decision:               str
    default_probability:    float
    risk_tier:              str
    credit_limit_factor:    float
    top_risk_factors:       list
    latency_ms:             float

def assemble_features(app: CreditApplication):
    data = app.model_dump()
    data["credit_income_ratio"]  = data["amt_credit"]  / max(data["amt_income_total"], 1)
    data["annuity_income_ratio"] = data["amt_annuity"] / max(data["amt_income_total"], 1)
    data["credit_goods_ratio"]   = data["amt_credit"]  / max(data.get("amt_goods_price", 1) or 1, 1)
    data["ext_source_mean"]      = np.mean([data.get(f"ext_source_{i}") or 0 for i in [1,2,3]])
    data["ext_source_nulls"]     = sum(1 for i in [1,2,3] if data.get(f"ext_source_{i}") is None)
    data["is_unemployed"]        = 1 if data.get("years_employed", 0) <= 0 else 0
    for k, v in data.items():
        if v is None:
            data[k] = 0.0
    row = [float(data.get(col, 0.0)) for col in feature_cols]
    return np.array([row], dtype=np.float32)

@app.post("/score", response_model=ScoreResponse)
async def score(application: CreditApplication, bg: BackgroundTasks):
    t0   = time.perf_counter()
    X    = assemble_features(application)
    prob = float(model.predict_proba(X)[0][1])

    X_df      = pd.DataFrame(X, columns=feature_cols)
    shap_vals = explainer.shap_values(X_df, check_additivity=False)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    pairs     = sorted(zip(feature_cols, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
    top       = [{"feature": f, "impact": round(float(v), 4),
                  "direction": "increases_risk" if v > 0 else "decreases_risk"}
                 for f, v in pairs]

    tier = ("very_high" if prob >= 0.6 else "high" if prob >= 0.4
            else "medium" if prob >= 0.2 else "low")
    decision = ("APPROVE"       if prob < threshold * 0.7 else
                "DECLINE"       if prob > threshold * 1.5 else "MANUAL_REVIEW")
    clf = {"low": 1.0, "medium": 0.75, "high": 0.40, "very_high": 0.0}[tier]

    record = application.model_dump()
    record["prob"] = prob
    prediction_log.append(record)
    if len(prediction_log) % 500 == 0:
        bg.add_task(run_drift_check)

    latency = (time.perf_counter() - t0) * 1000
    logger.info(f"/score | decision={decision} prob={prob:.4f} latency={latency:.1f}ms")

    return ScoreResponse(
        decision=decision, default_probability=round(prob, 4),
        risk_tier=tier, credit_limit_factor=clf,
        top_risk_factors=top, latency_ms=round(latency, 2)
    )

@app.get("/health")
async def health():
    return {"status": "ok", "predictions_served": len(prediction_log),
            "threshold": round(threshold, 4)}

@app.get("/drift")
async def drift():
    if len(prediction_log) < 100:
        return {"status": "insufficient_data", "n": len(prediction_log)}
    current = pd.DataFrame(prediction_log)
    cols    = [c for c in feature_cols if c in current.columns]
    return check_drift(current[cols])

def run_drift_check():
    if len(prediction_log) < 100:
        return
    current = pd.DataFrame(prediction_log)
    cols    = [c for c in feature_cols if c in current.columns]
    result  = check_drift(current[cols])
    if result.get("should_retrain"):
        logger.warning("Drift detected — retraining recommended")
