"""
CFPB complaint response predictor — single-file FastAPI service.
 
Run with:
    uvicorn cfpb_api:app --reload
 
Expects models/cfpb_predictor.joblib to be a dict with keys:
    model          : trained XGBClassifier
    encoder        : fitted LabelEncoder
 
That file is created by running xgboost_optuna.py.
"""

import os
from typing import Any
import joblib
import requests
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PREDICTOR_PATH = os.path.join(PROJECT_ROOT, "models", "cfpb_predictor.joblib")
_predictor = joblib.load(PREDICTOR_PATH)

_model = _predictor["model"]
_encoder = _predictor["encoder"] 

app = FastAPI(title="CFPB Response Predictor")

class InputData(BaseModel):
    product: str = Field(alias="Product")
    issue: str = Field(alias="Issue")
    company: str = Field(alias="Company")
    submitted_via: str = Field(alias="Submitted via")
    consumer_consent_provided: str = Field(alias="Consumer consent provided?")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "running",
        "docs": "/docs",
        "prediction_endpoint": "/predict"
    }
@app.post("/predict")
def predict(data: InputData) -> dict:
    """
    Predict the company response for a single complaint.

    Parameters:
    data :  features for predicting the response to a complaint

    Returns:
    predicition : likely response to the complaint
    """

    row = pd.DataFrame(
        [data.model_dump(by_alias =True)]
        ).astype("category")
    try:
        probabilities  = _model.predict_proba(row)[0]
        best_position = int(probabilities.argmax())
        encoded_class = _model.classes_[best_position]
        prediction = _encoder.inverse_transform(
            [encoded_class]
        )[0]

        confidence = float(probabilities[best_position])

    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"prediction": prediction,
            "confidence": confidence,}


aliases = [
    field.alias
    for field in InputData.model_fields.values()
]

