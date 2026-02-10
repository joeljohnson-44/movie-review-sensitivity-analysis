from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from typing import List
import os

# Load trained model
from typing import List
import pandas as pd
from src import model_utils

MODEL_PATH = os.path.join("src", "models","distilbert_model")
model = model_utils.TransformerSentimentModel()
model.load(MODEL_PATH)

# ---------- FastAPI App with Swagger Metadata ----------

app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API for sentiment analysis using trained ML model",
    version="1.0.0",
    contact={
        "name": "J44",
        "email": "joeljohnson1207@gmail.com"
    }
)


# ---------- Request Models ----------

class SinglePredictionRequest(BaseModel):
    text: str = Field(
        ...,
        example="I absolutely loved this movie!"
    )


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        example=[
            "This product is amazing",
            "Worst experience ever",
            "Pretty decent overall"
        ]
    )


# ---------- Response Models ----------

class PredictionResponse(BaseModel):
    text: str
    sentiment: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------- Utility Function ----------

def predict_texts(texts: List[str]) -> List[str]:
    df = pd.Series(texts)
    preds = model.predict(df)

    label_map = {1: "positive", 0: "negative"}

    return [label_map.get(int(p), "unknown") for p in preds]


# ---------- Endpoints ----------

@app.get(
    "/",
    tags=["General"]
)
def home():
    return {
        "message": "Sentiment Analysis API is running",
        "docs": "/docs",
        "health": "/health"
    }


# ---- Health Check Endpoint ----
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health Check Endpoint"
)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


# ---- Single Prediction ----
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict sentiment for single text"
)
def predict_single(request: SinglePredictionRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty"
        )

    sentiment = predict_texts([request.text])[0]

    return PredictionResponse(
        text=request.text,
        sentiment=sentiment
    )


# ---- Batch Prediction ----
@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict sentiment for multiple texts"
)
def predict_batch(request: BatchPredictionRequest):

    texts = request.texts

    if not texts:
        raise HTTPException(
            status_code=400,
            detail="Input list cannot be empty"
        )

    # Optional safety limit
    if len(texts) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum allowed is 1000"
        )

    sentiments = predict_texts(texts)

    results = [
        PredictionResponse(text=t, sentiment=s)
        for t, s in zip(texts, sentiments)
    ]

    return BatchPredictionResponse(predictions=results)


# ---- Model Info Endpoint ----
@app.get(
    "/model_info",
    tags=["Monitoring"],
    summary="Get model information"
)
def model_info():
    return {
        "model_type": type(model).__name__,
        "description": "Bert based Transformer Model"
    }
