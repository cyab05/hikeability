from __future__ import annotations
from transformers import DistilBertForSequenceClassification

import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://35.202.68.106:5000/")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "trail-condition-distilbert-default")
MODEL_STAGE          = os.getenv("MODEL_STAGE", "latest")   # "latest" | "Production" etc.
PRETRAINED_TOKENIZER = "distilbert-base-uncased"
MAX_LENGTH           = 128
LABEL_ORDER          = ["hikeable", "modest_conditions", "not_hikeable"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global state ──────────────────────────────────────────────────────────────
state: dict = {"model": None, "tokenizer": None, "loaded": False, "error": None}


# ── Lifespan: load model once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3,
            id2label={0: "hikeable", 1: "modest_conditions", 2: "not_hikeable"},
            label2id={"hikeable": 0, "modest_conditions": 1, "not_hikeable": 2},
        ).to(DEVICE)
        state["model"].eval()
        state["tokenizer"] = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        state["loaded"] = True
        print(f"Model loaded on {DEVICE}")
    except Exception as e:
        state["error"] = str(e)
        print(f"WARNING: Could not load model — {e}")
    yield
    state.clear()


app = FastAPI(
    title="Trail Condition Classifier",
    description="Classifies hiking trail conditions from user-submitted comments.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    comment_text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"comment_text": "Trail is clear and dry. Great views at the summit!"}
            ]
        }
    }


class PredictResponse(BaseModel):
    comment_text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    error: Optional[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Trail Condition Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if state["loaded"] else "degraded",
        model_loaded=state["loaded"],
        device=str(DEVICE),
        model_name=REGISTERED_MODEL_NAME,
        error=state.get("error"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {state.get('error', 'unknown error')}",
        )

    tokenizer: DistilBertTokenizerFast = state["tokenizer"]
    model = state["model"]

    encodings = tokenizer(
        [request.comment_text],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            input_ids=encodings["input_ids"].to(DEVICE),
            attention_mask=encodings["attention_mask"].to(DEVICE),
        ).logits

    probs    = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    label    = model.config.id2label[pred_idx]

    id2label = model.config.id2label
    prob_map = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}

    return PredictResponse(
        comment_text=request.comment_text,
        label=label,
        confidence=round(probs[pred_idx], 4),
        probabilities=prob_map,
    )