"""REST API for real-time churn scoring.

    uvicorn api.main:app --host 0.0.0.0 --port 8000      (or: make serve)
    Interactive docs: http://localhost:8000/docs
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from fastapi import FastAPI, HTTPException

from api.schemas import BatchRequest, BatchResponse, CustomerRecord, PredictionResponse
from src.model_inference import ModelInference
from utils.logger import get_logger

logger = get_logger(__name__)
STATE: Dict[str, Any] = {"inference": None, "load_error": None}


def _load_model() -> None:
    try:
        STATE["inference"] = ModelInference()
        STATE["load_error"] = None
    except Exception as e:  # keep the API up so /health can report the problem
        STATE["inference"], STATE["load_error"] = None, str(e)
        logger.error(f"Model failed to load: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Telecom Customer Churn API",
              description="Ensemble churn model with a validation-selected decision threshold.",
              version="1.0.0", lifespan=lifespan)


def _inference() -> ModelInference:
    if STATE["inference"] is None:
        raise HTTPException(status_code=503,
                            detail=f"Model not loaded: {STATE['load_error']}. Run `make train-pipeline`.")
    return STATE["inference"]


def _to_response(row: Dict[str, Any]) -> PredictionResponse:
    clean = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
    clean["churn_probability"] = round(float(clean["churn_probability"]), 4)
    return PredictionResponse(**{k: clean[k] for k in PredictionResponse.model_fields})


@app.get("/health")
def health():
    ok = STATE["inference"] is not None
    return {"status": "ok" if ok else "degraded", "model_loaded": ok, "error": STATE["load_error"]}


@app.get("/model-info")
def model_info():
    meta = _inference().metadata
    return {k: meta.get(k) for k in ("model_name", "model_version", "threshold", "threshold_metric",
                                     "trained_at", "mlflow_run_id", "test_metrics",
                                     "test_f1_ci", "business_economics")}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord):
    try:
        return _to_response(_inference().predict(record.model_dump()))
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    inf = _inference()
    try:
        df = inf.predict_batch([r.model_dump() for r in request.records])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    preds = [_to_response(r) for r in df.to_dict(orient="records")]
    return BatchResponse(n_records=len(preds), n_flagged=int(df["prediction"].sum()),
                         predictions=preds)


@app.post("/reload")
def reload_model():
    """Hot-reload after retraining without restarting the server."""
    _load_model()
    return health()
