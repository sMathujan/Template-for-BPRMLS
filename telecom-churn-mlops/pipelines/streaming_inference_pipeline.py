"""Streaming inference pipeline with MLflow monitoring (template's InferenceTracker pattern).

    python pipelines/streaming_inference_pipeline.py              # one sample customer
    python pipelines/streaming_inference_pipeline.py --stream 50  # replay 50 records
    python pipelines/streaming_inference_pipeline.py --batch-file new_customers.csv
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mlflow
import numpy as np
import pandas as pd

from src.model_inference import ModelInference
from utils.config import get_columns, get_inference_config, get_path, resolve_path
from utils.logger import banner, get_logger
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags

logger = get_logger(__name__)

SAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": "29.85",
}


class InferenceTracker:
    """Buffers predictions and logs batch-level monitoring metrics to MLflow."""

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.buffer: List[Dict[str, Any]] = []
        self.tracker = None
        self.batch_index = 0

    def start(self, model_name: str, threshold: float) -> None:
        try:
            self.tracker = MLflowTracker()
            self.tracker.start_run("streaming_inference", create_mlflow_run_tags(
                "inference_pipeline", {"inference_type": "streaming", "model_name": model_name,
                                       "threshold": threshold, "batch_size": self.batch_size}))
        except Exception as e:
            logger.warning(f"⚠ Inference tracking disabled: {e}")
            self.tracker = None

    def track(self, record: Dict[str, Any], result: Dict[str, Any], latency_s: float) -> None:
        self.buffer.append({"timestamp": datetime.now().isoformat(), "input": record,
                            "result": result, "latency_ms": latency_s * 1000})
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer or self.tracker is None:
            self.buffer = []
            return
        probs = np.array([b["result"]["churn_probability"] for b in self.buffer])
        preds = np.array([b["result"]["prediction"] for b in self.buffer])
        lat = np.array([b["latency_ms"] for b in self.buffer])
        bands = pd.Series([b["result"]["risk_band"] for b in self.buffer]).value_counts()
        MLflowTracker.log_metrics_safe({
            "batch_size": len(self.buffer), "avg_latency_ms": lat.mean(),
            "p95_latency_ms": np.percentile(lat, 95), "avg_churn_probability": probs.mean(),
            "std_churn_probability": probs.std(), "flagged_for_retention": int(preds.sum()),
            "flag_rate": preds.mean(), "high_risk": int(bands.get("High", 0)),
            "medium_risk": int(bands.get("Medium", 0)), "low_risk": int(bands.get("Low", 0)),
        }, step=self.batch_index)
        out_dir = os.path.join(get_path("artifacts_dir"), "inference_batches")
        os.makedirs(out_dir, exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{self.batch_index}"
        path = os.path.join(out_dir, f"inference_batch_{stamp}.json")
        with open(path, "w") as f:
            json.dump(self.buffer, f, indent=2, default=str)
        mlflow.log_artifact(path, "inference_batches")
        logger.info(f"✓ Logged inference batch {self.batch_index} ({len(self.buffer)} predictions)")
        self.batch_index += 1
        self.buffer = []

    def end(self) -> None:
        self.flush()
        if self.tracker:
            self.tracker.end_run()


def initialize_inference_system() -> ModelInference:
    return ModelInference()


def streaming_inference(inference: ModelInference, data: Dict[str, Any],
                        tracker: InferenceTracker = None) -> Dict[str, Any]:
    if inference is None:
        raise ValueError("ModelInference instance cannot be None")
    start = time.time()
    result = inference.predict(data)
    if tracker:
        tracker.track(data, result, time.time() - start)
    return result


def batch_inference(inference: ModelInference, csv_path: str, save_path: str) -> pd.DataFrame:
    banner(logger, f"BATCH INFERENCE - {csv_path}")
    df = pd.read_csv(csv_path)
    target = get_columns()["target"]
    preds = inference.predict_batch(df.drop(columns=[target], errors="ignore"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    preds.to_csv(save_path, index=False)
    logger.info(f"✓ Predictions saved → {save_path}")
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn inference")
    parser.add_argument("--stream", type=int, default=0,
                        help="Replay N random customers from the raw data as a stream")
    parser.add_argument("--batch-file", type=str, help="Score a CSV of customers")
    args = parser.parse_args()

    inf_cfg = get_inference_config()
    inference = initialize_inference_system()

    if args.batch_file:
        out = batch_inference(inference, args.batch_file, resolve_path(inf_cfg["save_path"]))
        print(out.head(10).to_string(index=False))
        sys.exit(0)

    tracker = InferenceTracker(batch_size=inf_cfg.get("batch_size", 100))
    tracker.start(inference.model_name, inference.threshold)
    try:
        if args.stream > 0:
            raw = pd.read_csv(resolve_path(inf_cfg["data_path"]))
            target = get_columns()["target"]
            sample = raw.sample(n=min(args.stream, len(raw)), random_state=None)
            for record in sample.drop(columns=[target]).to_dict(orient="records"):
                r = streaming_inference(inference, record, tracker)
                print(f"{r['customerID']:<12} p={r['churn_probability']:.3f}  "
                      f"{r['status']:<7} risk={r['risk_band']:<6} → {r['recommended_action']}")
        else:
            print(json.dumps(streaming_inference(inference, SAMPLE_CUSTOMER, tracker), indent=2))
    finally:
        tracker.end()
