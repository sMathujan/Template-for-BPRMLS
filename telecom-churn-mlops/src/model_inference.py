"""Model inference: raw customer record(s) → churn probability → frozen-threshold decision.

Example input (raw schema, same as the source CSV minus 'Churn'):
{
  "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
  "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85, "TotalCharges": "29.85"
}
"""
import json
import os
import time
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import ChurnPreprocessor
from utils.config import get_columns, get_inference_config, get_model_config, resolve_path
from utils.logger import banner, get_logger

logger = get_logger(__name__)


class ModelInference:
    def __init__(self, model_path: str = None, metadata_path: str = None):
        banner(logger, "INITIALISING MODEL INFERENCE")
        model_cfg = get_model_config()
        self.model_path = model_path or resolve_path(model_cfg["model_path"])
        self.metadata_path = metadata_path or resolve_path(model_cfg["metadata_path"])
        self.id_column = get_columns().get("id_column", "customerID")
        self.risk_bands = get_inference_config().get("risk_bands", {"high": 0.7, "medium": 0.4})

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. "
                                    f"Run `make train-pipeline` first.")
        self.model = joblib.load(self.model_path)
        with open(self.metadata_path) as f:
            self.metadata: Dict[str, Any] = json.load(f)
        self.threshold: float = float(self.metadata["threshold"])
        self.model_name: str = self.metadata.get("model_name", type(self.model).__name__)
        self.preprocessor = ChurnPreprocessor().load()

        logger.info(f"✓ Model: {self.model_name} | frozen threshold = {self.threshold:.2f} | "
                    f"version = {self.metadata.get('model_version', 'n/a')}")

    def _risk_band(self, p: float) -> str:
        if p >= self.risk_bands["high"]:
            return "High"
        if p >= self.risk_bands["medium"]:
            return "Medium"
        return "Low"

    def predict_batch(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
        if df.empty:
            raise ValueError("No records supplied for prediction")
        ids = df[self.id_column].tolist() if self.id_column in df.columns else [None] * len(df)

        start = time.time()
        X = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X)[:, 1]
        elapsed_ms = (time.time() - start) * 1000

        pred = (proba >= self.threshold).astype(int)
        out = pd.DataFrame({
            self.id_column: ids,
            "churn_probability": np.round(proba, 4),
            "prediction": pred,
            "status": np.where(pred == 1, "Churn", "Retain"),
            "risk_band": [self._risk_band(p) for p in proba],
            "recommended_action": np.where(pred == 1, "Send retention offer", "No action"),
            "threshold": self.threshold,
            "model_name": self.model_name,
        })
        logger.info(f"✓ Scored {len(out)} record(s) in {elapsed_ms:.1f} ms "
                    f"({int(pred.sum())} flagged for retention)")
        return out

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data or not isinstance(data, dict):
            raise ValueError("Input data must be a non-empty dictionary")
        row = self.predict_batch([data]).iloc[0].to_dict()
        row["churn_probability"] = round(float(row["churn_probability"]), 4)
        row["prediction"] = int(row["prediction"])
        row["threshold"] = float(row["threshold"])
        row["confidence"] = f"{row['churn_probability'] * 100:.2f}%"
        return row
