"""Model training + persistence (model and its metadata travel together)."""
import json
import os
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from utils.logger import banner, get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def train(self, model, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, float]:
        banner(logger, f"MODEL TRAINING - {type(model).__name__}")
        if X_train is None or y_train is None or len(X_train) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError(f"X/y length mismatch: {len(X_train)} vs {len(y_train)}")

        logger.info(f"  Samples: {len(X_train):,} | Features: {X_train.shape[1]} | "
                    f"Class counts: {np.bincount(np.asarray(y_train).astype(int)).tolist()}")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        logger.info(f"✓ Trained in {elapsed:.2f}s")
        return model, elapsed

    @staticmethod
    def save_model(model, filepath: str, metadata: Dict[str, Any], metadata_path: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        size_mb = os.path.getsize(filepath) / 1024**2
        logger.info(f"✓ Model saved → {filepath} ({size_mb:.2f} MB)")
        logger.info(f"✓ Metadata saved → {metadata_path}")

    @staticmethod
    def load_model(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}. Run the training pipeline.")
        return joblib.load(filepath)
