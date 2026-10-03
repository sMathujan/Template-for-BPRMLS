"""Feature scaling. The scaler is FITTED ON TRAIN ONLY and persisted for inference.

(Notebook 01 fitted the scaler on the full dataset before splitting — a small
leakage. The production pipeline fixes that.)
"""
import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.logger import banner, get_logger

logger = get_logger(__name__)


class ScalingType(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"


class FeatureScalingStrategy(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, columns: List[str]) -> "FeatureScalingStrategy":
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class SklearnScalingStrategy(FeatureScalingStrategy):
    def __init__(self, scaling_type: str = "standard", scale_dir: str = "artifacts/scale"):
        self.scaling_type = ScalingType(scaling_type)
        self.scaler = StandardScaler() if self.scaling_type == ScalingType.STANDARD else MinMaxScaler()
        self.scale_dir = scale_dir
        self.columns: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "SklearnScalingStrategy":
        banner(logger, f"FEATURE SCALING - FIT ({self.scaling_type.value.upper()}, TRAIN ONLY)")
        self.columns = list(columns)
        self.scaler.fit(df[self.columns])
        self.fitted = True
        for col in self.columns:
            logger.info(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
        return self

    def transform(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted/loaded. Call fit() or load() first.")
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns])
        if verbose:
            logger.info(f"✓ Scaled {self.columns}")
        return df

    def save(self) -> None:
        os.makedirs(self.scale_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(self.scale_dir, "scaler.joblib"))
        meta = {"scaling_type": self.scaling_type.value, "columns_to_scale": self.columns}
        if self.scaling_type == ScalingType.STANDARD:
            meta.update(mean=self.scaler.mean_.tolist(), scale=self.scaler.scale_.tolist())
        else:
            meta.update(data_min=self.scaler.data_min_.tolist(), data_max=self.scaler.data_max_.tolist())
        with open(os.path.join(self.scale_dir, "scaling_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"✓ Scaler saved to {self.scale_dir}")

    def load(self) -> "SklearnScalingStrategy":
        scaler_path = os.path.join(self.scale_dir, "scaler.joblib")
        meta_path = os.path.join(self.scale_dir, "scaling_metadata.json")
        if not (os.path.exists(scaler_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"Scaler artifacts missing in {self.scale_dir}. "
                                    f"Run the data pipeline first.")
        self.scaler = joblib.load(scaler_path)
        with open(meta_path) as f:
            meta = json.load(f)
        self.columns = meta["columns_to_scale"]
        self.scaling_type = ScalingType(meta["scaling_type"])
        self.fitted = True
        logger.info(f"✓ Scaler loaded ({self.scaling_type.value}) for {self.columns}")
        return self
