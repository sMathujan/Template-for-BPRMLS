"""ChurnPreprocessor — ONE feature pipeline used by both training and inference.

The template duplicated preprocessing logic inside model_inference.py; that is a
common source of training/serving skew. Here, the data pipeline FITS this object
on the training split and SAVES it; inference LOADS it and calls transform().
"""
import json
import os
from typing import List, Optional

import pandas as pd

from src.data_cleaning import TelcoDataCleaner
from src.feature_binning import CustomBinningStrategy
from src.feature_encoding import BinaryEncodingStrategy, NominalEncodingStrategy
from src.feature_scaling import SklearnScalingStrategy
from src.handle_missing_values import FillMissingValuesStrategy
from utils.config import (get_binning_config, get_cleaning_config, get_columns,
                          get_encoding_config, get_missing_values_config, get_path,
                          get_scaling_config)
from utils.logger import get_logger

logger = get_logger(__name__)


class ChurnPreprocessor:
    def __init__(self):
        cols, clean_cfg = get_columns(), get_cleaning_config()
        enc_cfg, scale_cfg = get_encoding_config(), get_scaling_config()
        self.binning_cfg = get_binning_config()

        self.raw_feature_columns: List[str] = cols["raw_feature_columns"]
        self.cleaner = TelcoDataCleaner(
            coerce_numeric=clean_cfg["coerce_numeric"],
            category_normalisation=clean_cfg["category_normalisation"],
            normalise_columns=clean_cfg["normalise_columns"],
            drop_columns=cols["drop_columns"])
        self.inference_filler = FillMissingValuesStrategy(
            fill_values=get_missing_values_config().get("inference_fill", {}))
        self.binary_encoder = BinaryEncodingStrategy(enc_cfg["binary_mappings"])
        self.nominal_encoder = NominalEncodingStrategy(
            enc_cfg["nominal_columns"], enc_cfg.get("drop_first", True), get_path("encode_dir"))
        self.binner = (CustomBinningStrategy(self.binning_cfg["bins"])
                       if self.binning_cfg.get("enabled") else None)
        self.scaler = SklearnScalingStrategy(scale_cfg["scaling_type"], get_path("scale_dir"))
        self.columns_to_scale = scale_cfg["columns_to_scale"]
        self.feature_columns: Optional[List[str]] = None
        self.feature_columns_path = os.path.join(get_path("model_artifacts_dir"),
                                                 "feature_columns.json")

    # ----------------------------------------------------------------- steps
    def _encode(self, df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        df = self.binary_encoder.encode(df, verbose=verbose)
        df = self.nominal_encoder.encode(df, verbose=verbose)
        if self.binner is not None:
            df = self.binner.bin_feature(df, self.binning_cfg["column"])
        return df

    def fit(self, X_train: pd.DataFrame) -> "ChurnPreprocessor":
        """Learn categories, scaler statistics and column order from TRAIN only."""
        self.nominal_encoder.fit(X_train)
        encoded = self._encode(X_train, verbose=False)
        self.scaler.fit(encoded, self.columns_to_scale)
        self.feature_columns = list(encoded.columns)
        logger.info(f"✓ Preprocessor fitted - {len(self.feature_columns)} model features")
        return self

    def transform(self, df: pd.DataFrame, clean: bool = True, verbose: bool = False) -> pd.DataFrame:
        if self.feature_columns is None:
            raise RuntimeError("Preprocessor not fitted/loaded.")
        if clean:
            missing = [c for c in self.raw_feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f"Input is missing required fields: {missing}")
            df = self.cleaner.clean(df, verbose=verbose)
            df = self.inference_filler.handle(df)
        df = self._encode(df, verbose=verbose)
        df = self.scaler.transform(df, verbose=verbose)
        # Exact training column order; extra columns (e.g. Churn) are dropped
        return df.reindex(columns=self.feature_columns, fill_value=0)

    # ------------------------------------------------------------- persistence
    def save(self) -> None:
        self.nominal_encoder.save()
        self.scaler.save()
        os.makedirs(os.path.dirname(self.feature_columns_path), exist_ok=True)
        with open(self.feature_columns_path, "w") as f:
            json.dump(self.feature_columns, f, indent=2)
        logger.info(f"✓ Feature column order saved → {self.feature_columns_path}")

    def load(self) -> "ChurnPreprocessor":
        self.nominal_encoder.load()
        self.scaler.load()
        if not os.path.exists(self.feature_columns_path):
            raise FileNotFoundError(f"{self.feature_columns_path} missing. Run the data pipeline.")
        with open(self.feature_columns_path) as f:
            self.feature_columns = json.load(f)
        return self
