"""Feature encoding: binary maps + one-hot (drop_first) for nominal columns.

Categories are learned on the TRAINING split and saved to artifacts/encode so
inference produces exactly the same columns as training.
"""
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

from utils.logger import banner, get_logger

logger = get_logger(__name__)


class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class BinaryEncodingStrategy(FeatureEncodingStrategy):
    """Yes/No and Male/Female → 1/0 using fixed, config-defined mappings."""

    def __init__(self, binary_mappings: Dict[str, Dict[str, int]]):
        self.binary_mappings = binary_mappings

    def encode(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        df = df.copy()
        if verbose:
            banner(logger, "BINARY ENCODING")
        for col, mapping in self.binary_mappings.items():
            if col not in df.columns:
                continue
            original = df[col]
            df[col] = original.map(mapping)
            unmapped = df[col].isna() & original.notna()
            if unmapped.any():
                bad = sorted(original[unmapped].astype(str).unique())
                raise ValueError(f"Column '{col}' has unexpected values {bad}; "
                                 f"allowed: {list(mapping.keys())}")
            df[col] = df[col].astype(int)
        if verbose:
            logger.info(f"✓ Binary-encoded {len(self.binary_mappings)} columns")
        return df


class NominalEncodingStrategy(FeatureEncodingStrategy):
    """One-hot encoding with drop_first, matching pd.get_dummies(drop_first=True)."""

    def __init__(self, nominal_columns: List[str], drop_first: bool = True,
                 encode_dir: str = "artifacts/encode"):
        self.nominal_columns = nominal_columns
        self.drop_first = drop_first
        self.encode_dir = encode_dir
        self.categories: Dict[str, List[str]] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "NominalEncodingStrategy":
        for col in self.nominal_columns:
            self.categories[col] = sorted(df[col].dropna().astype(str).unique().tolist())
            logger.info(f"  {col}: {self.categories[col]}")
        self.fitted = True
        return self

    def encode(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("NominalEncodingStrategy must be fitted (or loaded) before encode().")
        df = df.copy()
        if verbose:
            banner(logger, "ONE-HOT ENCODING (NOMINAL)")
        for col in self.nominal_columns:
            cats = self.categories[col]
            values = df[col].astype(str)
            unknown = ~values.isin(cats)
            if unknown.any():
                logger.warning(f"⚠ '{col}' has unseen categories {sorted(values[unknown].unique())}"
                               f" — encoded as the reference category")
            kept = cats[1:] if self.drop_first else cats
            for cat in kept:
                df[f"{col}_{cat}"] = (values == cat).astype(int)
            df = df.drop(columns=[col])
            if verbose:
                logger.info(f"✓ {col} → {len(kept)} columns (reference: '{cats[0]}')")
        return df

    def save(self) -> None:
        os.makedirs(self.encode_dir, exist_ok=True)
        for col, cats in self.categories.items():
            path = os.path.join(self.encode_dir, f"{col}_encoder.json")
            with open(path, "w") as f:
                json.dump({"categories": cats, "encoding_type": "one_hot",
                           "drop_first": self.drop_first}, f, indent=2)
        logger.info(f"✓ Saved {len(self.categories)} encoders to {self.encode_dir}")

    def load(self) -> "NominalEncodingStrategy":
        for col in self.nominal_columns:
            path = os.path.join(self.encode_dir, f"{col}_encoder.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Encoder not found: {path}. Run the data pipeline first.")
            with open(path) as f:
                data = json.load(f)
            self.categories[col] = data["categories"]
            self.drop_first = data.get("drop_first", self.drop_first)
        self.fitted = True
        logger.info(f"✓ Loaded encoders for {self.nominal_columns}")
        return self


def encode_target(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    encoded = series.map(mapping)
    if encoded.isna().any():
        raise ValueError(f"Target has unexpected values: {series[encoded.isna()].unique()}")
    return encoded.astype(int)
