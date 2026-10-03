"""Missing-value strategies (Strategy pattern, as in the template)."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from utils.logger import banner, get_logger

logger = get_logger(__name__)


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """Drop rows with missing values in critical columns (training data)."""

    def __init__(self, critical_columns: Optional[List[str]] = None):
        self.critical_columns = critical_columns or []

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        banner(logger, "MISSING VALUES - DROP")
        subset = [c for c in self.critical_columns if c in df.columns] or None
        cleaned = df.dropna(subset=subset).reset_index(drop=True)
        logger.info(f"✓ Dropped {len(df) - len(cleaned)} rows with missing values in {subset}")
        remaining = int(cleaned.isna().sum().sum())
        if remaining:
            logger.warning(f"⚠ {remaining} missing values remain in non-critical columns")
        return cleaned


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """Fill with a constant per column, or with the column mean/median/mode."""

    def __init__(self, fill_values: Optional[Dict[str, float]] = None, method: Optional[str] = None,
                 columns: Optional[List[str]] = None):
        self.fill_values = fill_values or {}
        self.method = method
        self.columns = columns or []

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, val in self.fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        if self.method:
            for col in self.columns:
                if col not in df.columns:
                    continue
                if self.method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif self.method == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif self.method == "mode":
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                else:
                    raise ValueError(f"Unknown fill method: {self.method}")
        return df
