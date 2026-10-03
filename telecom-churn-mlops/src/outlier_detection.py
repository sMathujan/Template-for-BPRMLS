"""Outlier detection (IQR). For Telco the default is 'report' — log, don't remove."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd

from utils.logger import banner, get_logger

logger = get_logger(__name__)


class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass


class IQROutlierDetection(OutlierDetectionStrategy):
    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        flags = pd.DataFrame(False, index=df.index, columns=columns)
        for col in columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - self.multiplier * iqr, q3 + self.multiplier * iqr
            flags[col] = (df[col] < lo) | (df[col] > hi)
            logger.info(f"  {col}: bounds [{lo:.2f}, {hi:.2f}] → {int(flags[col].sum())} outliers")
        return flags


class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def handle_outliers(self, df: pd.DataFrame, columns: List[str],
                        method: str = "report") -> Tuple[pd.DataFrame, int]:
        banner(logger, f"OUTLIER DETECTION - {method.upper()}")
        flags = self._strategy.detect_outliers(df, columns)
        n_rows = int(flags.any(axis=1).sum())
        logger.info(f"✓ Rows with ≥1 outlier: {n_rows} ({n_rows / max(len(df), 1) * 100:.2f}%)")
        if method == "remove":
            df = df[~flags.any(axis=1)].reset_index(drop=True)
            logger.info(f"✓ Removed {n_rows} rows → {len(df)} remain")
        elif method == "report":
            logger.info("✓ Report-only mode: no rows removed")
        else:
            raise ValueError(f"Unknown outlier handling method: {method}")
        return df, n_rows
