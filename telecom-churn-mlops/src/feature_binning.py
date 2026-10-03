"""Optional tenure binning (disabled by default to match the published model)."""
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass


class CustomBinningStrategy(FeatureBinningStrategy):
    """Adds `<column>_bin` as an ordinal integer; keeps the original column."""

    def __init__(self, bin_definitions: Dict[str, List[float]]):
        self.bin_definitions = bin_definitions
        self.labels = list(bin_definitions.keys())

    def _assign(self, value: float) -> int:
        for idx, (lo, hi) in enumerate(self.bin_definitions.values()):
            if lo <= value <= hi:
                return idx
        return len(self.labels) - 1 if value > 0 else 0

    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df.copy()
        df[f"{column}_bin"] = df[column].apply(self._assign).astype(int)
        logger.info(f"✓ Binned '{column}' into {self.labels}")
        return df
