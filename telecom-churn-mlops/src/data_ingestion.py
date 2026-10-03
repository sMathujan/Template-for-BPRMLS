"""Data ingestion strategies (CSV / Excel) with basic schema validation."""
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from utils.logger import banner, get_logger

logger = get_logger(__name__)


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        pass

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: Optional[List[str]]) -> None:
        if not required_columns:
            return
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")
        logger.info(f"✓ Schema check passed ({len(required_columns)} required columns present)")


class DataIngestorCSV(DataIngestor):
    def __init__(self, required_columns: Optional[List[str]] = None):
        self.required_columns = required_columns

    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        banner(logger, "DATA INGESTION - CSV")
        logger.info(f"Loading: {file_path_or_link}")
        df = pd.read_csv(file_path_or_link)
        self.validate_schema(df, self.required_columns)
        logger.info(f"✓ Loaded shape={df.shape}, "
                    f"memory={df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df


class DataIngestorExcel(DataIngestor):
    def __init__(self, required_columns: Optional[List[str]] = None):
        self.required_columns = required_columns

    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        banner(logger, "DATA INGESTION - EXCEL")
        logger.info(f"Loading: {file_path_or_link}")
        df = pd.read_excel(file_path_or_link)
        self.validate_schema(df, self.required_columns)
        logger.info(f"✓ Loaded shape={df.shape}")
        return df
