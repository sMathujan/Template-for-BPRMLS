"""Stateless cleaning steps from Notebook 01, reused at training AND inference time.

1. Coerce TotalCharges (read as text, blanks for tenure-0 customers) to numeric.
2. Collapse "No internet service" / "No phone service" into "No".
3. Drop the non-predictive identifier column.
"""
from typing import Dict, List

import pandas as pd

from utils.logger import banner, get_logger

logger = get_logger(__name__)


class TelcoDataCleaner:
    def __init__(self, coerce_numeric: List[str], category_normalisation: Dict[str, str],
                 normalise_columns: List[str], drop_columns: List[str]):
        self.coerce_numeric = coerce_numeric
        self.category_normalisation = category_normalisation
        self.normalise_columns = normalise_columns
        self.drop_columns = drop_columns

    def clean(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        df = df.copy()
        if verbose:
            banner(logger, "DATA CLEANING")

        for col in self.coerce_numeric:
            if col in df.columns:
                before_na = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col].astype(str).str.strip().replace("", None),
                                        errors="coerce")
                if verbose:
                    logger.info(f"✓ '{col}' coerced to numeric "
                                f"({df[col].isna().sum() - before_na} blanks → NaN)")

        for col in self.normalise_columns:
            if col in df.columns:
                df[col] = df[col].replace(self.category_normalisation)
        if verbose:
            logger.info(f"✓ Normalised service categories in {len(self.normalise_columns)} columns")

        to_drop = [c for c in self.drop_columns if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            if verbose:
                logger.info(f"✓ Dropped identifier columns: {to_drop}")

        # SeniorCitizen is already 0/1 but may arrive as string from an API
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

        if verbose:
            logger.info(f"✓ Cleaning complete - shape={df.shape}")
        return df
