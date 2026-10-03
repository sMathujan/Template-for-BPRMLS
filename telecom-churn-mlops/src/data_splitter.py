"""Data splitting: stratified 60/20/20 train/validation/test (Notebook 02).

Three splits, not two, because the decision threshold is itself a parameter
learned from data. Validation chooses it; test judges it, once.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import banner, get_logger

logger = get_logger(__name__)


class SplitType(str, Enum):
    SIMPLE = "simple"
    STRATIFIED_THREE_WAY = "stratified_three_way"


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> Dict[str, pd.DataFrame]:
        pass


class StratifiedThreeWaySplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size: float = 0.20, val_size: float = 0.20, random_state: int = 42):
        if not 0 < test_size + val_size < 1:
            raise ValueError("test_size + val_size must be between 0 and 1")
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str) -> Dict[str, pd.DataFrame]:
        banner(logger, "DATA SPLITTING - STRATIFIED TRAIN / VAL / TEST")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        holdout = self.test_size + self.val_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=holdout, random_state=self.random_state, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.test_size / holdout,
            random_state=self.random_state, stratify=y_temp)

        splits = {"X_train": X_train, "X_val": X_val, "X_test": X_test,
                  "Y_train": y_train, "Y_val": y_val, "Y_test": y_test}

        # Fail loudly rather than silently produce invalid results
        assert not set(X_train.index) & set(X_val.index), "Train/val overlap!"
        assert not set(X_train.index) & set(X_test.index), "Train/test overlap!"
        assert not set(X_val.index) & set(X_test.index), "Val/test overlap!"

        for name, ys in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
            logger.info(f"  {name:<11} n={len(ys):>5}  churn rate={ys.mean() * 100:.1f}%")
        logger.info("✓ Split integrity checks passed")
        return splits
