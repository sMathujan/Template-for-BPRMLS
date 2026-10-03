"""Class-imbalance handling. SMOTE is applied to the TRAINING split only —
never to validation or test — and re-applied inside every CV fold during tuning."""
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE

from utils.logger import get_logger

logger = get_logger(__name__)


class ImbalanceStrategy(ABC):
    @abstractmethod
    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        pass


class SMOTEStrategy(ImbalanceStrategy):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def resample(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True):
        X_res, y_res = SMOTE(random_state=self.random_state).fit_resample(X, y)
        if verbose:
            logger.info(f"✓ SMOTE: {y.value_counts().sort_index().to_dict()} → "
                        f"{pd.Series(y_res).value_counts().sort_index().to_dict()}")
        return X_res, pd.Series(y_res, name=y.name)


class NoResamplingStrategy(ImbalanceStrategy):
    def resample(self, X, y, verbose: bool = True):
        return X, y


def get_imbalance_strategy(method: str, random_state: int = 42) -> ImbalanceStrategy:
    if method == "smote":
        return SMOTEStrategy(random_state)
    if method in (None, "none"):
        return NoResamplingStrategy()
    raise ValueError(f"Unknown imbalance method: {method}")
