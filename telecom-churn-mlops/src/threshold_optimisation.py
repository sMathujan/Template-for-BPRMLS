"""Decision-threshold optimisation — the core idea of the research.

The threshold is chosen on VALIDATION probabilities only, then frozen and
applied unchanged to test data and to live inference.

Two objectives are supported:
  * "f1"             — F1-optimal threshold (reproduces the published results)
  * "expected_value" — maximises retention revenue using UK economics:
        value  = TP × (save_rate × CLV)  −  (TP + FP) × offer_cost
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, recall_score

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BusinessEconomics:
    customer_lifetime_value_gbp: float = 1656.0
    retention_offer_cost_gbp: float = 99.0
    retention_success_rate: float = 0.30

    @property
    def value_per_saved_churner(self) -> float:
        return self.retention_success_rate * self.customer_lifetime_value_gbp

    def expected_value(self, y_true, y_pred) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        contacted = int((y_pred == 1).sum())
        return tp * self.value_per_saved_churner - contacted * self.retention_offer_cost_gbp

    def break_even_offer_cost(self) -> float:
        """Offer cost above which contacting a *true* churner stops paying."""
        return self.value_per_saved_churner


class ThresholdOptimiser:
    def __init__(self, metric: str = "f1", search_min: float = 0.10, search_max: float = 0.90,
                 step: float = 0.01, economics: Optional[BusinessEconomics] = None):
        if metric not in {"f1", "recall", "youden", "expected_value"}:
            raise ValueError(f"Unknown threshold metric: {metric}")
        self.metric = metric
        self.grid = np.round(np.arange(search_min, search_max, step), 4)
        self.economics = economics or BusinessEconomics()

    def _score(self, y_true, y_prob, thr: float) -> float:
        y_pred = (y_prob >= thr).astype(int)
        if self.metric == "f1":
            return f1_score(y_true, y_pred, zero_division=0)
        if self.metric == "recall":
            return recall_score(y_true, y_pred, zero_division=0)
        if self.metric == "youden":
            tpr = recall_score(y_true, y_pred, zero_division=0)
            tnr = recall_score(1 - np.asarray(y_true), 1 - y_pred, zero_division=0)
            return tpr + tnr - 1
        return self.economics.expected_value(y_true, y_pred)

    def find_best_threshold(self, y_true, y_prob) -> Tuple[float, float]:
        """Scan the grid and return (best_threshold, best_score). Pass VALIDATION data."""
        y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
        best_thr, best_score = 0.5, -np.inf
        for thr in self.grid:
            score = self._score(y_true, y_prob, thr)
            if score > best_score:
                best_score, best_thr = score, round(float(thr), 2)
        return best_thr, float(best_score)

    def select_threshold(self, model, X_val, y_val) -> Dict[str, float]:
        val_prob = model.predict_proba(X_val)[:, 1]
        thr, score = self.find_best_threshold(y_val, val_prob)
        logger.info(f"✓ Validation-optimal threshold ({self.metric}) = {thr:.2f} "
                    f"(val score = {score:.4f})")
        return {"threshold": thr, "val_score": score, "metric": self.metric}

    def sweep(self, y_true, y_prob) -> Dict[str, list]:
        """Full curve (for plotting): threshold vs score."""
        return {"threshold": self.grid.tolist(),
                "score": [self._score(np.asarray(y_true), np.asarray(y_prob), t) for t in self.grid]}
