"""Evaluation at a FROZEN threshold, bootstrap confidence intervals, and plots."""
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless servers / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src.threshold_optimisation import BusinessEconomics
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, model, model_name: str, economics: Optional[BusinessEconomics] = None):
        self.model = model
        self.model_name = model_name
        self.economics = economics or BusinessEconomics()

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y_true, threshold: float, split_name: str = "test") -> Dict[str, Any]:
        y_true = np.asarray(y_true).astype(int)
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results = {
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "PR-AUC": average_precision_score(y_true, y_prob),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "ExpectedValue_GBP": self.economics.expected_value(y_true, y_pred),
        }
        logger.info(f"[{self.model_name} | {split_name} @ thr={threshold:.2f}] "
                    f"F1={results['F1']:.4f} Recall={results['Recall']:.4f} "
                    f"Precision={results['Precision']:.4f} ROC-AUC={results['ROC-AUC']:.4f} "
                    f"PR-AUC={results['PR-AUC']:.4f} EV=£{results['ExpectedValue_GBP']:,.0f}")
        return results

    def bootstrap_ci(self, X, y_true, threshold: float, n_iterations: int = 1000,
                     confidence_level: float = 0.95, seed: int = 42) -> Dict[str, float]:
        """Percentile bootstrap CI for test F1 (the research reports ~0.068 width)."""
        rng = np.random.default_rng(seed)
        y_true = np.asarray(y_true).astype(int)
        y_pred = (self.predict_proba(X) >= threshold).astype(int)
        n = len(y_true)
        scores = []
        for _ in range(n_iterations):
            idx = rng.integers(0, n, n)
            scores.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
        alpha = (1 - confidence_level) / 2
        lo, hi = np.quantile(scores, [alpha, 1 - alpha])
        logger.info(f"✓ {self.model_name} F1 {confidence_level:.0%} CI: [{lo:.4f}, {hi:.4f}]")
        return {"F1_CI_low": float(lo), "F1_CI_high": float(hi), "F1_CI_width": float(hi - lo)}

    # ------------------------------------------------------------------ plots
    def create_plots(self, X, y_true, threshold: float, out_dir: str,
                     X_val=None, y_val=None) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        y_true = np.asarray(y_true).astype(int)
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        name = self.model_name.replace(" ", "_")
        paths = []

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontweight="bold",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["No Churn", "Churn"]); ax.set_yticks([0, 1], ["No Churn", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{self.model_name} (thr={threshold:.2f})")
        paths.append(self._save(fig, out_dir, f"confusion_matrix_{name}.png"))

        # ROC + PR
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        axes[1].plot(rec, prec, lw=2, label=f"PR-AUC={average_precision_score(y_true, y_prob):.3f}")
        axes[1].axhline(y_true.mean(), color="grey", ls="--", lw=1, label="No skill")
        axes[1].scatter(recall_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0),
                        s=90, zorder=5, color="red", label=f"Operating point (thr={threshold:.2f})")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        for a in axes:
            a.legend(); a.grid(alpha=0.3)
        fig.suptitle(self.model_name, fontweight="bold")
        paths.append(self._save(fig, out_dir, f"roc_pr_{name}.png"))

        # Threshold sweep: validation (solid, used to choose) vs test (dashed)
        sweep = np.arange(0.10, 0.90, 0.01)
        fig, ax = plt.subplots(figsize=(8, 5))
        sets = [(X, y_true, "--", "test")]
        if X_val is not None:
            sets.insert(0, (X_val, np.asarray(y_val).astype(int), "-", "val"))
        for Xs, ys, ls, tag in sets:
            p = self.predict_proba(Xs)
            ax.plot(sweep, [f1_score(ys, (p >= t).astype(int), zero_division=0) for t in sweep],
                    ls, lw=2, label=f"F1 ({tag})")
            ax.plot(sweep, [recall_score(ys, (p >= t).astype(int), zero_division=0) for t in sweep],
                    ls, lw=1.5, alpha=0.7, label=f"Recall ({tag})")
        ax.axvline(threshold, color="green", lw=1.5, label=f"Selected thr={threshold:.2f}")
        ax.set(xlabel="Threshold", ylabel="Score", ylim=(0, 1.02),
               title=f"{self.model_name}: metric vs decision threshold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        paths.append(self._save(fig, out_dir, f"threshold_sweep_{name}.png"))

        # Feature importance
        if hasattr(self.model, "feature_importances_") and hasattr(X, "columns"):
            fi = (pd.DataFrame({"feature": X.columns, "importance": self.model.feature_importances_})
                  .sort_values("importance").tail(15))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(fi["feature"], fi["importance"], color="steelblue")
            ax.set_title(f"{self.model_name}: top-15 feature importances")
            paths.append(self._save(fig, out_dir, f"feature_importance_{name}.png"))
            fi.sort_values("importance", ascending=False).to_csv(
                os.path.join(out_dir, f"feature_importance_{name}.csv"), index=False)
        return paths

    @staticmethod
    def _save(fig, out_dir: str, filename: str) -> str:
        path = os.path.join(out_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
