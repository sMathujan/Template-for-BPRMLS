"""Optuna (TPE / Bayesian) hyperparameter tuning — Notebook 03, Section 5.

Objective: mean 5-fold stratified CV F1 on the RAW training split, with SMOTE
fitted inside each fold (never on the held-out fold). For XGBoost and CatBoost
the mean early-stopping iteration is recorded so the final refit reuses it
without consuming the validation set (reserved for threshold selection).
"""
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.class_imbalance import SMOTEStrategy
from utils.logger import banner, get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuner:
    def __init__(self, model_key: str, n_trials: int = 50, cv_folds: int = 5,
                 random_state: int = 42, early_stopping_rounds: int = 40):
        self.model_key = model_key
        self.n_trials = n_trials
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.smote = SMOTEStrategy(random_state)
        self.study: optuna.Study = None

    # ---------------------------------------------------------------- objectives
    def _folds(self, X: pd.DataFrame, y: pd.Series):
        for tr_idx, va_idx in self.cv.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            X_tr_sm, y_tr_sm = self.smote.resample(X_tr, y_tr, verbose=False)
            yield X_tr_sm, y_tr_sm, X_va, y_va

    @staticmethod
    def _f1(model, X_va, y_va) -> float:
        prob = model.predict_proba(X_va)[:, 1]
        return f1_score(y_va, (prob >= 0.5).astype(int), zero_division=0)

    def _rf_objective(self, trial, X, y) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        scores = []
        for X_tr, y_tr, X_va, y_va in self._folds(X, y):
            m = RandomForestClassifier(**params, class_weight="balanced",
                                       random_state=self.random_state, n_jobs=-1)
            m.fit(X_tr, y_tr)
            scores.append(self._f1(m, X_va, y_va))
        return float(np.mean(scores))

    def _xgb_objective(self, trial, X, y) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
        }
        scores, best_iters = [], []
        for X_tr, y_tr, X_va, y_va in self._folds(X, y):
            m = XGBClassifier(**params, n_estimators=1000, eval_metric="aucpr",
                              early_stopping_rounds=self.early_stopping_rounds,
                              random_state=self.random_state, n_jobs=-1, verbosity=0)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            best_iters.append(m.best_iteration)
            scores.append(self._f1(m, X_va, y_va))
        trial.set_user_attr("mean_best_iteration", int(np.mean(best_iters)))
        return float(np.mean(scores))

    def _cat_objective(self, trial, X, y) -> float:
        minority_weight = trial.suggest_float("minority_weight", 1.0, 4.0)
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 128),
        }
        scores, best_iters = [], []
        for X_tr, y_tr, X_va, y_va in self._folds(X, y):
            m = CatBoostClassifier(**params, iterations=1000, class_weights=[1.0, minority_weight],
                                   eval_metric="F1", od_type="Iter",
                                   od_wait=self.early_stopping_rounds, loss_function="Logloss",
                                   random_state=self.random_state, verbose=False,
                                   allow_writing_files=False)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
            best_iters.append(m.get_best_iteration())
            scores.append(self._f1(m, X_va, y_va))
        trial.set_user_attr("mean_best_iteration", int(np.mean(best_iters)))
        return float(np.mean(scores))

    # ------------------------------------------------------------------- public
    def tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], float]:
        """Return (final_model_params, best_cv_f1). Params are ready for the model builder."""
        banner(logger, f"OPTUNA TUNING - {self.model_key.upper()} ({self.n_trials} trials)")
        objective = {"random_forest": self._rf_objective,
                     "xgboost": self._xgb_objective,
                     "catboost": self._cat_objective}[self.model_key]

        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        log_every = max(1, self.n_trials // 10)

        def _progress(study, trial):
            if (trial.number + 1) % log_every == 0 or trial.number + 1 == self.n_trials:
                logger.info(f"  trial {trial.number + 1:>3}/{self.n_trials}  "
                            f"CV F1={trial.value:.4f}  best={study.best_value:.4f}")

        self.study.optimize(lambda t: objective(t, X_train, y_train), n_trials=self.n_trials,
                            callbacks=[_progress])

        best = dict(self.study.best_params)
        mean_iter = max(int(self.study.best_trial.user_attrs.get("mean_best_iteration", 500)), 50)
        if self.model_key == "xgboost":
            best.update(n_estimators=mean_iter, eval_metric="aucpr")
        elif self.model_key == "catboost":
            best.update(iterations=mean_iter, eval_metric="F1")

        logger.info(f"✓ Best CV F1 = {self.study.best_value:.4f}")
        logger.info(f"✓ Best params = {best}")
        return best, float(self.study.best_value)

    def history(self) -> pd.DataFrame:
        vals = [t.value for t in self.study.trials if t.value is not None]
        return pd.DataFrame({"trial": range(len(vals)), "cv_f1": vals,
                             "best_so_far": np.maximum.accumulate(vals)})
