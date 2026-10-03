"""Training pipeline — Notebook 03 as a reproducible, tracked job.

For each model (RF, XGBoost, CatBoost):
  1. Optuna-tune on the RAW train split (SMOTE inside each CV fold)
  2. Refit on the SMOTE-balanced train split
  3. Choose the decision threshold on VALIDATION
  4. Evaluate once on TEST with the threshold frozen (+ bootstrap CI)
Then promote the champion (best VALIDATION F1) to artifacts/models and the MLflow registry.

    python pipelines/training_pipeline.py                  # full run (config trial counts)
    python pipelines/training_pipeline.py --n-trials 5     # quick run
    python pipelines/training_pipeline.py --no-tune        # research baselines only
    python pipelines/training_pipeline.py --models xgboost random_forest
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from pipelines.data_pipeline import data_pipeline
from src.class_imbalance import get_imbalance_strategy
from src.hyperparameter_tuning import OptunaTuner
from src.model_building import get_model_builder
from src.model_evaluation import ModelEvaluator
from src.model_training import ModelTrainer
from src.threshold_optimisation import BusinessEconomics, ThresholdOptimiser
from utils.config import (get_business_config, get_evaluation_config, get_imbalance_config,
                          get_model_config, get_path, get_project_config,
                          get_threshold_config, get_training_config, resolve_path)
from utils.logger import banner, get_logger
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags

logger = get_logger(__name__)


def _baseline_params(model_key: str, y_train_raw: pd.Series) -> Dict[str, Any]:
    """Research baselines; boosting models get the raw-train class ratio as weight."""
    params = dict(get_model_config()["model_types"].get(model_key, {}))
    ratio = float((y_train_raw == 0).sum() / (y_train_raw == 1).sum())
    if model_key == "xgboost":
        params.setdefault("scale_pos_weight", ratio)
    elif model_key == "catboost":
        params.setdefault("minority_weight", ratio)
    return params


def plot_model_comparison(results: pd.DataFrame, out_path: str) -> str:
    metrics = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ax = results[metrics].T.plot.bar(figsize=(11, 5), rot=0, width=0.75)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Tuned models - TEST set (thresholds chosen on validation)", fontweight="bold")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    return out_path


def train_one_model(model_key: str, splits: Dict[str, pd.DataFrame], X_train_sm, y_train_sm,
                    tune: bool, n_trials: Optional[int], economics: BusinessEconomics,
                    run_dir: str) -> Dict[str, Any]:
    train_cfg, thr_cfg = get_training_config(), get_threshold_config()
    eval_cfg = get_evaluation_config()
    seed = get_project_config().get("random_state", 42)

    with mlflow.start_run(run_name=model_key, nested=True):
        mlflow.set_tags({"model_type": model_key, "tuned": str(tune)})

        # 1. Hyperparameters
        cv_f1 = None
        if tune:
            tune_cfg = train_cfg["hyperparameter_tuning"]
            trials = n_trials or tune_cfg["n_trials"][model_key]
            tuner = OptunaTuner(model_key, trials, train_cfg.get("cv_folds", 5), seed,
                                tune_cfg.get("early_stopping_rounds", 40))
            params, cv_f1 = tuner.tune(splits["X_train"], splits["Y_train"])
            hist_path = os.path.join(run_dir, f"optuna_history_{model_key}.csv")
            tuner.history().to_csv(hist_path, index=False)
            mlflow.log_artifact(hist_path, "optuna")
            MLflowTracker.log_metrics_safe({"cv_f1_best": cv_f1, "optuna_trials": trials})
        else:
            params = _baseline_params(model_key, splits["Y_train"])
        MLflowTracker.log_params_safe(params)

        # 2. Refit on SMOTE-balanced train
        builder = get_model_builder(model_key, random_state=seed, **params)
        model, train_time = ModelTrainer().train(builder.build_model(), X_train_sm, y_train_sm)

        # 3. Threshold from VALIDATION only
        optimiser = ThresholdOptimiser(thr_cfg.get("metric", "f1"), thr_cfg.get("search_min", 0.10),
                                       thr_cfg.get("search_max", 0.90), thr_cfg.get("step", 0.01),
                                       economics)
        thr_info = optimiser.select_threshold(model, splits["X_val"], splits["Y_val"])
        thr = thr_info["threshold"]

        # 4. Evaluate: validation (selection) and test (reporting, threshold frozen)
        evaluator = ModelEvaluator(model, builder.display_name, economics)
        val_m = evaluator.evaluate(splits["X_val"], splits["Y_val"], thr, "validation")
        test_m = evaluator.evaluate(splits["X_test"], splits["Y_test"], thr, "test")
        test_default = evaluator.evaluate(splits["X_test"], splits["Y_test"], 0.50, "test@0.50")
        ci = {}
        if eval_cfg.get("bootstrap", {}).get("enabled", True):
            b = eval_cfg["bootstrap"]
            ci = evaluator.bootstrap_ci(splits["X_test"], splits["Y_test"], thr,
                                        b.get("n_iterations", 1000), b.get("confidence_level", 0.95), seed)

        MLflowTracker.log_metrics_safe({"threshold": thr, "training_time_seconds": train_time})
        MLflowTracker.log_metrics_safe({f"val_{k}": v for k, v in val_m.items()})
        MLflowTracker.log_metrics_safe({f"test_{k}": v for k, v in test_m.items()})
        MLflowTracker.log_metrics_safe({f"test_at_0.50_{k}": v for k, v in test_default.items()})
        MLflowTracker.log_metrics_safe({f"test_{k}": v for k, v in ci.items()})

        plot_dir = os.path.join(run_dir, model_key)
        evaluator.create_plots(splits["X_test"], splits["Y_test"], thr, plot_dir,
                               splits["X_val"], splits["Y_val"])
        mlflow.log_artifacts(plot_dir, f"model_performance/{model_key}")

    return {"model_key": model_key, "display_name": builder.display_name, "model": model,
            "params": params, "cv_f1": cv_f1, "threshold": thr, "threshold_metric": thr_info["metric"],
            "val": val_m, "test": test_m, "test_at_0.50": test_default, "ci": ci,
            "training_time": train_time}


def training_pipeline(models: Optional[List[str]] = None, tune: Optional[bool] = None,
                      n_trials: Optional[int] = None, force_data: bool = False) -> Dict[str, Any]:
    banner(logger, "STARTING TRAINING PIPELINE", 80)
    train_cfg = get_training_config()
    models = models or train_cfg["models"]
    tune = train_cfg["hyperparameter_tuning"]["enabled"] if tune is None else tune
    selection_metric = train_cfg.get("selection_metric", "F1")
    economics = BusinessEconomics(**get_business_config())
    seed = get_project_config().get("random_state", 42)

    splits = data_pipeline(force_rebuild=force_data)

    tracker = MLflowTracker()
    run = tracker.start_run("training_pipeline", create_mlflow_run_tags(
        "training_pipeline", {"models": ",".join(models), "tuned": tune,
                              "threshold_metric": get_threshold_config().get("metric", "f1")}))
    run_dir = os.path.join(get_path("artifacts_dir"), "mlflow_training_artifacts", run.info.run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        # SMOTE on the training split only
        imb_cfg = get_imbalance_config()
        X_train_sm, y_train_sm = get_imbalance_strategy(
            imb_cfg.get("method", "smote"), imb_cfg.get("random_state", seed)
        ).resample(splits["X_train"], splits["Y_train"])
        MLflowTracker.log_metrics_safe({"train_rows_raw": len(splits["X_train"]),
                                        "train_rows_smote": len(X_train_sm)})

        results = [train_one_model(m, splits, X_train_sm, y_train_sm, tune, n_trials,
                                   economics, run_dir) for m in models]

        # Comparison table (test metrics) + champion by VALIDATION metric
        table = pd.DataFrame({r["display_name"]: {"Threshold": r["threshold"],
                                                  f"Val {selection_metric}": r["val"][selection_metric],
                                                  **{k: r["test"][k] for k in
                                                     ["Accuracy", "Precision", "Recall", "F1",
                                                      "ROC-AUC", "PR-AUC", "ExpectedValue_GBP"]},
                                                  **r["ci"]} for r in results}).T
        table_path = os.path.join(run_dir, "model_comparison.csv")
        table.round(4).to_csv(table_path)
        mlflow.log_artifact(table_path, "comparison")
        if len(results) > 1:
            mlflow.log_artifact(plot_model_comparison(table, os.path.join(run_dir, "model_comparison.png")),
                                "comparison")
        logger.info("\n" + table.round(4).to_string())

        champion = max(results, key=lambda r: r["val"][selection_metric])
        banner(logger, f"CHAMPION: {champion['display_name']} "
                       f"(val {selection_metric}={champion['val'][selection_metric]:.4f}, "
                       f"thr={champion['threshold']:.2f})")

        # Persist champion + metadata (threshold travels with the model)
        model_cfg = get_model_config()
        model_path = resolve_path(model_cfg["model_path"])
        meta_path = resolve_path(model_cfg["metadata_path"])
        metadata = {
            "model_name": champion["display_name"], "model_key": champion["model_key"],
            "model_version": datetime.now().strftime("%Y%m%d%H%M%S"),
            "mlflow_run_id": run.info.run_id,
            "threshold": champion["threshold"], "threshold_metric": champion["threshold_metric"],
            "selection_metric": selection_metric, "hyperparameters": champion["params"],
            "cv_f1": champion["cv_f1"], "validation_metrics": champion["val"],
            "test_metrics": champion["test"], "test_metrics_at_0.50": champion["test_at_0.50"],
            "test_f1_ci": champion["ci"], "feature_columns": list(splits["X_train"].columns),
            "business_economics": get_business_config(),
            "trained_at": datetime.now().isoformat(),
        }
        ModelTrainer.save_model(champion["model"], model_path, metadata, meta_path)
        mlflow.log_artifact(meta_path, "champion")
        mlflow.set_tags({"champion_model": champion["display_name"]})
        MLflowTracker.log_metrics_safe({"champion_threshold": champion["threshold"],
                                        "champion_test_F1": champion["test"]["F1"],
                                        "champion_test_ExpectedValue_GBP":
                                            champion["test"]["ExpectedValue_GBP"]})
        tracker.log_model(champion["model"], input_example=splits["X_test"].head(3).astype(float))

        tracker.end_run()
        logger.info("✓ Training pipeline completed successfully")
        return {"results": results, "champion": champion, "comparison": table}
    except Exception:
        tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, tune and select the churn model")
    parser.add_argument("--models", nargs="+", choices=["random_forest", "xgboost", "catboost"])
    parser.add_argument("--n-trials", type=int, help="Override Optuna trials for every model")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna; use baseline params")
    parser.add_argument("--force-data", action="store_true", help="Rebuild data artifacts first")
    args = parser.parse_args()

    out = training_pipeline(models=args.models, tune=False if args.no_tune else None,
                            n_trials=args.n_trials, force_data=args.force_data)
    c = out["champion"]
    print("\n" + "=" * 70)
    print("🏆 TRAINING SUMMARY (test set, thresholds chosen on validation)")
    print("=" * 70)
    print(out["comparison"][["Threshold", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"]]
          .astype(float).round(4).to_string())
    print("-" * 70)
    print(f"Champion: {c['display_name']} @ threshold {c['threshold']:.2f}")
    print(f"  Test F1 {c['test']['F1']:.4f} vs {c['test_at_0.50']['F1']:.4f} at default 0.50")
    print(f"  Expected retention value on test: £{c['test']['ExpectedValue_GBP']:,.0f} "
          f"(vs £{c['test_at_0.50']['ExpectedValue_GBP']:,.0f} at 0.50)")
    print("=" * 70)
