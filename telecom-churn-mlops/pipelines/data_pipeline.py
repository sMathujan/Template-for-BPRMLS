"""Data pipeline: raw CSV → cleaned → split (60/20/20) → features fitted on train → saved.

    python pipelines/data_pipeline.py            # reuse existing artifacts if present
    python pipelines/data_pipeline.py --force    # rebuild everything
"""
import argparse
import json
import os
import sys
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.data_cleaning import TelcoDataCleaner
from src.data_ingestion import DataIngestorCSV
from src.data_splitter import StratifiedThreeWaySplitStrategy
from src.feature_encoding import encode_target
from src.handle_missing_values import DropMissingValuesStrategy
from src.outlier_detection import IQROutlierDetection, OutlierDetector
from src.preprocessing import ChurnPreprocessor
from utils.config import (get_cleaning_config, get_columns, get_encoding_config,
                          get_missing_values_config, get_outlier_config, get_path,
                          get_splitting_config)
from utils.logger import banner, get_logger
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags

logger = get_logger(__name__)

SPLIT_KEYS = ["X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test"]


def log_stage_metrics(df: pd.DataFrame, stage: str, extra: Dict = None) -> None:
    metrics = {f"{stage}_rows": df.shape[0], f"{stage}_columns": df.shape[1],
               f"{stage}_missing_values": int(df.isnull().sum().sum())}
    if extra:
        metrics.update({f"{stage}_{k}": v for k, v in extra.items()})
    MLflowTracker.log_metrics_safe(metrics)


def create_eda_visualisations(df: pd.DataFrame, target: str, out_dir: str) -> None:
    """Churn rate by key categorical drivers + numeric distributions (logged to MLflow)."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        drivers = [c for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
                   if c in df.columns]
        fig, axes = plt.subplots(1, len(drivers), figsize=(5 * len(drivers), 4))
        for ax, col in zip(axes, drivers):
            (df.groupby(col)[target].mean() * 100).sort_values().plot.barh(ax=ax, color="#E53935")
            ax.set_title(f"Churn rate by {col}"); ax.set_xlabel("%")
        fig.tight_layout()
        p1 = os.path.join(out_dir, "churn_rate_by_driver.png")
        fig.savefig(p1, dpi=120); plt.close(fig)

        nums = ["tenure", "MonthlyCharges", "TotalCharges"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col in zip(axes, nums):
            for val, colour, label in [(0, "#43A047", "No churn"), (1, "#E53935", "Churn")]:
                ax.hist(df.loc[df[target] == val, col], bins=30, alpha=0.6, color=colour, label=label)
            ax.set_title(col); ax.legend()
        fig.tight_layout()
        p2 = os.path.join(out_dir, "numeric_distributions.png")
        fig.savefig(p2, dpi=120); plt.close(fig)

        for p in (p1, p2):
            mlflow.log_artifact(p, "visualizations")
    except Exception as e:
        logger.warning(f"⚠ Could not create EDA plots: {e}")


def _artifacts_exist() -> bool:
    needed = [get_path(k) for k in SPLIT_KEYS]
    needed += [os.path.join(get_path("scale_dir"), "scaler.joblib"),
               os.path.join(get_path("model_artifacts_dir"), "feature_columns.json")]
    return all(os.path.exists(p) for p in needed)


def load_splits() -> Dict[str, pd.DataFrame]:
    splits = {k: pd.read_csv(get_path(k)) for k in SPLIT_KEYS}
    for k in ("Y_train", "Y_val", "Y_test"):
        splits[k] = splits[k].squeeze("columns")
    return splits


def data_pipeline(force_rebuild: bool = False) -> Dict[str, pd.DataFrame]:
    banner(logger, "STARTING DATA PIPELINE", 80)

    if _artifacts_exist() and not force_rebuild:
        logger.info("✓ Processed artifacts found - reusing (use --force to rebuild)")
        return load_splits()

    raw_path = get_path("raw_data")
    cols = get_columns()
    target = cols["target"]

    tracker = MLflowTracker()
    tracker.start_run("data_pipeline", create_mlflow_run_tags(
        "data_pipeline", {"data_source": os.path.basename(raw_path),
                          "force_rebuild": force_rebuild}))
    try:
        # 1. Ingest
        df = DataIngestorCSV(required_columns=cols["raw_feature_columns"] + [target]).ingest(raw_path)
        log_stage_metrics(df, "raw")
        try:
            mlflow.log_input(mlflow.data.from_pandas(df, source=raw_path, name="telco_raw",
                                                     targets=target), context="raw_data")
        except Exception as e:
            logger.warning(f"⚠ Could not log raw dataset lineage: {e}")

        # 2. Clean (TotalCharges → numeric, normalise service categories, drop ID)
        clean_cfg = get_cleaning_config()
        cleaner = TelcoDataCleaner(clean_cfg["coerce_numeric"], clean_cfg["category_normalisation"],
                                   clean_cfg["normalise_columns"], cols["drop_columns"])
        df = cleaner.clean(df)

        # 3. Missing values (the 11 tenure-0 rows with blank TotalCharges)
        before = len(df)
        mv_cfg = get_missing_values_config()
        df = DropMissingValuesStrategy(mv_cfg["critical_columns"]).handle(df)
        log_stage_metrics(df, "missing_handled", {"rows_removed": before - len(df)})

        # 4. Outliers (report-only by default)
        out_cfg = get_outlier_config()
        n_outliers = 0
        if out_cfg.get("enabled", True):
            df, n_outliers = OutlierDetector(IQROutlierDetection(out_cfg.get("iqr_multiplier", 1.5))) \
                .handle_outliers(df, out_cfg["columns"], out_cfg.get("handling_method", "report"))
        log_stage_metrics(df, "outliers_checked", {"outlier_rows": n_outliers})

        # 5. Target → 0/1 (stateless) and persist the cleaned dataset
        df[target] = encode_target(df[target], get_encoding_config()["target_mapping"])
        os.makedirs(os.path.dirname(get_path("cleaned_data")), exist_ok=True)
        df.to_csv(get_path("cleaned_data"), index=False)
        create_eda_visualisations(df, target, os.path.join(get_path("artifacts_dir"), "eda"))

        # 6. Split FIRST (stratified 60/20/20) ...
        split_cfg = get_splitting_config()
        splits = StratifiedThreeWaySplitStrategy(
            split_cfg["test_size"], split_cfg["val_size"], split_cfg["random_state"]
        ).split_data(df, target)

        # 7. ... THEN fit encoders + scaler on TRAIN only, and apply to all splits
        banner(logger, "FEATURE ENGINEERING (fit on train, transform all)")
        pre = ChurnPreprocessor().fit(splits["X_train"])
        for key in ("X_train", "X_val", "X_test"):
            splits[key] = pre.transform(splits[key], clean=False)
        pre.save()

        # 8. Save splits
        os.makedirs(get_path("data_artifacts_dir"), exist_ok=True)
        for key in SPLIT_KEYS:
            obj = splits[key]
            (obj.to_frame() if isinstance(obj, pd.Series) else obj).to_csv(get_path(key), index=False)
            mlflow.log_artifact(get_path(key), "processed_datasets")
        for d in (get_path("encode_dir"), get_path("scale_dir")):
            mlflow.log_artifacts(d, f"preprocessing/{os.path.basename(d)}")

        summary = {
            "total_samples": len(df), "num_features": splits["X_train"].shape[1],
            "train_samples": len(splits["X_train"]), "val_samples": len(splits["X_val"]),
            "test_samples": len(splits["X_test"]),
            "train_churn_rate": float(splits["Y_train"].mean()),
            "val_churn_rate": float(splits["Y_val"].mean()),
            "test_churn_rate": float(splits["Y_test"].mean()),
        }
        MLflowTracker.log_metrics_safe(summary)
        MLflowTracker.log_params_safe({
            "split": f"{1 - split_cfg['test_size'] - split_cfg['val_size']:.0%}/"
                     f"{split_cfg['val_size']:.0%}/{split_cfg['test_size']:.0%} stratified",
            "scaler_fit_on": "train_only",
            "outlier_handling": out_cfg.get("handling_method"),
            "feature_names": pre.feature_columns,
        })
        summary_path = os.path.join(get_path("data_artifacts_dir"), "data_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, "data_summary")

        tracker.end_run()
        logger.info("✓ Data pipeline completed successfully")
        return splits
    except Exception:
        tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the churn data pipeline")
    parser.add_argument("--force", action="store_true", help="Rebuild all processed artifacts")
    args = parser.parse_args()

    result = data_pipeline(force_rebuild=args.force)
    print("\n" + "=" * 60)
    print("📊 DATA PIPELINE SUMMARY")
    print("=" * 60)
    for k in ("X_train", "X_val", "X_test"):
        y = result["Y" + k[1:]]
        print(f"✅ {k:<8} {str(result[k].shape):<12} churn rate {y.mean() * 100:.1f}%")
    print("✅ Artifacts: artifacts/data, artifacts/encode, artifacts/scale")
    print("=" * 60)
