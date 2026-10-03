"""MLflow tracking utilities: experiments, runs, model logging and registry aliases."""
import inspect
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from utils.config import PROJECT_ROOT, get_mlflow_config
from utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_tracking_uri(uri: str) -> str:
    """Anchor relative sqlite/file URIs at the project root, whatever the cwd."""
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        return "sqlite:///" + os.path.join(PROJECT_ROOT, uri[len("sqlite:///"):])
    if uri.startswith("file:./") or uri.startswith("file:mlruns"):
        return "file:" + os.path.abspath(os.path.join(PROJECT_ROOT, uri.replace("file:", "", 1)))
    return uri


class MLflowTracker:
    """Thin wrapper around MLflow for experiment management and model versioning."""

    def __init__(self):
        self.config = get_mlflow_config()
        self.setup_mlflow()

    def setup_mlflow(self) -> None:
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            _resolve_tracking_uri(self.config.get("tracking_uri", "file:./mlruns")),
        )
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = self.config.get("experiment_name", "churn_experiment")
        if mlflow.get_experiment_by_name(experiment_name) is None:
            artifact_dir = os.path.join(PROJECT_ROOT, self.config.get("artifact_location", "mlruns"))
            os.makedirs(artifact_dir, exist_ok=True)
            mlflow.create_experiment(experiment_name, artifact_location=Path(artifact_dir).as_uri())
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {tracking_uri} | experiment: {experiment_name}")

    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base = (run_name or self.config.get("run_name_prefix", "run")).replace("_", " ")
        full_name = f"{base} | {timestamp}"
        all_tags = dict(self.config.get("tags", {}))
        if tags:
            all_tags.update(tags)
        run = mlflow.start_run(run_name=full_name, tags=all_tags)
        logger.info(f"Started MLflow run: {full_name} (ID: {run.info.run_id})")
        return run

    @staticmethod
    def log_metrics_safe(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        clean = {}
        for k, v in metrics.items():
            try:
                clean[k.replace(" ", "_").replace("(", "").replace(")", "")] = float(v)
            except (TypeError, ValueError):
                continue
        if clean:
            mlflow.log_metrics(clean, step=step)

    @staticmethod
    def log_params_safe(params: Dict[str, Any], prefix: str = "") -> None:
        clean = {f"{prefix}{k}": str(v)[:500] for k, v in params.items()}
        if clean:
            mlflow.log_params(clean)

    def log_model(self, model, input_example=None, register: bool = True) -> Optional[str]:
        """Log a fitted model; register it and point the champion alias at it."""
        registry_name = self.config.get("model_registry_name", "churn_model")
        try:
            kwargs = {"sk_model": model, "input_example": input_example}
            sig_params = inspect.signature(mlflow.sklearn.log_model).parameters
            # Recent MLflow defaults to skops, which rejects XGBoost/CatBoost objects
            if "serialization_format" in sig_params:
                kwargs["serialization_format"] = "cloudpickle"
            # MLflow 3 renamed artifact_path -> name
            if "name" in sig_params:
                kwargs["name"] = "model"
            else:
                kwargs["artifact_path"] = "model"
            if register:
                kwargs["registered_model_name"] = registry_name
            info = mlflow.sklearn.log_model(**kwargs)
            logger.info(f"Model logged to MLflow: {info.model_uri}")

            if register:
                self._set_champion_alias(registry_name)
            return info.model_uri
        except Exception as e:  # never let tracking break training
            logger.warning(f"Could not log model to MLflow registry: {e}")
            return None

    def _set_champion_alias(self, registry_name: str) -> None:
        try:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{registry_name}'")
            latest = max(versions, key=lambda v: int(v.version))
            alias = self.config.get("champion_alias", "champion")
            client.set_registered_model_alias(registry_name, alias, latest.version)
            logger.info(f"Registry: {registry_name} v{latest.version} -> @{alias}")
        except Exception as e:
            logger.warning(f"Could not set registry alias: {e}")

    def load_champion_model(self):
        registry_name = self.config.get("model_registry_name", "churn_model")
        alias = self.config.get("champion_alias", "champion")
        uri = f"models:/{registry_name}@{alias}"
        logger.info(f"Loading model from registry: {uri}")
        return mlflow.sklearn.load_model(uri)

    @staticmethod
    def end_run(status: str = "FINISHED") -> None:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run ({status})")


def create_mlflow_run_tags(pipeline_type: str,
                           additional_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    tags = {"pipeline_type": pipeline_type, "timestamp": datetime.now().isoformat()}
    if additional_tags:
        tags.update({k: str(v) for k, v in additional_tags.items()})
    return tags
