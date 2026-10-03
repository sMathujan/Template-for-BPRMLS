"""Central configuration access. Every module reads settings through here."""
import os
import logging
from functools import lru_cache
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.environ.get("CHURN_CONFIG", os.path.join(PROJECT_ROOT, "config.yaml"))


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_FILE}")
        raise


def resolve_path(relative_path: str) -> str:
    """Turn a config path (relative to project root) into an absolute path."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


def _section(name: str) -> Dict[str, Any]:
    return load_config().get(name, {})


def get_project_config():          return _section("project")
def get_data_paths():              return _section("data_paths")
def get_columns():                 return _section("columns")
def get_cleaning_config():         return _section("data_cleaning")
def get_missing_values_config():   return _section("missing_values")
def get_outlier_config():          return _section("outlier_detection")
def get_binning_config():          return _section("feature_binning")
def get_encoding_config():         return _section("feature_encoding")
def get_scaling_config():          return _section("feature_scaling")
def get_splitting_config():        return _section("data_splitting")
def get_imbalance_config():        return _section("class_imbalance")
def get_training_config():         return _section("training")
def get_model_config():            return _section("model")
def get_threshold_config():        return _section("threshold_optimisation")
def get_business_config():         return _section("business")
def get_evaluation_config():       return _section("evaluation")
def get_inference_config():        return _section("inference")
def get_logging_config():          return _section("logging")
def get_mlflow_config():           return _section("mlflow")
def get_api_config():              return _section("api")


def get_path(key: str) -> str:
    """Absolute path for a key in the data_paths section."""
    return resolve_path(get_data_paths()[key])


def get_config() -> Dict[str, Any]:
    return load_config()
