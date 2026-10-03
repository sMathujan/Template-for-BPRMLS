"""Model builders for the three ensemble learners (Builder pattern, as in the template)."""
from abc import ABC, abstractmethod
from typing import Any, Dict

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class BaseModelBuilder(ABC):
    display_name: str = "Base"

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.model_params: Dict[str, Any] = kwargs
        self.model = None

    @abstractmethod
    def build_model(self):
        pass


class RandomForestModelBuilder(BaseModelBuilder):
    display_name = "Random Forest"

    def build_model(self) -> RandomForestClassifier:
        params = {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1}
        params.update(self.model_params)
        self.model = RandomForestClassifier(random_state=self.random_state, **params)
        return self.model


class XGBoostModelBuilder(BaseModelBuilder):
    display_name = "XGBoost"

    def build_model(self) -> XGBClassifier:
        params = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
                  "eval_metric": "logloss", "n_jobs": -1, "verbosity": 0}
        params.update(self.model_params)
        self.model = XGBClassifier(random_state=self.random_state, **params)
        return self.model


class CatBoostModelBuilder(BaseModelBuilder):
    display_name = "CatBoost"

    def build_model(self) -> CatBoostClassifier:
        params = {"iterations": 500, "learning_rate": 0.05, "depth": 6,
                  "loss_function": "Logloss", "verbose": False, "allow_writing_files": False}
        params.update(self.model_params)
        # Optuna tunes a single "minority_weight"; CatBoost wants class_weights
        minority_weight = params.pop("minority_weight", None)
        if minority_weight is not None:
            params["class_weights"] = [1.0, float(minority_weight)]
        self.model = CatBoostClassifier(random_state=self.random_state, **params)
        return self.model


MODEL_BUILDERS = {
    "random_forest": RandomForestModelBuilder,
    "xgboost": XGBoostModelBuilder,
    "catboost": CatBoostModelBuilder,
}


def get_model_builder(model_key: str, random_state: int = 42, **params) -> BaseModelBuilder:
    if model_key not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[model_key](random_state=random_state, **params)
