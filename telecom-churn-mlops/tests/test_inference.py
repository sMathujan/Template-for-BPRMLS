"""Integration tests against the trained artifacts and the API (skipped if untrained)."""
import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_model


@requires_model
def test_single_prediction(sample_customer):
    from src.model_inference import ModelInference
    r = ModelInference().predict(sample_customer)
    assert 0.0 <= r["churn_probability"] <= 1.0
    assert r["prediction"] == int(r["churn_probability"] >= r["threshold"])
    assert r["risk_band"] in {"High", "Medium", "Low"}


@requires_model
def test_blank_totalcharges_is_handled(sample_customer):
    from src.model_inference import ModelInference
    sample_customer.update(tenure=0, TotalCharges=" ")
    assert 0.0 <= ModelInference().predict(sample_customer)["churn_probability"] <= 1.0


@requires_model
def test_features_match_training_columns(sample_customer):
    from src.preprocessing import ChurnPreprocessor
    import pandas as pd
    pre = ChurnPreprocessor().load()
    X = pre.transform(pd.DataFrame([sample_customer]))
    assert list(X.columns) == pre.feature_columns


@requires_model
def test_missing_field_raises(sample_customer):
    from src.model_inference import ModelInference
    sample_customer.pop("Contract")
    with pytest.raises(ValueError, match="missing required fields"):
        ModelInference().predict(sample_customer)


@requires_model
def test_api_endpoints(sample_customer):
    from api.main import app
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert "threshold" in client.get("/model-info").json()

        r = client.post("/predict", json=sample_customer)
        assert r.status_code == 200 and r.json()["status"] in {"Churn", "Retain"}

        r = client.post("/predict/batch", json={"records": [sample_customer] * 3})
        assert r.status_code == 200 and r.json()["n_records"] == 3

        bad = dict(sample_customer, Contract="Three year")
        assert client.post("/predict", json=bad).status_code == 422
