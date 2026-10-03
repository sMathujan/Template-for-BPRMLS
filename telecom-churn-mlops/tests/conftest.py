import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "artifacts", "models", "churn_model.joblib")

requires_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="No trained model - run `make train-pipeline-fast` first")


@pytest.fixture
def sample_customer():
    return {
        "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85",
    }
