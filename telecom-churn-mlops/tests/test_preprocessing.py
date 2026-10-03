"""Unit tests for the stateless cleaning/encoding steps and the splitter."""
import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import TelcoDataCleaner
from src.data_splitter import StratifiedThreeWaySplitStrategy
from src.feature_encoding import BinaryEncodingStrategy, NominalEncodingStrategy, encode_target


@pytest.fixture
def cleaner():
    return TelcoDataCleaner(["TotalCharges"],
                            {"No internet service": "No", "No phone service": "No"},
                            ["OnlineSecurity", "MultipleLines"], ["customerID"])


def test_cleaner_coerces_blank_totalcharges_to_nan(cleaner):
    df = pd.DataFrame({"customerID": ["a", "b"], "TotalCharges": ["12.5", " "],
                       "OnlineSecurity": ["Yes", "No"], "MultipleLines": ["No", "Yes"]})
    out = cleaner.clean(df, verbose=False)
    assert out["TotalCharges"].iloc[0] == 12.5
    assert np.isnan(out["TotalCharges"].iloc[1])
    assert "customerID" not in out.columns


def test_cleaner_normalises_service_categories(cleaner):
    df = pd.DataFrame({"TotalCharges": [1, 2], "OnlineSecurity": ["No internet service", "Yes"],
                       "MultipleLines": ["No phone service", "Yes"]})
    out = cleaner.clean(df, verbose=False)
    assert out["OnlineSecurity"].tolist() == ["No", "Yes"]
    assert out["MultipleLines"].tolist() == ["No", "Yes"]


def test_binary_encoder_rejects_unknown_values():
    enc = BinaryEncodingStrategy({"Partner": {"Yes": 1, "No": 0}})
    with pytest.raises(ValueError, match="unexpected values"):
        enc.encode(pd.DataFrame({"Partner": ["Yes", "Maybe"]}), verbose=False)


def test_nominal_encoder_matches_get_dummies_drop_first(tmp_path):
    df = pd.DataFrame({"Contract": ["Two year", "Month-to-month", "One year", "Month-to-month"]})
    enc = NominalEncodingStrategy(["Contract"], drop_first=True, encode_dir=str(tmp_path)).fit(df)
    ours = enc.encode(df, verbose=False)
    ref = pd.get_dummies(df, columns=["Contract"], drop_first=True).astype(int)
    pd.testing.assert_frame_equal(ours, ref, check_dtype=False)


def test_nominal_encoder_round_trip(tmp_path):
    df = pd.DataFrame({"Contract": ["Two year", "Month-to-month", "One year"]})
    NominalEncodingStrategy(["Contract"], encode_dir=str(tmp_path)).fit(df).save()
    loaded = NominalEncodingStrategy(["Contract"], encode_dir=str(tmp_path)).load()
    assert loaded.categories["Contract"] == ["Month-to-month", "One year", "Two year"]


def test_encode_target():
    assert encode_target(pd.Series(["Yes", "No"]), {"Yes": 1, "No": 0}).tolist() == [1, 0]


def test_three_way_split_is_stratified_and_disjoint():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=1000), "Churn": (rng.random(1000) < 0.27).astype(int)})
    s = StratifiedThreeWaySplitStrategy(0.2, 0.2, 42).split_data(df, "Churn")
    assert (len(s["X_train"]), len(s["X_val"]), len(s["X_test"])) == (600, 200, 200)
    rates = [s[k].mean() for k in ("Y_train", "Y_val", "Y_test")]
    assert max(rates) - min(rates) < 0.02
    assert not set(s["X_train"].index) & set(s["X_test"].index)
