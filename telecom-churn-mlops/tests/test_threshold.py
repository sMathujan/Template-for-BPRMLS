"""Tests for threshold selection and the business-value calculation."""
import numpy as np
import pytest

from src.threshold_optimisation import BusinessEconomics, ThresholdOptimiser


def test_expected_value_reproduces_readme_figures():
    """README: at 0.50 → 247 caught, 439 offers → £79,249; at 0.37 → 300, 610 → £88,650."""
    econ = BusinessEconomics(1656.0, 99.0, 0.30)
    assert econ.value_per_saved_churner == pytest.approx(496.80)

    def ev(tp, contacted):
        y_true = np.array([1] * tp + [0] * (contacted - tp))
        return econ.expected_value(y_true, np.ones(contacted, dtype=int))

    assert ev(247, 439) == pytest.approx(79_248.6)
    assert ev(300, 610) == pytest.approx(88_650.0)


def test_f1_threshold_finds_separating_cutoff():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    thr, score = ThresholdOptimiser("f1").find_best_threshold(y, p)
    assert 0.3 < thr <= 0.6 and score == pytest.approx(1.0)


def test_expected_value_prefers_lower_threshold_when_misses_are_expensive():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.27).astype(int)
    p = np.clip(0.27 + 0.35 * (y - 0.27) + rng.normal(0, 0.18, 2000), 0, 1)
    f1_thr, _ = ThresholdOptimiser("f1").find_best_threshold(y, p)
    ev_thr, _ = ThresholdOptimiser("expected_value").find_best_threshold(y, p)
    assert ev_thr <= f1_thr


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        ThresholdOptimiser("accuracy")
