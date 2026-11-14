import numpy as np
import pytest

from ml.calibrate import (
    TemperatureScaler,
    expected_calibration_error,
    label_indices,
    nll,
    softmax,
)
from ml.conformal import SplitConformal, set_metrics

RNG = np.random.default_rng(0)


def overconfident_data(n: int = 4000, k: int = 5):
    """Scores where the right class usually wins, but scaled up so the model is too sure."""
    y = RNG.integers(0, k, size=n)
    scores = RNG.normal(0, 1, size=(n, k))
    scores[np.arange(n), y] += 1.5
    return scores * 4.0, y  # x4 makes the softmax far too confident


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_label_indices_marks_unknown_labels():
    assert list(label_indices(["32", "99", "53"], ["32", "53"])) == [0, -1, 1]


def test_temperature_fixes_overconfidence_without_changing_ranking():
    scores, y = overconfident_data()
    raw = softmax(scores)
    scaler = TemperatureScaler().fit(scores, y)
    cal = scaler.transform(scores)
    assert scaler.temperature > 1.5  # it learned to soften
    assert nll(cal, y) < nll(raw, y)
    assert expected_calibration_error(cal, y) < expected_calibration_error(raw, y)
    assert (cal.argmax(axis=1) == raw.argmax(axis=1)).all()  # accuracy unchanged


def test_conformal_hits_target_coverage_on_similar_data():
    scores, y = overconfident_data(n=8000)
    probs = TemperatureScaler().fit(scores[:2000], y[:2000]).transform(scores)
    conformal = SplitConformal(alpha=0.1).fit(probs[2000:5000], y[2000:5000])
    m = set_metrics(conformal.predict_sets(probs[5000:]), y[5000:])
    assert m["coverage"] == pytest.approx(0.90, abs=0.03)
    assert m["avg_set_size"] >= 1.0


def test_sets_are_never_empty():
    probs = softmax(RNG.normal(size=(50, 6)))
    conformal = SplitConformal(alpha=0.5)
    conformal.qhat = 0.0  # extreme threshold that would include nothing
    assert conformal.predict_sets(probs).sum(axis=1).min() == 1


def test_bad_alpha_raises():
    with pytest.raises(ValueError):
        SplitConformal(alpha=1.5)