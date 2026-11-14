"""Calibration: make "90% sure" actually mean right about 90% of the time.

Raw classifiers are often over- or under-confident. Temperature scaling fixes this
with a single number T learned on held-out data:

    calibrated_probs = softmax(scores / T)

T > 1 softens over-confident predictions; T < 1 sharpens timid ones. It never changes
which class is ranked first, so accuracy stays the same; only the confidence moves.
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max(axis=1, keepdims=True)  # subtract max for numerical safety
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def label_indices(y: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """Map labels to column positions. Labels the model never saw get -1."""
    pos = {c: i for i, c in enumerate(classes)}
    return np.array([pos.get(label, -1) for label in y])


def nll(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Negative log-likelihood: lower means the true class got more probability."""
    known = y_idx >= 0
    p_true = probs[np.flatnonzero(known), y_idx[known]]
    return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy, over confidence bins.

    Example: if predictions made with ~80% confidence are right 70% of the time,
    that bin contributes a 10-point gap. 0 = perfectly calibrated.
    """
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_idx
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, scores: np.ndarray, y_idx: np.ndarray) -> TemperatureScaler:
        """Pick T that minimises NLL on held-out data (searched on log scale)."""

        def loss(log_t: float) -> float:
            return nll(softmax(scores / np.exp(log_t)), y_idx)

        result = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return softmax(scores / self.temperature)