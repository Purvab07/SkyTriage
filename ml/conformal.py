"""Conformal prediction: a short list of likely answers with a guaranteed hit rate.

Instead of one guess ("ATA 32"), return a set ("ATA 32 or 29") that contains the true
answer at least (1 - alpha) of the time, e.g. 90% for alpha = 0.1. The guarantee holds
for new data that looks like the calibration data.

Method (split conformal, "LAC"):
  1. On held-out data, score each example by how little probability the true class got:
     score = 1 - p(true class).
  2. Take a high quantile of those scores (qhat).
  3. For a new report, include every class with p >= 1 - qhat.
A confident prediction gives a set of one; an unsure one gives a longer set. The set size
is itself a useful signal (the cascade can send long-set cases to the LLM or a human).
"""
from __future__ import annotations

import math

import numpy as np


class SplitConformal:
    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.qhat: float | None = None

    def fit(self, probs: np.ndarray, y_idx: np.ndarray) -> SplitConformal:
        known = y_idx >= 0
        scores = 1.0 - probs[np.flatnonzero(known), y_idx[known]]
        n = len(scores)
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """Boolean matrix: True where a class is in the set. The top class is always kept."""
        if self.qhat is None:
            raise RuntimeError("Call fit() first")
        sets = probs >= 1.0 - self.qhat
        sets[np.arange(len(probs)), probs.argmax(axis=1)] = True  # never return an empty set
        return sets


def set_metrics(sets: np.ndarray, y_idx: np.ndarray) -> dict:
    """Coverage = how often the true answer is in the set (target 1 - alpha)."""
    rows = np.arange(len(y_idx))
    known = y_idx >= 0
    covered = np.zeros(len(y_idx), dtype=bool)
    covered[known] = sets[rows[known], y_idx[known]]
    sizes = sets.sum(axis=1)
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "singleton_accuracy": float(covered[sizes == 1].mean()) if (sizes == 1).any() else None,
    }