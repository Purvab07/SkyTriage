"""The saved model bundle: classifier + temperature + conformal threshold.

This class lives in its own module on purpose. A saved (pickled) object records the
module its class came from. If the class were defined in a script run with
`python -m ...`, that module would be recorded as `__main__`, and loading the file
anywhere else (like the API) would fail. Keeping it here makes saved models portable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.pipeline import Pipeline

from ml.calibrate import softmax
from ml.datasets import ata_name


@dataclass
class CalibratedClassifier:
    """Everything needed at prediction time, saved as one file."""

    pipeline: Pipeline
    temperature: float
    conformal_qhat: float
    alpha: float
    meta: dict = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return softmax(self.pipeline.decision_function(texts) / self.temperature)

    def predict(self, texts: list[str]) -> list[dict]:
        """For each text: top chapter, calibrated probability, and the conformal set."""
        probs = self.predict_proba(texts)
        out = []
        for p in probs:
            order = np.argsort(p)[::-1]
            in_set = [i for i in order if p[i] >= 1.0 - self.conformal_qhat] or [order[0]]
            out.append(
                {
                    "top": self.classes[order[0]],
                    "top_name": ata_name(self.classes[order[0]]),
                    "probability": round(float(p[order[0]]), 4),
                    "prediction_set": [
                        {"chapter": self.classes[i], "name": ata_name(self.classes[i]),
                         "probability": round(float(p[i]), 4)}
                        for i in in_set
                    ],
                    "coverage_target": 1 - self.alpha,
                }
            )
        return out