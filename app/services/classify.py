"""Loads the calibrated classifier once and serves predictions.

Loading a model takes a second or two, so it happens once per process (cached), not
on every request. If no model has been trained yet, the API returns a clear 503 error
instead of crashing.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from app.config import settings
from app.schemas.triage import ChapterOption, ClassifyResponse


class ModelNotAvailableError(RuntimeError):
    """No trained model file was found."""


class ClassifierService:
    def __init__(self, model, model_version: str, confidence_threshold: float) -> None:
        self.model = model
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold

    def classify(self, text: str) -> ClassifyResponse:
        result = self.model.predict([text])[0]
        options = [ChapterOption(**option) for option in result["prediction_set"]]
        top = ChapterOption(
            chapter=result["top"], name=result["top_name"], probability=result["probability"]
        )
        return ClassifyResponse(
            top=top,
            prediction_set=options,
            coverage_target=result["coverage_target"],
            confident=len(options) == 1 and top.probability >= self.confidence_threshold,
            model_version=self.model_version,
        )


def latest_model_path(models_dir: str | Path) -> Path:
    paths = sorted(Path(models_dir).glob("ata_calibrated_*.joblib"))
    if not paths:
        raise ModelNotAvailableError(
            f"No calibrated model in '{models_dir}/'. Train one with: python -m ml.train_calibrated"
        )
    return paths[-1]  # file names contain a timestamp, so the last one is the newest


@lru_cache(maxsize=1)
def get_classifier() -> ClassifierService:
    """FastAPI dependency: returns the shared, already-loaded classifier."""
    path = latest_model_path(settings.models_dir)
    try:
        model = joblib.load(path)
    except Exception as exc:  # corrupt file, or saved by an incompatible code version
        raise ModelNotAvailableError(
            f"Could not load '{path.name}' ({type(exc).__name__}: {exc}). "
            "Retrain with: python -m ml.train_calibrated"
        ) from exc
    return ClassifierService(
        model=model,
        model_version=path.stem.removeprefix("ata_calibrated_"),
        confidence_threshold=settings.cascade_confidence_threshold,
    )