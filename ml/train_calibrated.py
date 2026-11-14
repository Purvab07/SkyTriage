"""Train the baseline classifier with calibrated confidence and conformal sets.

Data use (each piece of data has one job, so no score is measured on data it was tuned on):
  train (Jan 2024 - Jun 2025)  -> fit the classifier
  val, first half (by date)    -> fit the temperature T          (calibration)
  val, second half (by date)   -> fit the conformal threshold    (prediction sets)
  test (2026)                  -> measure everything, once

Run from the project folder:
    python -m ml.train_calibrated
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from ml.calibrate import TemperatureScaler, expected_calibration_error, label_indices, nll, softmax
from ml.conformal import SplitConformal, set_metrics
from ml.datasets import DATE_COL, LABEL_COL, TEXT_COL, ata_name, load_sdr, time_split
from ml.train_baseline import build_pipeline

log = logging.getLogger("train_calibrated")

MODEL_DIR = Path("models")
REPORT_DIR = Path("eval/reports")


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


def latest_baseline_alpha(default: float = 2e-7) -> float:
    metas = sorted(MODEL_DIR.glob("ata_baseline_*.json"))
    if not metas:
        return default
    return float(json.loads(metas[-1].read_text())["alpha"])


def write_report(m: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Calibration report ({m['version']})",
        "",
        f"Data: {m['splits']}",
        (f"Classifier alpha: {m['alpha']} | temperature T = {m['temperature']:.3f} "
        f"| conformal alpha = {m['conformal_alpha']} (target coverage {1 - m['conformal_alpha']:.0%})"),
        "",
        "## Calibration on the 2026 test set",
        "",
        "| | ECE (lower is better) | NLL (lower is better) |",
        "| --- | --- | --- |",
        f"| Before (T = 1) | {m['test_ece_before']:.4f} | {m['test_nll_before']:.4f} |",
        f"| After (T = {m['temperature']:.2f}) | {m['test_ece_after']:.4f} | {m['test_nll_after']:.4f} |",
        "",
        f"Accuracy is unchanged by calibration: {m['test_accuracy']:.3f}",
        "",
        "## Conformal prediction sets on the 2026 test set",
        "",
        "| Coverage | Target | Avg set size | Single-answer rate | Accuracy when single |",
        "| --- | --- | --- | --- | --- |",
        (f"| {m['sets']['coverage']:.3f} | {1 - m['conformal_alpha']:.2f} | {m['sets']['avg_set_size']:.2f} "
        f"| {m['sets']['singleton_rate']:.3f} | {m['sets']['singleton_accuracy']:.3f} |"),
        "",
        "Coverage a little below target on 2026 data would point to time drift: the",
        "guarantee assumes new reports look like the calibration reports.",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a calibrated ATA classifier.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Classifier regularisation; default = best from the latest baseline run.")
    parser.add_argument("--conformal-alpha", type=float, default=0.1,
                        help="Allowed miss rate for prediction sets (0.1 = 90%% coverage).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    alpha = args.alpha if args.alpha is not None else latest_baseline_alpha()
    splits = time_split(load_sdr())
    val = splits.val.sort_values(DATE_COL)
    half = len(val) // 2
    cal_t, cal_conf = val.iloc[:half], val.iloc[half:]
    log.info("Splits: %s | val split into %s (temperature) + %s (conformal)",
             splits.summary(), f"{len(cal_t):,}", f"{len(cal_conf):,}")

    started = time.perf_counter()
    pipeline = build_pipeline(alpha).fit(splits.train[TEXT_COL], splits.train[LABEL_COL])
    classes = list(pipeline.classes_)
    log.info("Trained classifier (alpha=%g) in %.0fs", alpha, time.perf_counter() - started)

    def scores_and_labels(df):
        return pipeline.decision_function(df[TEXT_COL]), label_indices(df[LABEL_COL], classes)

    s_t, y_t = scores_and_labels(cal_t)
    s_c, y_c = scores_and_labels(cal_conf)
    s_test, y_test = scores_and_labels(splits.test)

    scaler = TemperatureScaler().fit(s_t, y_t)
    conformal = SplitConformal(alpha=args.conformal_alpha).fit(scaler.transform(s_c), y_c)

    raw_test, cal_test = softmax(s_test), scaler.transform(s_test)
    metrics = {
        "version": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        "alpha": alpha,
        "temperature": scaler.temperature,
        "conformal_alpha": args.conformal_alpha,
        "conformal_qhat": conformal.qhat,
        "splits": splits.summary(),
        "test_accuracy": float((cal_test.argmax(axis=1) == y_test).mean()),
        "test_ece_before": expected_calibration_error(raw_test, y_test),
        "test_ece_after": expected_calibration_error(cal_test, y_test),
        "test_nll_before": nll(raw_test, y_test),
        "test_nll_after": nll(cal_test, y_test),
        "sets": set_metrics(conformal.predict_sets(cal_test), y_test),
    }
    log.info("T=%.3f | ECE %.4f -> %.4f | coverage %.3f (target %.2f) | avg set size %.2f",
             metrics["temperature"], metrics["test_ece_before"], metrics["test_ece_after"],
             metrics["sets"]["coverage"], 1 - args.conformal_alpha, metrics["sets"]["avg_set_size"])

    bundle = CalibratedClassifier(
        pipeline=pipeline, temperature=scaler.temperature, conformal_qhat=conformal.qhat,
        alpha=args.conformal_alpha, meta=metrics,
    )
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"ata_calibrated_{metrics['version']}.joblib"
    joblib.dump(bundle, model_path)
    model_path.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    report = write_report(metrics, REPORT_DIR / f"calibration_{metrics['version']}.md")
    log.info("Saved %s and %s", model_path, report)


if __name__ == "__main__":
    main()