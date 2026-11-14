"""Tests for data loading, reports, the trained models and the training scripts.

The training scripts run end to end on the tiny synthetic dataset from conftest.py, inside
a temporary folder, so these tests are fast and never touch your real models or reports.
"""
import json

import joblib
import numpy as np
import pytest

from eval import eval_classifier
from eval.eval_classifier import evaluate, majority_baseline, write_report
from ml import train_baseline, train_calibrated
from ml.conformal import SplitConformal
from ml.datasets import load_sdr, time_split


def save_parquet(df, root):
    path = root / "data" / "processed" / "sdr.parquet"
    path.parent.mkdir(parents=True)
    df.to_parquet(path, index=False)
    return path


# ---------- ml/datasets.py ----------

def test_load_sdr_reads_parquet_and_drops_rows_without_text_or_label(tiny_sdr, in_tmp_project):
    tiny_sdr.loc[0, "discrepancy"] = None
    tiny_sdr.loc[1, "ata_chapter"] = None
    df = load_sdr(save_parquet(tiny_sdr, in_tmp_project))
    assert len(df) == len(tiny_sdr) - 2


def test_load_sdr_missing_file_explains_how_to_fix(in_tmp_project):
    with pytest.raises(FileNotFoundError, match="ingest_sdr"):
        load_sdr()


def test_split_summary_and_empty_split(tiny_sdr):
    s = time_split(tiny_sdr)
    assert "train" in s.summary() and "test" in s.summary()
    with pytest.raises(ValueError, match="empty"):
        time_split(tiny_sdr, val_start="2030-01-01", test_start="2031-01-01")


# ---------- eval/eval_classifier.py ----------

def test_perfect_predictions_score_one():
    m = evaluate(["32", "33", "53"], ["32", "33", "53"])
    assert m["macro_f1"] == 1.0 and m["accuracy"] == 1.0
    assert m["top_confusions"] == []


def test_predicted_labels_missing_from_truth_do_not_count_as_classes():
    m = evaluate(["32", "32"], ["32", "99"])
    assert set(m["per_class"]) == {"32"}


def test_write_report_contains_table_and_mixups(in_tmp_project):
    y_true, y_pred = ["32", "33", "53", "53"], ["32", "53", "53", "53"]
    path = write_report(
        evaluate(y_true, y_pred), evaluate(y_true, majority_baseline(y_true, 4)),
        {"model": "demo", "version": "v1", "splits": "x"}, in_tmp_project / "r.md",
    )
    text = path.read_text()
    assert "| Always majority class |" in text
    assert "Landing Gear" in text
    assert "33 Lights -> 53 Fuselage: 1" in text


# ---------- ml/conformal.py ----------

def test_conformal_must_be_fitted_first():
    with pytest.raises(RuntimeError, match="fit"):
        SplitConformal().predict_sets(np.ones((1, 3)) / 3)


# ---------- training scripts, end to end ----------

def test_train_baseline_end_to_end(tiny_sdr, in_tmp_project):
    save_parquet(tiny_sdr, in_tmp_project)
    train_baseline.main(["--alphas", "1e-4"])
    meta = json.loads(next((in_tmp_project / "models").glob("ata_baseline_*.json")).read_text())
    assert meta["test"]["accuracy"] > 0.9  # synthetic classes are easy to separate
    assert meta["test"]["macro_f1"] > meta["always_majority"]["macro_f1"]
    assert next((in_tmp_project / "eval" / "reports").glob("baseline_*.md")).exists()

    # The standalone scoring command works on the saved model too.
    model_path = next((in_tmp_project / "models").glob("ata_baseline_*.joblib"))
    eval_classifier.main([str(model_path)])
    assert model_path.with_suffix(".report.md").exists()


def test_train_calibrated_end_to_end_and_predict(tiny_sdr, in_tmp_project):
    save_parquet(tiny_sdr, in_tmp_project)
    train_calibrated.main(["--alpha", "1e-4"])
    model_path = next((in_tmp_project / "models").glob("ata_calibrated_*.joblib"))
    model = joblib.load(model_path)
    meta = json.loads(model_path.with_suffix(".json").read_text())

    assert meta["temperature"] > 0
    assert 0 <= meta["test_ece_after"] <= 1
    assert meta["sets"]["coverage"] >= 0.85

    probs = model.predict_proba(["MLG ACTUATOR LEAK", "NAV LIGHT INOP"])
    assert np.allclose(probs.sum(axis=1), 1.0)

    result = model.predict(["MLG ACTUATOR RETRACT LEAK", "FUSELAGE SKIN CRACK AT STRINGER"])
    assert result[0]["top"] == "32" and result[0]["top_name"] == "Landing Gear"
    assert result[1]["top"] == "53"
    for r in result:
        assert len(r["prediction_set"]) >= 1  # never empty
        assert r["prediction_set"][0]["chapter"] == r["top"]  # best answer listed first
        assert 0 < r["probability"] <= 1


def test_latest_baseline_alpha_defaults_when_no_baseline_exists(in_tmp_project):
    assert train_calibrated.latest_baseline_alpha(default=3e-7) == 3e-7


def test_model_trained_from_command_line_loads_in_another_process(tiny_sdr, in_tmp_project):
    """Regression test: train exactly like a user does (`python -m ...`), then load the file
    the way the API does. This failed when the model class lived in the script itself,
    because pickle recorded it as `__main__.CalibratedClassifier`."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    save_parquet(tiny_sdr, in_tmp_project)
    project_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(project_root)}
    subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "ml.train_calibrated", "--alpha", "1e-4"],
        cwd=in_tmp_project, env=env, check=True, capture_output=True,
    )
    model_path = next((in_tmp_project / "models").glob("ata_calibrated_*.joblib"))
    model = joblib.load(model_path)
    assert type(model).__module__ == "ml.model_bundle"
    assert model.predict(["MLG ACTUATOR LEAK"])[0]["top"] == "32"