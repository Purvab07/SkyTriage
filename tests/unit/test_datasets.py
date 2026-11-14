from datetime import date

import pandas as pd
import pytest

from eval.eval_classifier import evaluate, majority_baseline
from ml.datasets import ata_name, time_split


def make_df() -> pd.DataFrame:
    dates = [date(2024, 3, 1), date(2025, 6, 30), date(2025, 7, 1), date(2025, 12, 31),
             date(2026, 1, 1), date(2026, 5, 5)]
    return pd.DataFrame(
        {
            "difficulty_date": dates,
            "discrepancy": [f"REPORT {i}" for i in range(len(dates))],
            "ata_chapter": ["53", "32", "53", "33", "32", "53"],
        }
    )


def test_time_split_boundaries():
    s = time_split(make_df(), val_start="2025-07-01", test_start="2026-01-01")
    assert list(s.train["difficulty_date"]) == [date(2024, 3, 1), date(2025, 6, 30)]
    assert list(s.val["difficulty_date"]) == [date(2025, 7, 1), date(2025, 12, 31)]
    assert list(s.test["difficulty_date"]) == [date(2026, 1, 1), date(2026, 5, 5)]


def test_no_future_leaks_into_training():
    s = time_split(make_df())
    assert max(s.train["difficulty_date"]) < min(s.val["difficulty_date"])
    assert max(s.val["difficulty_date"]) < min(s.test["difficulty_date"])


def test_bad_split_dates_raise():
    with pytest.raises(ValueError):
        time_split(make_df(), val_start="2026-01-01", test_start="2025-07-01")


def test_majority_baseline_and_metrics():
    y_true = ["53", "53", "32", "33"]
    y_pred = majority_baseline(["53", "53", "32"], len(y_true))
    m = evaluate(y_true, y_pred)
    assert y_pred == ["53"] * 4
    assert m["accuracy"] == 0.5
    assert m["macro_f1"] < m["accuracy"]  # rare chapters drag macro-F1 down
    assert ("32", "53", 1) in m["top_confusions"]


def test_ata_name_fallback():
    assert ata_name("32") == "Landing Gear"
    assert ata_name("99") == "ATA 99"