from datetime import date

import pandas as pd
import pytest

from pipelines.clean import COLUMN_MAP, clean_sdr, select_and_rename

TODAY = date(2026, 11, 13)


def make_raw(**overrides) -> pd.DataFrame:
    """One valid raw CSV row as text; override any column by its CSV name."""
    row = {col: "" for col in COLUMN_MAP}
    row.update(
        OperatorControlNumber="ABC2026010100001",
        DifficultyDate="01/01/2026",
        SubmissionDate="2026-01-02T14:09:41.140-05:00",
        JASCCode="3230",
        Discrepancy="  NLG  WOULD NOT\nRETRACT.   REPLACED SELECTOR VALVE ",
        AircraftMake="BOEING",
        AircraftModel="7378",
        AircraftTotalTime="7,759",
        AircraftTotalCycles="2526",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_valid_row_is_cleaned():
    result = clean_sdr(make_raw(), source_file="t.csv", today=TODAY)
    row = result.clean.iloc[0]
    assert len(result.rejected) == 0
    assert row["difficulty_date"] == date(2026, 1, 1)
    assert row["ata_chapter"] == "32"
    assert row["discrepancy"] == "NLG WOULD NOT RETRACT. REPLACED SELECTOR VALVE"
    assert row["aircraft_total_time"] == 7759
    assert pd.isna(row["part_name"])  # empty text becomes missing
    assert row["source_file"] == "t.csv"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"OperatorControlNumber": ""}, "missing_control_number"),
        ({"DifficultyDate": "13/45/2026"}, "bad_difficulty_date"),
        ({"DifficultyDate": "01/01/2030"}, "future_difficulty_date"),
        ({"JASCCode": "32A"}, "bad_jasc_code"),
        ({"Discrepancy": "   "}, "empty_discrepancy"),
    ],
)
def test_bad_rows_are_rejected_with_reason(overrides, reason):
    result = clean_sdr(make_raw(**overrides), source_file="t.csv", today=TODAY)
    assert len(result.clean) == 0
    assert result.rejected.iloc[0]["reject_reason"] == reason


def test_impossible_hours_are_blanked_not_rejected():
    result = clean_sdr(make_raw(AircraftTotalTime="7228621"), source_file="t.csv", today=TODAY)
    assert len(result.clean) == 1
    assert pd.isna(result.clean.iloc[0]["aircraft_total_time"])
    assert result.warnings["aircraft_total_time_out_of_range"] == 1


def test_duplicates_keep_latest_submission():
    raw = pd.concat(
        [
            make_raw(SubmissionDate="2026-01-02T10:00:00-05:00", Discrepancy="OLD TEXT"),
            make_raw(SubmissionDate="2026-01-05T10:00:00-05:00", Discrepancy="NEW TEXT"),
        ],
        ignore_index=True,
    )
    result = clean_sdr(raw, source_file="t.csv", today=TODAY)
    assert len(result.clean) == 1
    assert result.clean.iloc[0]["discrepancy"] == "NEW TEXT"
    assert result.warnings["duplicates_removed"] == 1


def test_missing_column_raises():
    with pytest.raises(ValueError, match="JASCCode"):
        select_and_rename(make_raw().drop(columns=["JASCCode"]))