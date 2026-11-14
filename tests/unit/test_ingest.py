"""Tests for the ingestion script (everything except the real database load)."""
import math

import pandas as pd
import pytest

from pipelines.clean import read_sdr_csv
from pipelines.ingest_sdr import clean_all, main, save_files, to_records
from tests.conftest import raw_csv_row


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def raw_dir(in_tmp_project):
    raw = in_tmp_project / "data" / "raw"
    raw.mkdir(parents=True)
    write_csv(raw / "SDR-2025.csv", [
        raw_csv_row(OperatorControlNumber="A1", SubmissionDate="2025-01-01T10:00:00-05:00",
                    Discrepancy="OLD VERSION"),
        raw_csv_row(OperatorControlNumber="A2"),
        raw_csv_row(OperatorControlNumber="BAD", JASCCode="XX"),
    ])
    # A1 appears again in the next year's file with a later submission: the newer one must win.
    write_csv(raw / "SDR-2026.csv", [
        raw_csv_row(OperatorControlNumber="A1", SubmissionDate="2026-03-01T10:00:00-05:00",
                    Discrepancy="NEW VERSION"),
        raw_csv_row(OperatorControlNumber="A3"),
    ])
    return raw


def test_read_csv_keeps_everything_as_text(raw_dir):
    df = read_sdr_csv(raw_dir / "SDR-2025.csv")
    assert df["AircraftTotalTime"].iloc[0] == "7759"  # not converted to a number yet
    assert df["PartName"].iloc[0] == ""  # empty stays empty text, not NaN


def test_clean_all_combines_files_dedupes_across_them_and_rejects_bad_rows(raw_dir):
    clean, rejected, warnings = clean_all(raw_dir, "SDR-*.csv")
    assert sorted(clean["control_number"]) == ["A1", "A2", "A3"]
    assert clean.set_index("control_number").loc["A1", "discrepancy"] == "NEW VERSION"
    assert clean.set_index("control_number").loc["A1", "source_file"] == "SDR-2026.csv"
    assert list(rejected["reject_reason"]) == ["bad_jasc_code"]
    assert warnings["duplicates_removed"] == 1


def test_clean_all_with_no_files_raises(in_tmp_project):
    (in_tmp_project / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        clean_all(in_tmp_project / "empty", "SDR-*.csv")


def test_save_files_writes_parquet_and_rejects(raw_dir, in_tmp_project):
    clean, rejected, _ = clean_all(raw_dir, "SDR-*.csv")
    out = in_tmp_project / "data" / "processed"
    save_files(clean, rejected, out)
    assert len(pd.read_parquet(out / "sdr.parquet")) == 3
    assert len(pd.read_csv(out / "sdr_rejected.csv")) == 1


def test_to_records_turns_every_missing_value_into_none(raw_dir):
    clean, _, _ = clean_all(raw_dir, "SDR-*.csv")
    records = to_records(clean)
    assert records[0]["part_name"] is None  # was pd.NA
    assert "control_number" in records[0]
    assert "source_file" in records[0]
    assert not any(isinstance(v, float) and math.isnan(v) for r in records for v in r.values())  # no NaN


def test_main_no_db_runs_end_to_end(raw_dir, in_tmp_project):
    main(["--no-db"])
    assert (in_tmp_project / "data" / "processed" / "sdr.parquet").exists()