"""Shared test helpers.

`tiny_sdr` builds a small, realistic-looking SDR table (three aircraft systems, dates
from 2024 to 2026) so end-to-end tests run in seconds without the real CSVs.
`in_tmp_project` runs a test inside an empty temporary folder, so scripts that write
to data/, models/ and eval/reports/ never touch your real project files.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pipelines.clean import COLUMN_MAP

WORDS = {
    "32": ["LANDING", "GEAR", "ACTUATOR", "RETRACT", "NLG", "MLG", "STRUT", "LEAK"],
    "33": ["LIGHT", "BULB", "INOP", "LAMP", "LANDING", "NAV", "STROBE", "REPLACED"],
    "53": ["FUSELAGE", "SKIN", "CRACK", "CORROSION", "STRINGER", "FRAME", "STA", "DOUBLER"],
}


@pytest.fixture
def tiny_sdr() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    day = date(2024, 1, 1)
    for i in range(900):  # roughly one report a day, Jan 2024 to mid 2026
        chapter = ["32", "33", "53"][i % 3]
        text = " ".join(rng.choice(WORDS[chapter], size=8))
        rows.append(
            {
                "control_number": f"TEST{i:05d}",
                "difficulty_date": day + timedelta(days=i),
                "discrepancy": f"{text} FOUND DURING CHECK",
                "ata_chapter": chapter,
                "jasc_code": f"{chapter}00",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def in_tmp_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def raw_csv_row(**overrides) -> dict:
    """One valid raw SDR CSV row (all text), keyed by the FAA column names."""
    row = {col: "" for col in COLUMN_MAP}
    row.update(
        OperatorControlNumber="ABC2026010100001",
        DifficultyDate="01/01/2026",
        SubmissionDate="2026-01-02T14:09:41.140-05:00",
        JASCCode="3230",
        Discrepancy="NLG WOULD NOT RETRACT. REPLACED SELECTOR VALVE",
        AircraftMake="BOEING",
        AircraftModel="7378",
        AircraftTotalTime="7759",
        AircraftTotalCycles="2526",
    )
    row.update(overrides)
    return row