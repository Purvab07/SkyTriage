"""Cleaning rules for FAA Service Difficulty Report (SDR) CSVs.

Each step is a small function so it can be tested on its own. `clean_sdr` runs them
in order and returns clean rows, rejected rows (with a reason) and warning counts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# CSV column -> database column. Anything not listed here is dropped.
COLUMN_MAP: dict[str, str] = {
    "OperatorControlNumber": "control_number",
    "DifficultyDate": "difficulty_date",
    "SubmissionDate": "submission_date",
    "JASCCode": "jasc_code",
    "Discrepancy": "discrepancy",
    "AircraftMake": "aircraft_make",
    "AircraftModel": "aircraft_model",
    "AircraftTotalTime": "aircraft_total_time",
    "AircraftTotalCycles": "aircraft_total_cycles",
    "StageOfOperationCode": "stage_of_operation_code",
    "HowDiscoveredCode": "how_discovered_code",
    "NatureOfConditionA": "nature_of_condition_a",
    "PrecautionaryProcedureA": "precautionary_procedure_a",
    "SubmitterTypeCode": "submitter_type_code",
    "AircraftSerialNumber": "aircraft_serial_number",
    "RegistryNNumber": "registry_n_number",
    "PartNumber": "part_number",
    "PartName": "part_name",
    "PartCondition": "part_condition",
    "PartLocation": "part_location",
}

# Values above these are data-entry errors (no aircraft has flown this much).
# They are blanked, not rejected: the rest of the report is still useful.
MAX_TOTAL_HOURS = 200_000
MAX_TOTAL_CYCLES = 200_000


@dataclass
class CleanResult:
    clean: pd.DataFrame
    rejected: pd.DataFrame
    warnings: Counter = field(default_factory=Counter)


def read_sdr_csv(path: Path) -> pd.DataFrame:
    """Read every column as text, so nothing is silently converted on the way in."""
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def select_and_rename(raw: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMN_MAP) - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    return raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)


def strip_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces, collapse runs of whitespace/newlines, and turn empty strings into missing."""
    df = df.copy()
    for col in df.columns:
        s = df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        df[col] = s.mask(s == "")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["difficulty_date"] = pd.to_datetime(
        df["difficulty_date"], format="%m/%d/%Y", errors="coerce"
    ).dt.date
    df["submission_date"] = pd.to_datetime(
        df["submission_date"], errors="coerce", utc=True, format="ISO8601"
    )
    return df


def parse_numbers(df: pd.DataFrame, warnings: Counter) -> pd.DataFrame:
    df = df.copy()
    limits = {"aircraft_total_time": MAX_TOTAL_HOURS, "aircraft_total_cycles": MAX_TOTAL_CYCLES}
    for col, limit in limits.items():
        text = df[col]
        num = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
        unparsable = text.notna() & num.isna()
        out_of_range = (num < 0) | (num > limit)
        warnings[f"{col}_unparsable"] += int(unparsable.sum())
        warnings[f"{col}_out_of_range"] += int(out_of_range.sum())
        df[col] = num.mask(out_of_range).astype("Float64")
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ata_chapter"] = df["jasc_code"].str[:2]
    return df


def validate(df: pd.DataFrame, today: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into clean and rejected. Each rejected row gets the first rule it failed."""
    today = today or date.today()
    rules = [
        ("missing_control_number", df["control_number"].isna()),
        ("bad_difficulty_date", df["difficulty_date"].isna()),
        (
            "future_difficulty_date",
            df["difficulty_date"].map(lambda d: d is not None and pd.notna(d) and d > today + timedelta(days=1)),
        ),
        ("bad_jasc_code", ~df["jasc_code"].fillna("").str.fullmatch(r"\d{4}")),
        ("empty_discrepancy", df["discrepancy"].isna()),
    ]
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, failed in rules:
        reason = reason.mask(reason.isna() & failed.fillna(False).astype(bool), name)
    bad = reason.notna()
    rejected = df[bad].assign(reject_reason=reason[bad])
    return df[~bad].copy(), rejected


def deduplicate(df: pd.DataFrame, warnings: Counter) -> pd.DataFrame:
    """Keep one row per control number: the most recently submitted version."""
    before = len(df)
    df = df.sort_values("submission_date", na_position="first").drop_duplicates(
        "control_number", keep="last"
    )
    warnings["duplicates_removed"] += before - len(df)
    return df


def clean_sdr(raw: pd.DataFrame, source_file: str, today: date | None = None) -> CleanResult:
    warnings: Counter = Counter()
    df = select_and_rename(raw)
    df = strip_text(df)
    df = parse_dates(df)
    df = parse_numbers(df, warnings)
    df = add_derived(df)
    df["source_file"] = source_file
    clean, rejected = validate(df, today=today)
    clean = deduplicate(clean, warnings)
    return CleanResult(clean=clean.reset_index(drop=True), rejected=rejected, warnings=warnings)