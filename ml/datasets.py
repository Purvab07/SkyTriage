"""Load the cleaned SDR data and split it by time.

Why split by time and not randomly? In real use the model always sees reports from
the future. A random split lets it peek at reports from the same week (often from the
same heavy check on the same aircraft), which makes scores look better than they are.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/processed/sdr.parquet")

TEXT_COL = "discrepancy"
LABEL_COL = "ata_chapter"
DATE_COL = "difficulty_date"

# Default split: train on older reports, tune on the next six months, test on the newest.
VAL_START = "2025-07-01"
TEST_START = "2026-01-01"

# Readable names for the most common chapters. Unknown chapters fall back to "ATA xx".
ATA_NAMES: dict[str, str] = {
    "21": "Air Conditioning",
    "22": "Auto Flight",
    "23": "Communications",
    "24": "Electrical Power",
    "25": "Equipment/Furnishings",
    "26": "Fire Protection",
    "27": "Flight Controls",
    "28": "Fuel",
    "29": "Hydraulic Power",
    "30": "Ice and Rain Protection",
    "31": "Indicating/Recording",
    "32": "Landing Gear",
    "33": "Lights",
    "34": "Navigation",
    "35": "Oxygen",
    "36": "Pneumatic",
    "38": "Water/Waste",
    "49": "Auxiliary Power Unit",
    "52": "Doors",
    "53": "Fuselage",
    "54": "Nacelles/Pylons",
    "55": "Stabilizers",
    "56": "Windows",
    "57": "Wings",
    "61": "Propellers",
    "71": "Powerplant",
    "72": "Engine",
    "73": "Engine Fuel and Control",
    "74": "Ignition",
    "75": "Engine Air",
    "76": "Engine Controls",
    "77": "Engine Indicating",
    "78": "Exhaust",
    "79": "Engine Oil",
    "80": "Starting",
}


def ata_name(chapter: str) -> str:
    return ATA_NAMES.get(chapter, f"ATA {chapter}")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        def span(df: pd.DataFrame) -> str:
            return f"{df[DATE_COL].min()} to {df[DATE_COL].max()}"

        return (
            f"train {len(self.train):,} ({span(self.train)}) | "
            f"val {len(self.val):,} ({span(self.val)}) | "
            f"test {len(self.test):,} ({span(self.test)})"
        )


def load_sdr(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m pipelines.ingest_sdr")
    df = pd.read_parquet(path, columns=None)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
    return df.dropna(subset=[TEXT_COL, LABEL_COL])


def time_split(
    df: pd.DataFrame, val_start: str = VAL_START, test_start: str = TEST_START
) -> Splits:
    """Split by difficulty date: train < val_start <= val < test_start <= test."""
    dates = pd.to_datetime(df[DATE_COL])
    val_start_ts, test_start_ts = pd.Timestamp(val_start), pd.Timestamp(test_start)
    if not val_start_ts < test_start_ts:
        raise ValueError("val_start must be before test_start")
    train = df[dates < val_start_ts]
    val = df[(dates >= val_start_ts) & (dates < test_start_ts)]
    test = df[dates >= test_start_ts]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("A split is empty; check the dates against the data range.")
    return Splits(train=train.copy(), val=val.copy(), test=test.copy())
