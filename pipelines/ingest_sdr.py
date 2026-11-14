"""Load FAA SDR CSVs into Postgres.

Steps: read every CSV in data/raw -> clean -> validate -> save a Parquet copy
-> insert into the sdr_reports table, skipping rows that are already there.

Run from the project folder:
    python -m pipelines.ingest_sdr            # clean + load into the database
    python -m pipelines.ingest_sdr --no-db    # clean + save files only
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, Table
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Base, SDRReport
from app.db.session import make_sync_engine
from pipelines.clean import clean_sdr, deduplicate, read_sdr_csv

log = logging.getLogger("ingest_sdr")

BATCH_SIZE = 5_000


def clean_all(raw_dir: Path, pattern: str) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {raw_dir}")

    cleans, rejects, warnings = [], [], Counter()
    for path in files:
        raw = read_sdr_csv(path)
        result = clean_sdr(raw, source_file=path.name)
        log.info(
            "%s: %s rows read, %s clean, %s rejected",
            path.name, f"{len(raw):,}", f"{len(result.clean):,}", f"{len(result.rejected):,}",
        )
        cleans.append(result.clean)
        rejects.append(result.rejected)
        warnings.update(result.warnings)

    clean = deduplicate(pd.concat(cleans, ignore_index=True), warnings)  # across files too
    rejected = pd.concat(rejects, ignore_index=True)
    return clean, rejected, warnings


def save_files(clean: pd.DataFrame, rejected: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_dir / "sdr.parquet", index=False)
    rejected.to_csv(out_dir / "sdr_rejected.csv", index=False)
    log.info("Saved %s and %s", out_dir / "sdr.parquet", out_dir / "sdr_rejected.csv")


def to_records(df: pd.DataFrame, table: Table = SDRReport.__table__) -> list[dict]:
    """Turn a DataFrame into plain dicts, with every kind of missing value as None."""
    columns = [c.name for c in table.columns if c.name in df.columns]
    obj = df[columns].astype(object)
    return obj.where(obj.notna(), None).to_dict("records")


def load_to_db(clean: pd.DataFrame, engine: Engine, table: Table = SDRReport.__table__) -> int:
    """Insert rows in batches; rows whose control_number already exists are skipped.

    `table` defaults to the real sdr_reports table; tests pass a throwaway copy.
    """
    Base.metadata.create_all(engine, tables=[table])
    records = to_records(clean, table)
    inserted = 0
    with engine.begin() as conn:
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            stmt = (
                insert(table)
                .on_conflict_do_nothing(index_elements=["control_number"])
                .returning(table.c.control_number)
            )
            inserted += len(conn.execute(stmt, batch).all())
    return inserted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean and load FAA SDR CSVs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pattern", default="SDR-*.csv")
    parser.add_argument("--no-db", action="store_true", help="Skip the database load.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()

    clean, rejected, warnings = clean_all(args.raw_dir, args.pattern)
    save_files(clean, rejected, args.out_dir)

    for name, count in sorted(warnings.items()):
        if count:
            log.warning("%s: %s", name, f"{count:,}")
    if len(rejected):
        log.warning("Rejected by reason: %s", rejected["reject_reason"].value_counts().to_dict())

    if args.no_db:
        log.info("Skipping database load (--no-db).")
    else:
        inserted = load_to_db(clean, make_sync_engine())
        log.info(
            "Loaded %s new reports (%s already in the database).",
            f"{inserted:,}", f"{len(clean) - inserted:,}",
        )

    log.info(
        "Done: %s clean, %s rejected, in %.1fs.",
        f"{len(clean):,}", f"{len(rejected):,}", time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()