"""Database tests: need Postgres running (docker compose up -d db).

They are skipped automatically when the database is not reachable, so `pytest` still
passes on a machine without Docker. They write to their own throwaway table
(sdr_reports_test), never to your real sdr_reports data.
"""
import pandas as pd
import pytest
from sqlalchemy import MetaData, Table, func, inspect, select, text
from sqlalchemy.exc import OperationalError

from app.db.models import SDRReport
from app.db.session import make_sync_engine, sync_url
from pipelines.ingest_sdr import load_to_db


def test_sync_url_swaps_driver():
    assert sync_url("postgresql+asyncpg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"


@pytest.fixture
def engine():
    eng = make_sync_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip("Postgres is not running (start it with: docker compose up -d db)")
    yield eng
    eng.dispose()


@pytest.fixture
def test_table(engine):
    """A throwaway copy of sdr_reports with the same columns, in its own metadata."""
    meta = MetaData()
    table = Table("sdr_reports_test", meta, *[c._copy() for c in SDRReport.__table__.columns])
    meta.drop_all(engine)
    yield table
    meta.drop_all(engine)


def sample_rows(ids: list[str], hours: list[float | None]) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "control_number": ids,
            "difficulty_date": [pd.Timestamp("2026-01-01").date()] * n,
            "submission_date": [pd.Timestamp("2026-01-02", tz="UTC")] * n,
            "jasc_code": ["3230"] * n,
            "ata_chapter": ["32"] * n,
            "discrepancy": ["NLG WOULD NOT RETRACT"] * n,
            "aircraft_total_time": pd.array(hours, dtype="Float64"),
            "source_file": ["t.csv"] * n,
        }
    )


def count_rows(engine, table) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


def test_load_is_idempotent_and_stores_missing_as_null(engine, test_table):
    has_real = inspect(engine).has_table("sdr_reports")
    real_before = count_rows(engine, SDRReport.__table__) if has_real else 0

    assert load_to_db(sample_rows(["A", "B"], [7759.0, None]), engine, test_table) == 2
    assert load_to_db(sample_rows(["A", "B", "C"], [1.0, 2.0, 3.0]), engine, test_table) == 1

    with engine.connect() as conn:
        rows = conn.execute(
            select(test_table.c.control_number, test_table.c.aircraft_total_time)
            .order_by(test_table.c.control_number)
        ).all()
    assert [r[0] for r in rows] == ["A", "B", "C"]
    assert rows[0][1] == 7759.0  # first version kept; re-running does not overwrite
    assert rows[1][1] is None  # missing number stored as NULL, not NaN

    real_after = count_rows(engine, SDRReport.__table__) if has_real else 0
    assert real_after == real_before  # the real table was not touched