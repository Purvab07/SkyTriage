"""Database tables.

Only the SDR table exists for now. ADs, aircraft, users, feedback and the audit log
are added in later phases.
"""
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Parent class for every table, so they can all be created together."""


class SDRReport(Base):
    """One FAA Service Difficulty Report, cleaned.

    Column roles (see docs/data_notes.md):
      label    -> jasc_code / ata_chapter
      features -> discrepancy, aircraft type, total time/cycles, stage, how discovered
      keys     -> control_number, serial number, tail number, submitter type, part number
      context  -> part name/condition/location, precautionary procedure
    """

    __tablename__ = "sdr_reports"

    # Key: one row per report
    control_number: Mapped[str] = mapped_column(String(40), primary_key=True)

    # Dates
    difficulty_date: Mapped[date] = mapped_column(Date, index=True)
    submission_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Label
    jasc_code: Mapped[str] = mapped_column(String(4))
    ata_chapter: Mapped[str] = mapped_column(String(2), index=True)

    # Main feature: the free-text description
    discrepancy: Mapped[str] = mapped_column(Text)

    # Aircraft type and usage (features)
    aircraft_make: Mapped[str | None] = mapped_column(String)
    aircraft_model: Mapped[str | None] = mapped_column(String)
    aircraft_total_time: Mapped[float | None] = mapped_column(Float)
    aircraft_total_cycles: Mapped[float | None] = mapped_column(Float)

    # Coded context (letters; decode with the FAA file layout)
    stage_of_operation_code: Mapped[str | None] = mapped_column(String(4))
    how_discovered_code: Mapped[str | None] = mapped_column(String(4))
    nature_of_condition_a: Mapped[str | None] = mapped_column(String(4))
    precautionary_procedure_a: Mapped[str | None] = mapped_column(String(4))
    submitter_type_code: Mapped[str | None] = mapped_column(String(4))

    # Keys for lookups, never model inputs
    aircraft_serial_number: Mapped[str | None] = mapped_column(String)
    registry_n_number: Mapped[str | None] = mapped_column(String)
    part_number: Mapped[str | None] = mapped_column(String)

    # Part context
    part_name: Mapped[str | None] = mapped_column(String)
    part_condition: Mapped[str | None] = mapped_column(String)
    part_location: Mapped[str | None] = mapped_column(String)

    # Lineage: where the row came from and when it was loaded
    source_file: Mapped[str] = mapped_column(String)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_sdr_make_model", "aircraft_make", "aircraft_model"),
        Index("ix_sdr_tail_jasc", "registry_n_number", "jasc_code"),
    )