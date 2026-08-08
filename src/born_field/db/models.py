"""PostGIS schema."""

from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for the schema."""


class Cell(Base):
    """One analysis cell, with its geometry and road mileage."""

    __tablename__ = "cells"

    cell_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    grid_row: Mapped[int] = mapped_column(Integer, nullable=False)
    grid_col: Mapped[int] = mapped_column(Integer, nullable=False)
    road_miles: Mapped[float] = mapped_column(Float, nullable=False)
    dominant_class: Mapped[str] = mapped_column(String(16), nullable=False)
    geometry: Mapped[object] = mapped_column(
        Geometry(geometry_type="POLYGON", srid=4326), nullable=False
    )

    segments: Mapped[list[Segment]] = relationship(back_populates="cell")

    __table_args__ = (
        # GiST over the polygon. The point query rides this index, which is the
        # only reason a spatial database is here.
        Index("ix_cells_geometry", "geometry", postgresql_using="gist"),
        Index("ix_cells_grid", "grid_row", "grid_col"),
    )


class Segment(Base):
    """One ``(cell, road_class)`` stratum: the analysis unit."""

    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(ForeignKey("cells.cell_id"), nullable=False)
    road_class: Mapped[str] = mapped_column(String(16), nullable=False)
    road_miles: Mapped[float] = mapped_column(Float, nullable=False)

    cell: Mapped[Cell] = relationship(back_populates="segments")

    __table_args__ = (UniqueConstraint("cell_id", "road_class", name="uq_segment"),)


class ScoreRecord(Base):
    """An audit trail of what was scored, with what model, and what came back."""

    __tablename__ = "score_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    query_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cell_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    point: Mapped[object] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326), nullable=False
    )

    refused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    refusal_reason: Mapped[str | None] = mapped_column(String(48), nullable=True)

    expected_crashes: Mapped[float | None] = mapped_column(Float, nullable=True)
    rate_per_100m_vmt: Mapped[float | None] = mapped_column(Float, nullable=True)
    rate_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    rate_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    exposure_vehicle_miles: Mapped[float | None] = mapped_column(Float, nullable=True)

    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        Index("ix_score_records_scored_at", "scored_at"),
        Index("ix_score_records_cell", "cell_id"),
        # Partial index. The operational question is usually what we are refusing
        # and where, which earns its own index.
        Index(
            "ix_score_records_refusals",
            "refusal_reason",
            postgresql_where=text("refused = true"),
        ),
    )
