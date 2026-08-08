"""Reads and writes against the PostGIS schema."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import geopandas as gpd
from shapely import wkb
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from born_field.db.models import Cell, ScoreRecord, Segment
from born_field.field.grid import build_segments
from born_field.types import STORAGE_CRS, HazardEstimate, Refusal, ScoringResult


def upsert_cells(session: Session, cells: gpd.GeoDataFrame) -> int:
    """Write the grid, replacing any existing definition of the same cells."""
    frame = cells.to_crs(STORAGE_CRS)
    rows = [
        {
            "cell_id": str(row.cell_id),
            "grid_row": int(row.row),
            "grid_col": int(row.col),
            "road_miles": float(cast("float", row.road_miles)),
            "dominant_class": str(row.dominant_class),
            "geometry": row.geometry.wkt,
        }
        for row in frame.itertuples()
    ]
    statement = insert(Cell).values(rows)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[Cell.cell_id],
            set_={
                "road_miles": statement.excluded.road_miles,
                "dominant_class": statement.excluded.dominant_class,
                "geometry": statement.excluded.geometry,
            },
        )
    )

    segments = build_segments(cells)
    segment_rows = [
        {
            "cell_id": str(row.cell_id),
            "road_class": str(row.road_class),
            "road_miles": float(cast("float", row.road_miles)),
        }
        for row in segments.itertuples()
    ]
    segment_statement = insert(Segment).values(segment_rows)
    session.execute(
        segment_statement.on_conflict_do_update(
            index_elements=[Segment.cell_id, Segment.road_class],
            set_={"road_miles": segment_statement.excluded.road_miles},
        )
    )
    return len(rows)


def load_cells(session: Session) -> gpd.GeoDataFrame:
    """Read the grid back as a GeoDataFrame."""
    records = session.execute(select(Cell)).scalars().all()
    return gpd.GeoDataFrame(
        {
            "cell_id": [r.cell_id for r in records],
            "row": [r.grid_row for r in records],
            "col": [r.grid_col for r in records],
            "road_miles": [r.road_miles for r in records],
            "dominant_class": [r.dominant_class for r in records],
        },
        # GeoAlchemy2 returns a WKBElement; the ORM column is typed `object`
        # because geoalchemy2 ships no usable annotation for it.
        geometry=[wkb.loads(bytes(cast("Any", r.geometry).data)) for r in records],
        crs=STORAGE_CRS,
    )


def record_score(
    session: Session,
    result: ScoringResult,
    lon: float,
    lat: float,
    query_timestamp: datetime,
    model_name: str,
    model_version: str,
) -> None:
    """Append one audit row, for an estimate or a refusal alike."""
    point = f"POINT({lon} {lat})"
    if isinstance(result, Refusal):
        session.add(
            ScoreRecord(
                scored_at=datetime.now(UTC),
                query_timestamp=query_timestamp,
                cell_id=result.cell_id,
                point=point,
                refused=True,
                refusal_reason=result.reason.value,
                model_name=model_name,
                model_version=model_version,
            )
        )
        return

    estimate: HazardEstimate = result
    session.add(
        ScoreRecord(
            scored_at=datetime.now(UTC),
            query_timestamp=query_timestamp,
            cell_id=estimate.cell_id,
            point=point,
            refused=False,
            expected_crashes=estimate.expected_crashes,
            rate_per_100m_vmt=estimate.rate_per_100m_vmt,
            rate_lower=estimate.rate_interval.lower,
            rate_upper=estimate.rate_interval.upper,
            exposure_vehicle_miles=estimate.exposure_vehicle_miles,
            model_name=model_name,
            model_version=model_version,
        )
    )
