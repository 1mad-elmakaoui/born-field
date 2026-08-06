"""Grid construction: turning a road network into analysis cells."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from born_field.types import METRES_PER_MILE, STORAGE_CRS, WORKING_CRS, GridCell, RoadClass


def build_cells(
    network: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    cell_size_m: float,
) -> gpd.GeoDataFrame:
    """Tile ``bbox`` with square cells and attribute road mileage to each."""
    bounds = (
        gpd.GeoSeries.from_xy([bbox[0], bbox[2]], [bbox[1], bbox[3]], crs=STORAGE_CRS)
        .to_crs(WORKING_CRS)
        .total_bounds
    )
    min_x, min_y, max_x, max_y = bounds

    xs = np.arange(min_x, max_x, cell_size_m)
    ys = np.arange(min_y, max_y, cell_size_m)

    # Ids encode grid position, so neighbours are recoverable by arithmetic and
    # an id survives the study area being extended.
    records = []
    geometries = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            records.append({"cell_id": f"r{j:04d}c{i:04d}", "col": i, "row": j})
            geometries.append(
                gpd.GeoSeries.from_wkt(
                    [
                        f"POLYGON (({x} {y}, {x + cell_size_m} {y}, "
                        f"{x + cell_size_m} {y + cell_size_m}, {x} {y + cell_size_m}, {x} {y}))"
                    ]
                ).iloc[0]
            )

    cells = gpd.GeoDataFrame(records, geometry=geometries, crs=WORKING_CRS)

    # Clip road geometry to cell boundaries so a segment crossing four cells
    # contributes its actual length to each, not its whole length to all four.
    clipped = gpd.overlay(
        network[["road_class", "geometry"]],
        cells[["cell_id", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    clipped["miles"] = clipped.geometry.length / METRES_PER_MILE

    by_class = clipped.groupby(["cell_id", "road_class"])["miles"].sum().unstack(fill_value=0.0)
    for road_class in RoadClass:
        if road_class.value not in by_class.columns:
            by_class[road_class.value] = 0.0
    by_class = by_class[[c.value for c in RoadClass]]

    totals = by_class.sum(axis=1)
    populated = totals[totals > 0].index

    cells = cells.set_index("cell_id").loc[populated]
    cells = cells.join(by_class)
    cells["road_miles"] = totals.loc[populated]

    # Dominant class by mileage, not by count of segments: a cell with one
    # motorway and three short residential stubs is a motorway cell.
    cells["dominant_class"] = by_class.loc[populated].idxmax(axis=1)

    centroids = cells.geometry.centroid.to_crs(STORAGE_CRS)
    cells["centroid_lon"] = centroids.x
    cells["centroid_lat"] = centroids.y

    return cells.reset_index()


def build_segments(cells: gpd.GeoDataFrame) -> pd.DataFrame:
    """Explode cells into ``(cell, road_class)`` strata: the analysis unit."""
    identifiers = ["cell_id", "row", "col", "centroid_lon", "centroid_lat"]
    per_class = [c.value for c in RoadClass]
    # Selected explicitly because the cell frame also carries a road_miles total,
    # which would collide with the melted per-class column.
    long = pd.DataFrame(cells[identifiers + per_class]).melt(
        id_vars=identifiers,
        value_vars=per_class,
        var_name="road_class",
        value_name="road_miles",
    )
    return long[long["road_miles"] > 0].reset_index(drop=True)


def cells_to_models(cells: gpd.GeoDataFrame) -> list[GridCell]:
    """Convert the cell frame to validated domain objects."""
    return [
        GridCell(
            cell_id=str(row.cell_id),
            centroid_lon=float(row.centroid_lon),
            centroid_lat=float(row.centroid_lat),
            road_miles=float(row.road_miles),
            dominant_class=RoadClass(row.dominant_class),
        )
        for row in cells.itertuples()
    ]


def aggregate_cells(cells: gpd.GeoDataFrame, factor: int) -> pd.DataFrame:
    """Map each cell to a coarser cell ``factor`` times larger on a side."""
    if factor < 1:
        msg = f"aggregation factor must be >= 1, got {factor}"
        raise ValueError(msg)
    return pd.DataFrame(
        {
            "cell_id": cells["cell_id"],
            "coarse_cell_id": [
                f"r{row.row // factor:04d}c{row.col // factor:04d}" for row in cells.itertuples()
            ],
        }
    )
