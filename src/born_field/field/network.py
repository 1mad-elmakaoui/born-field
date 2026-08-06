"""Road network: loading a vendored extract, or generating a stand-in."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from born_field.types import STORAGE_CRS, WORKING_CRS, RoadClass

# Approximate spacing between roads of each class, in metres. Ordered coarse to
# fine: motorways are corridors, residential streets are a mesh.
_CLASS_SPACING_M: dict[RoadClass, float] = {
    RoadClass.MOTORWAY: 9000.0,
    RoadClass.TRUNK: 6000.0,
    RoadClass.PRIMARY: 2600.0,
    RoadClass.SECONDARY: 1300.0,
    RoadClass.RESIDENTIAL: 380.0,
}

# Lateral wander per vertex, in metres. Without it every segment is axis-aligned
# and cell mileage collapses to a handful of values.
_CLASS_JITTER_M: dict[RoadClass, float] = {
    RoadClass.MOTORWAY: 420.0,
    RoadClass.TRUNK: 320.0,
    RoadClass.PRIMARY: 190.0,
    RoadClass.SECONDARY: 120.0,
    RoadClass.RESIDENTIAL: 40.0,
}

_VERTEX_SPACING_M = 300.0


def _jittered_line(
    start: tuple[float, float],
    end: tuple[float, float],
    jitter_m: float,
    rng: np.random.Generator,
) -> LineString:
    """A line from start to end with smooth lateral wander."""
    x0, y0 = start
    x1, y1 = end
    length = float(np.hypot(x1 - x0, y1 - y0))
    n_vertices = max(2, int(length / _VERTEX_SPACING_M))
    t = np.linspace(0.0, 1.0, n_vertices)

    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)

    # Unit normal to the segment, so wander is perpendicular rather than
    # stretching the line along its own axis.
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length

    # A random walk smoothed by a cumulative sum, tapered to zero at both ends
    # so segments still meet at their intended junctions.
    steps = rng.normal(0.0, 1.0, n_vertices)
    wander = np.cumsum(steps)
    wander -= np.linspace(wander[0], wander[-1], n_vertices)
    spread = float(np.abs(wander).max())
    if spread > 0:
        wander = wander / spread * jitter_m

    return LineString(np.column_stack([xs + nx * wander, ys + ny * wander]))


def _developed_mask(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> bool:
    """Whether residential density reaches this point."""
    min_x, min_y, max_x, max_y = bounds
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    half_x, half_y = (max_x - min_x) / 2, (max_y - min_y) / 2
    radial = np.hypot((x - cx) / half_x, (y - cy) / half_y)
    # Soft edge: probability of development falls off away from the centre.
    return bool(rng.random() < np.clip(1.35 - radial, 0.0, 1.0))


def generate_network(
    bbox: tuple[float, float, float, float],
    seed: int = 0,
) -> gpd.GeoDataFrame:
    """Generate a road network with realistic topology over ``bbox``."""
    rng = np.random.default_rng(seed)

    frame = gpd.GeoDataFrame(
        geometry=[
            LineString([(bbox[0], bbox[1]), (bbox[2], bbox[3])]),
        ],
        crs=STORAGE_CRS,
    ).to_crs(WORKING_CRS)
    min_x, min_y, max_x, max_y = frame.total_bounds

    geometries: list[LineString] = []
    classes: list[str] = []

    for road_class, spacing in _CLASS_SPACING_M.items():
        jitter = _CLASS_JITTER_M[road_class]

        # Vertical runs.
        for x in np.arange(min_x + spacing / 2, max_x, spacing):
            if road_class is RoadClass.RESIDENTIAL and not _developed_mask(
                x, (min_y + max_y) / 2, (min_x, min_y, max_x, max_y), rng
            ):
                continue
            geometries.append(_jittered_line((x, min_y), (x, max_y), jitter, rng))
            classes.append(road_class.value)

        # Horizontal runs.
        for y in np.arange(min_y + spacing / 2, max_y, spacing):
            if road_class is RoadClass.RESIDENTIAL and not _developed_mask(
                (min_x + max_x) / 2, y, (min_x, min_y, max_x, max_y), rng
            ):
                continue
            geometries.append(_jittered_line((min_x, y), (max_x, y), jitter, rng))
            classes.append(road_class.value)

    return gpd.GeoDataFrame(
        {"road_class": classes},
        geometry=geometries,
        crs=WORKING_CRS,
    )


FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "fixtures" / "study_area_roads.parquet"
)
"""Vendored network extract, used when present."""


def default_network(bbox: tuple[float, float, float, float], seed: int) -> gpd.GeoDataFrame:
    """Load the vendored extract, falling back to generation."""
    if FIXTURE_PATH.exists():
        return load_network(FIXTURE_PATH)
    return generate_network(bbox, seed=seed)


def load_network(path: Path) -> gpd.GeoDataFrame:
    """Load a vendored network extract and reproject to the working CRS."""
    frame = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    if "road_class" not in frame.columns:
        msg = f"network extract at {path} has no 'road_class' column"
        raise ValueError(msg)

    unknown = set(frame["road_class"]) - {c.value for c in RoadClass}
    if unknown:
        msg = f"network extract contains unmapped road classes: {sorted(unknown)}"
        raise ValueError(msg)

    return frame.to_crs(WORKING_CRS)
