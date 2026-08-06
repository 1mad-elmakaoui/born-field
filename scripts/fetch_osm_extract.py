"""Fetch a real OSM road network for the study bbox and vendor it as a fixture."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString

from born_field.config import DEFAULT_BBOX
from born_field.types import STORAGE_CRS, RoadClass

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
FIXTURE_PATH = Path("data/fixtures/study_area_roads.parquet")

# OSM has dozens of highway values; the model needs a few populated strata.
# Link roads fold into their parent class, being too sparse on their own.
_OSM_TO_CLASS: dict[str, RoadClass] = {
    "motorway": RoadClass.MOTORWAY,
    "motorway_link": RoadClass.MOTORWAY,
    "trunk": RoadClass.TRUNK,
    "trunk_link": RoadClass.TRUNK,
    "primary": RoadClass.PRIMARY,
    "primary_link": RoadClass.PRIMARY,
    "secondary": RoadClass.SECONDARY,
    "secondary_link": RoadClass.SECONDARY,
    "tertiary": RoadClass.SECONDARY,
    "tertiary_link": RoadClass.SECONDARY,
    "residential": RoadClass.RESIDENTIAL,
    "living_street": RoadClass.RESIDENTIAL,
    "unclassified": RoadClass.RESIDENTIAL,
}


def build_query(bbox: tuple[float, float, float, float]) -> str:
    """Overpass QL for drivable ways within ``bbox``."""
    lon_min, lat_min, lon_max, lat_max = bbox
    classes = "|".join(sorted(_OSM_TO_CLASS))
    return (
        "[out:json][timeout:180];"
        f'way["highway"~"^({classes})$"]'
        f"({lat_min},{lon_min},{lat_max},{lon_max});"
        "out geom;"
    )


def fetch(bbox: tuple[float, float, float, float], endpoint: str) -> dict[str, object]:
    """Execute the Overpass query."""
    payload = urllib.parse.urlencode({"data": build_query(bbox)}).encode()
    request = urllib.request.Request(
        endpoint, data=payload, headers={"User-Agent": "born-field/0.1 (fixture builder)"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result: dict[str, object] = json.load(response)
    return result


def to_geodataframe(payload: dict[str, object]) -> gpd.GeoDataFrame:
    """Convert an Overpass response to the fixture schema."""
    geometries: list[LineString] = []
    classes: list[str] = []

    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        msg = "unexpected Overpass payload: 'elements' is not a list"
        raise TypeError(msg)

    for element in elements:
        geometry = element.get("geometry")
        highway = element.get("tags", {}).get("highway")
        if not geometry or len(geometry) < 2 or highway not in _OSM_TO_CLASS:
            continue
        geometries.append(LineString([(p["lon"], p["lat"]) for p in geometry]))
        classes.append(_OSM_TO_CLASS[highway].value)

    if not geometries:
        msg = "Overpass returned no usable ways for this bbox"
        raise ValueError(msg)

    return gpd.GeoDataFrame({"road_class": classes}, geometry=geometries, crs=STORAGE_CRS)


def main(argv: list[str] | None = None) -> int:
    """Fetch, convert, and write the fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=OVERPASS_ENDPOINT)
    parser.add_argument("--output", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args(argv)

    try:
        payload = fetch(DEFAULT_BBOX, args.endpoint)
    except OSError as exc:
        print(f"Overpass request failed: {exc}", file=sys.stderr)
        print(
            "The generated stand-in network in data/fixtures/ remains in use; "
            "see src/born_field/field/network.py.",
            file=sys.stderr,
        )
        return 1

    frame = to_geodataframe(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output)
    print(f"wrote {len(frame)} ways to {args.output}")
    print(frame["road_class"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
