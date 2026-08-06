"""Regenerate the vendored stand-in road network."""

from __future__ import annotations

import argparse
from pathlib import Path

from born_field.config import DEFAULT_BBOX
from born_field.field.network import generate_network

FIXTURE_PATH = Path("data/fixtures/study_area_roads.parquet")


def main(argv: list[str] | None = None) -> int:
    """Generate and write the network fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args(argv)

    frame = generate_network(DEFAULT_BBOX, seed=args.seed).to_crs("EPSG:4326")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output)

    print(f"wrote {len(frame)} ways to {args.output}")
    print(frame["road_class"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
