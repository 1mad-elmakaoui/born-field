"""Shared fixtures."""

from __future__ import annotations

import pytest

from born_field.adapters import SyntheticAdapter
from born_field.config import GridConfig
from born_field.models import HazardPanel

# Coarse cells over the whole study bbox: a sub-region of this network contains
# only residential roads, which would silently drop four of the five classes.
TEST_GRID = GridConfig(bbox=(-122.35, 37.75, -122.15, 37.90), cell_size_m=2000.0)


@pytest.fixture(scope="session")
def adapter() -> SyntheticAdapter:
    """A synthetic source with enough days to produce a scoreable event count."""
    return SyntheticAdapter(grid=TEST_GRID, days=45)


@pytest.fixture(scope="session")
def panel(adapter: SyntheticAdapter) -> HazardPanel:
    """The full validated panel."""
    return HazardPanel(adapter.frame())
