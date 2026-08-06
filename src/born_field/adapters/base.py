"""The data-source extension point."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import pandas as pd

from born_field.types import CrashObservation, FlowObservation, GridCell, TimeWindow


class Coverage:
    """Declared spatial and temporal support of a source."""

    __slots__ = ("bbox", "window")

    def __init__(self, bbox: tuple[float, float, float, float], window: TimeWindow) -> None:
        """Store the declared support."""
        self.bbox = bbox
        self.window = window

    def contains_point(self, lon: float, lat: float) -> bool:
        """Whether a coordinate falls inside declared spatial support."""
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Supplies the three inputs the pipeline needs, and its own support."""

    @property
    def name(self) -> str:
        """Stable identifier recorded with every fitted model."""
        ...

    @property
    def version(self) -> str:
        """Source version or vintage; part of run provenance."""
        ...

    def coverage(self) -> Coverage:
        """Declared spatial and temporal support."""
        ...

    def cells(self) -> Sequence[GridCell]:
        """The analysis grid, including per-cell road mileage."""
        ...

    def flow(self, window: TimeWindow) -> Iterable[FlowObservation]:
        """Vehicle throughput per cell for each sub-window in ``window``."""
        ...

    def crashes(self, window: TimeWindow) -> Iterable[CrashObservation]:
        """Collision records occurring within ``window``."""
        ...


@runtime_checkable
class VectorisedSource(Protocol):
    """Optional fast path for sources that can emit a whole panel at once."""

    def frame(self) -> pd.DataFrame:
        """Full cell-window panel."""
        ...
