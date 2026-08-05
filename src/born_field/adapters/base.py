"""The data-source extension point.

A buyer forking this repo replaces exactly one thing: the adapter. Everything
downstream -- intensity field, hazard model, validation, API -- is written
against :class:`DataSourceAdapter` and nothing else.

This is a ``Protocol`` rather than an abstract base class deliberately. A buyer
with an existing internal telematics client should be able to satisfy the
contract structurally, without importing from and inheriting our package.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import pandas as pd

from born_field.types import CrashObservation, FlowObservation, GridCell, TimeWindow


class Coverage:
    """Declared spatial and temporal support of a source.

    Every adapter must state where and when it has data. This is what the
    refusal path is checked against: a request outside declared coverage is
    refused rather than silently extrapolated. Sources that cannot describe
    their own support cannot be safely served from.
    """

    __slots__ = ("bbox", "window")

    def __init__(self, bbox: tuple[float, float, float, float], window: TimeWindow) -> None:
        """Store the declared support.

        Args:
            bbox: ``(lon_min, lat_min, lon_max, lat_max)`` in EPSG:4326.
            window: Half-open time interval over which the source has data.
        """
        self.bbox = bbox
        self.window = window

    def contains_point(self, lon: float, lat: float) -> bool:
        """Whether a coordinate falls inside declared spatial support."""
        lon_min, lat_min, lon_max, lat_max = self.bbox
        return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Supplies the three inputs the pipeline needs, and its own support.

    The three methods are separate rather than one ``load()`` because the join
    key differs: cells are static geometry, flow is dense per cell-window, and
    crashes are sparse point events that must be snapped to cells. Conflating
    them is how exposure denominators get silently misaligned with numerators.
    """

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
        """The analysis grid, including per-cell road mileage.

        Road mileage is required, not optional: it is half the exposure
        denominator. An adapter that cannot supply it cannot support a
        per-vehicle-mile model.
        """
        ...

    def flow(self, window: TimeWindow) -> Iterable[FlowObservation]:
        """Vehicle throughput per cell for each sub-window in ``window``."""
        ...

    def crashes(self, window: TimeWindow) -> Iterable[CrashObservation]:
        """Collision records occurring within ``window``."""
        ...


@runtime_checkable
class VectorisedSource(Protocol):
    """Optional fast path for sources that can emit a whole panel at once.

    :class:`DataSourceAdapter` is deliberately row-wise and lazy: it is the
    contract a buyer implements, and an ``Iterable`` streams without bounding
    memory. But the recovery experiment fits hundreds of models, and
    constructing a million validated Pydantic rows per fit would dominate its
    runtime for no analytical benefit.

    Sources that can produce a columnar panel advertise it here. Callers check
    with ``isinstance(source, VectorisedSource)`` and fall back to the row-wise
    path otherwise, so implementing this is genuinely optional.
    """

    def frame(self) -> pd.DataFrame:
        """Full cell-window panel.

        Required columns: ``cell_id``, ``window_start``, ``crash_count``,
        ``exposure_vehicle_miles``, ``flow_veh_per_hour``, ``road_class``,
        ``road_miles``, ``is_wet``, ``is_night``.
        """
        ...
