"""Caltrans PeMS adapter, documented stub."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from born_field.adapters.base import Coverage
from born_field.types import CrashObservation, FlowObservation, GridCell, TimeWindow

_NOT_IMPLEMENTED = (
    "PeMSAdapter is a documented stub. Implementing it requires an "
    "errors-in-variables correction for imputed detector volume; without one the "
    "recovered flow exponent is biased below 1 (README, failure condition F5)."
)


class PeMSAdapter:
    """Caltrans Performance Measurement System. Not implemented."""

    name = "pems"

    def __init__(self, station_ids: Sequence[str] | None = None) -> None:
        """Record intended configuration; construction does not connect."""
        self.station_ids = tuple(station_ids or ())

    @property
    def version(self) -> str:
        """Source vintage."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def coverage(self) -> Coverage:
        """Declared support. Must reflect state highways only."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def cells(self) -> Sequence[GridCell]:
        """Grid with per-cell road mileage."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def flow(self, window: TimeWindow) -> Iterable[FlowObservation]:
        """Detector volume. Must set `imputed` truthfully per observation."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def crashes(self, window: TimeWindow) -> Iterable[CrashObservation]:
        """PeMS carries no collision records; pair with SWITRSAdapter."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
