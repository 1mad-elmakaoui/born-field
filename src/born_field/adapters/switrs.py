"""SWITRS / TIMS collision adapter, documented stub."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from born_field.adapters.base import Coverage
from born_field.types import CrashObservation, FlowObservation, GridCell, TimeWindow

_NOT_IMPLEMENTED = (
    "SWITRSAdapter is a documented stub. Implementing it requires handling "
    "spatially varying reporting bias, severity-dependent completeness, and "
    "variable geocoding precision (MODEL_CARD.md section 3)."
)


class SWITRSAdapter:
    """California SWITRS collision records via TIMS. Not implemented."""

    name = "switrs"

    def __init__(self, county: str | None = None) -> None:
        """Record intended configuration; construction does not connect."""
        self.county = county

    @property
    def version(self) -> str:
        """Source vintage."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def coverage(self) -> Coverage:
        """Declared support."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def cells(self) -> Sequence[GridCell]:
        """SWITRS carries no network geometry; pair with an OSM-derived grid."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def flow(self, window: TimeWindow) -> Iterable[FlowObservation]:
        """SWITRS carries no flow; pair with PeMSAdapter."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def crashes(self, window: TimeWindow) -> Iterable[CrashObservation]:
        """Collision records. Must set `reported` and carry severity."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
