"""Core domain types."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Length and area maths needs a metre-based projected CRS. A degree is not a
# distance, and measuring in EPSG:4326 quietly corrupts the exposure denominator.
STORAGE_CRS: Final[str] = "EPSG:4326"
"""Canonical storage/interchange CRS (lon/lat, WGS 84)."""

WORKING_CRS: Final[str] = "EPSG:3310"
"""California Albers. Equal-area, metre units -- used for all length/area maths."""

METRES_PER_MILE: Final[float] = 1609.344


class Frozen(BaseModel):
    """Base for immutable, strictly-validated domain models."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class RoadClass(StrEnum):
    """Functional road classification."""

    MOTORWAY = "motorway"
    TRUNK = "trunk"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    RESIDENTIAL = "residential"


Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
NonNegative = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class TimeWindow(Frozen):
    """A half-open time interval ``[start, end)`` in UTC."""

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _check_ordering(self) -> TimeWindow:
        if self.end <= self.start:
            msg = f"window end {self.end!r} must be strictly after start {self.start!r}"
            raise ValueError(msg)
        if self.start.tzinfo is None or self.end.tzinfo is None:
            msg = "TimeWindow bounds must be timezone-aware (UTC)"
            raise ValueError(msg)
        return self

    @property
    def duration(self) -> timedelta:
        """Length of the window."""
        return self.end - self.start

    @property
    def hours(self) -> float:
        """Length of the window in hours."""
        return self.duration.total_seconds() / 3600.0


class GridCell(Frozen):
    """One spatial unit of analysis."""

    cell_id: str
    centroid_lon: Longitude
    centroid_lat: Latitude
    road_miles: NonNegative
    dominant_class: RoadClass


class FlowObservation(Frozen):
    """Measured or imputed vehicle throughput for one cell-window."""

    cell_id: str
    window_start: datetime
    flow_veh_per_hour: NonNegative
    imputed: bool = False


class CrashObservation(Frozen):
    """A single recorded collision."""

    crash_id: str
    lon: Longitude
    lat: Latitude
    occurred_at: datetime
    severity: Literal["fatal", "injury", "pdo"] = "injury"
    reported: bool = True


class ExposureRecord(Frozen):
    """Vehicle-miles travelled in one cell-window: the Poisson offset."""

    cell_id: str
    window_start: datetime
    vehicle_miles: NonNegative


class Interval(Frozen):
    """A two-sided uncertainty interval."""

    lower: float
    upper: float
    level: Probability = 0.95

    @model_validator(mode="after")
    def _check_ordering(self) -> Interval:
        if self.upper < self.lower:
            msg = f"interval upper {self.upper!r} is below lower {self.lower!r}"
            raise ValueError(msg)
        return self


class HazardEstimate(Frozen):
    """A calibrated risk estimate for one cell-window."""

    cell_id: str
    window: TimeWindow
    expected_crashes: NonNegative
    rate_per_100m_vmt: NonNegative
    rate_interval: Interval
    probability_at_least_one: Probability
    probability_interval: Interval
    exposure_vehicle_miles: NonNegative
    model_name: str
    model_version: str


class RefusalReason(StrEnum):
    """Why the service declined to score."""

    OUTSIDE_FITTED_REGION = "outside_fitted_region"
    OUTSIDE_FITTED_TIME_RANGE = "outside_fitted_time_range"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    EXPOSURE_UNAVAILABLE = "exposure_unavailable"
    EXTRAPOLATION_BEYOND_SUPPORT = "extrapolation_beyond_support"


class Refusal(Frozen):
    """A declined scoring request, with enough detail to act on."""

    reason: RefusalReason
    detail: str
    cell_id: str | None = None


ScoringResult = HazardEstimate | Refusal
"""What every scoring path returns. Callers must handle both arms."""


class CalibrationBin(Frozen):
    """One bin of a reliability diagram."""

    predicted_mean: NonNegative
    observed_mean: NonNegative
    observed_interval_lower: NonNegative
    observed_interval_upper: NonNegative
    n_observations: int


class CalibrationReport(Frozen):
    """Served by the API, not just plotted in a notebook."""

    model_name: str
    model_version: str
    bins: Sequence[CalibrationBin]
    expected_calibration_error: NonNegative
    coverage_of_nominal_95: Probability


def vehicle_miles_travelled(
    flow_veh_per_hour: float,
    road_miles: float,
    hours: float,
) -> float:
    """Compute exposure (VMT) for one cell-window."""
    if flow_veh_per_hour < 0 or road_miles < 0 or hours < 0:
        msg = (
            "VMT inputs must be non-negative, got "
            f"flow={flow_veh_per_hour!r}, road_miles={road_miles!r}, hours={hours!r}"
        )
        raise ValueError(msg)
    return flow_veh_per_hour * hours * road_miles


def vehicle_miles_travelled_array(
    flow_veh_per_hour: npt.NDArray[np.float64],
    road_miles: npt.NDArray[np.float64],
    hours: float,
) -> npt.NDArray[np.float64]:
    """Vectorised :func:`vehicle_miles_travelled`, for whole-panel computation."""
    if np.any(flow_veh_per_hour < 0) or np.any(road_miles < 0) or hours < 0:
        msg = "VMT inputs must be non-negative"
        raise ValueError(msg)
    result: npt.NDArray[np.float64] = flow_veh_per_hour * hours * road_miles
    return result
