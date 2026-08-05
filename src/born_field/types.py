"""Core domain types.

These models are the stable contract between the four swappable layers
(source adapter -> intensity field -> hazard model -> risk API). Every type here
is immutable and validated; nothing in this package passes free-floating dicts
across a layer boundary.

Units are stated explicitly in every field name. Traffic-safety work mixes
metric and imperial constantly (metres for geometry, vehicle-*miles* for
exposure because ``crashes per 100M VMT`` is the sector-standard denominator),
and unit confusion is the most common silent error in this domain.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

# A metre-based projected CRS is required for any length or area computation.
# Geographic degrees are not a unit of distance; computing segment length in
# EPSG:4326 is the classic way to silently corrupt an exposure denominator.
STORAGE_CRS: Final[str] = "EPSG:4326"
"""Canonical storage/interchange CRS (lon/lat, WGS 84)."""

WORKING_CRS: Final[str] = "EPSG:3310"
"""California Albers. Equal-area, metre units -- used for all length/area maths."""

METRES_PER_MILE: Final[float] = 1609.344


class Frozen(BaseModel):
    """Base for immutable, strictly-validated domain models."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class RoadClass(StrEnum):
    """Functional road classification.

    Deliberately coarse and mapped from OSM ``highway`` tags rather than adopting
    them wholesale: a hazard model needs a handful of well-populated strata, not
    thirty sparsely-observed tag values.
    """

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
    """A half-open time interval ``[start, end)`` in UTC.

    Half-open by construction so that consecutive windows tile the timeline
    without double-counting exposure at the boundary.
    """

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
    """One spatial unit of analysis.

    ``road_miles`` is a first-class field rather than something derived later:
    it is half of the exposure denominator. Two cells with identical traffic
    flow but different road mileage do not carry the same crash opportunity, and
    a model that ignores that will attribute the difference to location.
    """

    cell_id: str
    centroid_lon: Longitude
    centroid_lat: Latitude
    road_miles: NonNegative
    dominant_class: RoadClass


class FlowObservation(Frozen):
    """Measured or imputed vehicle throughput for one cell-window.

    ``imputed`` is carried through the whole pipeline rather than dropped at
    ingest. Real detector networks (PeMS in particular) impute a large fraction
    of samples, and imputation is a measurement-error source that biases the
    fitted flow exponent toward 1. Downstream code must be able to see it.
    """

    cell_id: str
    window_start: datetime
    flow_veh_per_hour: NonNegative
    imputed: bool = False


class CrashObservation(Frozen):
    """A single recorded collision.

    ``reported`` distinguishes occurrence from observation. Crash records are
    a *reporting* process layered on the true event process, and the reporting
    rate varies spatially. Keeping the distinction in the type system is what
    lets the underreporting experiment in Stage 3 exist at all.
    """

    crash_id: str
    lon: Longitude
    lat: Latitude
    occurred_at: datetime
    severity: Literal["fatal", "injury", "pdo"] = "injury"
    reported: bool = True


class ExposureRecord(Frozen):
    """Vehicle-miles travelled in one cell-window: the Poisson offset.

    This is the single most important quantity in the project. See
    :func:`vehicle_miles_travelled` and the README section "The exposure
    confound".
    """

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
    """A calibrated risk estimate for one cell-window.

    Both a rate and a probability are returned because they answer different
    commercial questions. ``rate_per_100m_vmt`` is the sector-standard
    comparable figure; ``probability_at_least_one`` is what an operational
    system thresholds on. Neither is reported without an interval -- an
    uncertainty-free risk score cannot be priced against, and pricing is the
    product.
    """

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
    """Why the service declined to score.

    A refusal is a first-class return value, not an exception: "we do not know"
    is a legitimate and commercially important answer, and callers must be able
    to branch on *which* kind of not-knowing they hit.
    """

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
    """Served by the API, not just plotted in a notebook.

    Exposing this is the difference between "here is a score" and "here is a
    score you can underwrite against". It is the endpoint a buyer's actuarial
    team will look at first.
    """

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
    """Compute exposure (VMT) for one cell-window.

    This is the Poisson offset. Crash *counts* scale with exposure almost
    tautologically -- more vehicles travelling more miles produce more
    collisions -- so a model fitted to counts without this offset rediscovers
    the traffic volume map and calls it a risk map.

    Note that flow enters here **and** appears again as a covariate in the
    hazard model. That is intentional and is the core identification strategy:
    with ``offset = log(VMT)``, the free coefficient on ``log(flow)`` estimates
    ``alpha - 1``, so the null hypothesis "risk is purely exposure-driven" is
    exactly the hypothesis that this coefficient is zero.

    Args:
        flow_veh_per_hour: Mean vehicle throughput over the window.
        road_miles: Centre-line road mileage within the cell.
        hours: Window duration in hours.

    Returns:
        Vehicle-miles travelled in the cell-window.

    Raises:
        ValueError: If any argument is negative.
    """
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
    """Vectorised :func:`vehicle_miles_travelled`, for whole-panel computation.

    Kept as an explicit sibling rather than inlining the product at each call
    site. The exposure denominator is the one quantity in this project that must
    not drift between code paths, and ``test_scalar_and_array_paths_agree``
    pins the two implementations together.
    """
    if np.any(flow_veh_per_hour < 0) or np.any(road_miles < 0) or hours < 0:
        msg = "VMT inputs must be non-negative"
        raise ValueError(msg)
    result: npt.NDArray[np.float64] = flow_veh_per_hour * hours * road_miles
    return result
