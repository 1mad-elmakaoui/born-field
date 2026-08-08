"""Request and response models for the versioned REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from born_field.types import Latitude, Longitude, RefusalReason


class ScoreRequest(BaseModel):
    """A single point-in-time risk query."""

    model_config = ConfigDict(extra="forbid")

    lat: Latitude = Field(description="WGS84 latitude.")
    lon: Longitude = Field(description="WGS84 longitude.")
    timestamp: datetime = Field(description="Timezone-aware instant to score.")
    flow_veh_per_hour: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Observed vehicle throughput. Omit to use the cell's fitted typical "
            "flow; the exposure figure in the response states which was used."
        ),
    )
    is_wet: bool = Field(default=False, description="Wet road surface at the query time.")


class IntervalOut(BaseModel):
    """A two-sided uncertainty interval."""

    lower: float
    upper: float
    level: float


class EstimateOut(BaseModel):
    """A calibrated risk estimate. Never returned without intervals."""

    outcome: Literal["estimate"] = "estimate"
    cell_id: str
    window_start: datetime
    window_end: datetime
    expected_crashes: float
    rate_per_100m_vmt: float
    rate_interval: IntervalOut
    probability_at_least_one: float
    probability_interval: IntervalOut
    exposure_vehicle_miles: float
    model_name: str
    model_version: str


class RefusalOut(BaseModel):
    """A declined query. HTTP 200: a refusal is a valid answer, not an error."""

    outcome: Literal["refusal"] = "refusal"
    reason: RefusalReason
    detail: str
    cell_id: str | None = None


ScoreResponse = EstimateOut | RefusalOut


class BatchScoreRequest(BaseModel):
    """Several queries in one call."""

    model_config = ConfigDict(extra="forbid")

    queries: list[ScoreRequest] = Field(min_length=1, max_length=1000)


class BatchScoreResponse(BaseModel):
    """Results in request order, each independently an estimate or a refusal."""

    results: list[ScoreResponse]
    n_estimates: int
    n_refusals: int


class CalibrationBinOut(BaseModel):
    """One bin of the reliability diagram."""

    predicted_mean: float
    observed_mean: float
    observed_interval_lower: float
    observed_interval_upper: float
    n_observations: int


class CalibrationOut(BaseModel):
    """The calibration report. The endpoint an actuarial team looks at first."""

    model_name: str
    model_version: str
    expected_calibration_error: float
    coverage_of_nominal_95: float
    bins: list[CalibrationBinOut]


class ModelInfoOut(BaseModel):
    """What is deployed, what it recovered, and what it will refuse to answer."""

    model_name: str
    model_version: str
    recovered_alpha: float
    exposure_null_rejected: bool
    support: dict[str, object]


class HealthOut(BaseModel):
    """Liveness and readiness."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str
