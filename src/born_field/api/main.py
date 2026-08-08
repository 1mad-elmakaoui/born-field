"""Versioned REST API."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from born_field import __version__
from born_field.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    CalibrationBinOut,
    CalibrationOut,
    EstimateOut,
    HealthOut,
    IntervalOut,
    ModelInfoOut,
    RefusalOut,
    ScoreRequest,
    ScoreResponse,
)
from born_field.api.service import RiskService, build_demo_service
from born_field.config import Settings
from born_field.logging import configure_logging, get_logger
from born_field.types import HazardEstimate, Refusal

log = get_logger(__name__)

REQUESTS = Counter(
    "bornfield_requests_total", "API requests by endpoint and outcome", ["endpoint", "outcome"]
)
LATENCY = Histogram("bornfield_request_seconds", "Request latency by endpoint", ["endpoint"])
REFUSALS = Counter(
    "bornfield_refusals_total",
    "Refusals by reason. A rising rate here means the fitted support no longer "
    "matches the traffic being sent, which is an operational signal, not noise.",
    ["reason"],
)

_state: dict[str, RiskService] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Fit the demo model once at startup rather than per request."""
    settings = Settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    _state["service"] = build_demo_service()
    log.info("api_ready", version=__version__)
    yield
    _state.clear()


app = FastAPI(
    title="Born Field",
    version=__version__,
    summary="Calibrated geospatial collision-risk scoring.",
    description=(
        "Risk per vehicle-mile, with uncertainty intervals on every estimate and "
        "an explicit refusal path for queries outside the fitted support.\n\n"
        "**Intended use:** infrastructure prioritisation and network screening. "
        "**Not** for individual insurance pricing or enforcement targeting -- see "
        "the model card."
    ),
    lifespan=lifespan,
)


def get_service() -> RiskService:
    """Dependency: the fitted service, or 503 before startup completes."""
    service = _state.get("service")
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model not loaded"
        )
    return service


# Auth and rate limiting: deliberate stubs.

_RATE_LIMIT_REQUESTS = 120
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_recent: dict[str, deque[float]] = defaultdict(deque)


async def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> str:
    """Auth **stub**."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def enforce_rate_limit(
    caller: Annotated[str, Depends(require_api_key)],
) -> str:
    """Rate-limit **stub**: a fixed window held in process memory."""
    now = time.monotonic()
    seen = _recent[caller]
    while seen and now - seen[0] > _RATE_LIMIT_WINDOW_SECONDS:
        seen.popleft()
    if len(seen) >= _RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit of {_RATE_LIMIT_REQUESTS} requests per minute exceeded",
            headers={"Retry-After": str(int(_RATE_LIMIT_WINDOW_SECONDS))},
        )
    seen.append(now)
    return caller


Caller = Annotated[str, Depends(enforce_rate_limit)]
Service = Annotated[RiskService, Depends(get_service)]


@app.middleware("http")
async def observe(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Record latency and outcome for every request."""
    endpoint = request.url.path
    start = time.perf_counter()
    response = await call_next(request)
    LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    REQUESTS.labels(endpoint=endpoint, outcome=str(response.status_code)).inc()
    return response


# Conversion.


def _to_response(result: HazardEstimate | Refusal) -> ScoreResponse:
    if isinstance(result, Refusal):
        REFUSALS.labels(reason=result.reason.value).inc()
        return RefusalOut(reason=result.reason, detail=result.detail, cell_id=result.cell_id)
    return EstimateOut(
        cell_id=result.cell_id,
        window_start=result.window.start,
        window_end=result.window.end,
        expected_crashes=result.expected_crashes,
        rate_per_100m_vmt=result.rate_per_100m_vmt,
        rate_interval=IntervalOut(
            lower=result.rate_interval.lower,
            upper=result.rate_interval.upper,
            level=result.rate_interval.level,
        ),
        probability_at_least_one=result.probability_at_least_one,
        probability_interval=IntervalOut(
            lower=result.probability_interval.lower,
            upper=result.probability_interval.upper,
            level=result.probability_interval.level,
        ),
        exposure_vehicle_miles=result.exposure_vehicle_miles,
        model_name=result.model_name,
        model_version=result.model_version,
    )


# Endpoints.


@app.get("/health", response_model=HealthOut, tags=["ops"])
async def health() -> HealthOut:
    """Liveness and model-readiness. Unauthenticated, for probes."""
    loaded = "service" in _state
    return HealthOut(
        status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__
    )


@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
async def score(request: ScoreRequest, service: Service, caller: Caller) -> ScoreResponse:
    """Score one point in space and time."""
    log.info("score_request", caller=caller, lat=request.lat, lon=request.lon)
    result = service.score(
        lon=request.lon,
        lat=request.lat,
        timestamp=request.timestamp,
        flow_veh_per_hour=request.flow_veh_per_hour,
        is_wet=request.is_wet,
    )
    return _to_response(result)


@app.post("/v1/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
async def score_batch(
    request: BatchScoreRequest, service: Service, caller: Caller
) -> BatchScoreResponse:
    """Score many queries."""
    results = [
        _to_response(
            service.score(
                lon=q.lon,
                lat=q.lat,
                timestamp=q.timestamp,
                flow_veh_per_hour=q.flow_veh_per_hour,
                is_wet=q.is_wet,
            )
        )
        for q in request.queries
    ]
    refusals = sum(1 for r in results if isinstance(r, RefusalOut))
    log.info("batch_scored", caller=caller, n=len(results), refusals=refusals)
    return BatchScoreResponse(
        results=results, n_estimates=len(results) - refusals, n_refusals=refusals
    )


@app.get("/v1/calibration", response_model=CalibrationOut, tags=["reporting"])
async def calibration(service: Service, caller: Caller) -> CalibrationOut:
    """The reliability of the deployed model, on its forward-in-time holdout."""
    report = service.calibration
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no calibration report attached to the deployed model",
        )
    log.info("calibration_requested", caller=caller)
    return CalibrationOut(
        model_name=report.model_name,
        model_version=report.model_version,
        expected_calibration_error=report.expected_calibration_error,
        coverage_of_nominal_95=report.coverage_of_nominal_95,
        bins=[
            CalibrationBinOut(
                predicted_mean=b.predicted_mean,
                observed_mean=b.observed_mean,
                observed_interval_lower=b.observed_interval_lower,
                observed_interval_upper=b.observed_interval_upper,
                n_observations=b.n_observations,
            )
            for b in report.bins
        ],
    )


@app.get("/v1/model", response_model=ModelInfoOut, tags=["reporting"])
async def model_info(service: Service, caller: Caller) -> ModelInfoOut:
    """What is deployed, what it recovered, and what it will refuse."""
    log.info("model_info_requested", caller=caller)
    return ModelInfoOut(
        model_name=service.model.name,
        model_version=service.model.version,
        recovered_alpha=service.fitted_alpha,
        exposure_null_rejected=service.model.exposure_null_rejected,
        support=service.support_summary(),
    )
