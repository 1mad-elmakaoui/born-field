"""Deterministic tools."""

from __future__ import annotations

from born_field.agent.state import QueryState, ToolCall
from born_field.api.service import RiskService
from born_field.types import HazardEstimate, Refusal


def locate_cell(service: RiskService, state: QueryState) -> ToolCall:
    """Resolve a coordinate to a grid cell."""
    cell_id = service.locate(state.lon, state.lat)
    return ToolCall(
        tool="locate_cell",
        arguments={"lat": state.lat, "lon": state.lon},
        text_outputs={"cell_id": cell_id or "none"},
    )


def score_point(
    service: RiskService, state: QueryState
) -> tuple[ToolCall, HazardEstimate | Refusal]:
    """Score the query, returning both the audit record and the result."""
    result = service.score(
        lon=state.lon,
        lat=state.lat,
        timestamp=state.timestamp,
        flow_veh_per_hour=state.flow_veh_per_hour,
        is_wet=state.is_wet,
    )
    arguments = {
        "lat": state.lat,
        "lon": state.lon,
        "timestamp": state.timestamp.isoformat(),
        "is_wet": state.is_wet,
    }

    if isinstance(result, Refusal):
        return (
            ToolCall(
                tool="score_point",
                arguments=arguments,
                text_outputs={"outcome": "refusal", "reason": result.reason.value},
            ),
            result,
        )

    return (
        ToolCall(
            tool="score_point",
            arguments=arguments,
            numeric_outputs={
                "expected_crashes": result.expected_crashes,
                "rate_per_100m_vmt": result.rate_per_100m_vmt,
                "rate_lower": result.rate_interval.lower,
                "rate_upper": result.rate_interval.upper,
                "probability_at_least_one": result.probability_at_least_one,
                "probability_lower": result.probability_interval.lower,
                "probability_upper": result.probability_interval.upper,
                "exposure_vehicle_miles": result.exposure_vehicle_miles,
            },
            text_outputs={
                "outcome": "estimate",
                "cell_id": result.cell_id,
                "model": f"{result.model_name}:{result.model_version}",
            },
        ),
        result,
    )


def model_context(service: RiskService) -> ToolCall:
    """The fitted model's headline parameters, for use in explanations."""
    numeric = {"recovered_alpha": service.fitted_alpha}
    report = service.calibration
    if report is not None:
        numeric["expected_calibration_error"] = report.expected_calibration_error
        numeric["coverage_of_nominal_95"] = report.coverage_of_nominal_95

    return ToolCall(
        tool="model_context",
        arguments={},
        numeric_outputs=numeric,
        text_outputs={
            "model": service.model.name,
            "exposure_null_rejected": str(service.model.exposure_null_rejected),
        },
    )
