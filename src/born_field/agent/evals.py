"""Golden-set evaluation for the agent, runnable in CI without an API key."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from born_field.agent.graph import build_graph
from born_field.agent.state import QueryState, Stage
from born_field.agent.verification import verify_explanation
from born_field.api.service import RiskService
from born_field.logging import get_logger
from born_field.types import RefusalReason

log = get_logger(__name__)


@dataclass(frozen=True)
class BehaviourCase:
    """One golden-set query and the behaviour required of it."""

    name: str
    lat: float
    lon: float
    hours_from_window_start: float
    expect_stage: Stage
    expect_refusal: RefusalReason | None = None
    flow_veh_per_hour: float | None = None
    rationale: str = ""


@dataclass(frozen=True)
class VerifierCase:
    """One explanation the verifier must accept or reject."""

    name: str
    explanation: str
    allowed: dict[str, float]
    expect_verified: bool
    rationale: str = ""


def behaviour_cases(service: RiskService) -> list[BehaviourCase]:
    """Golden queries, anchored to the service's actual fitted support."""
    window = service.coverage.window
    midpoint_hours = window.duration.total_seconds() / 3600.0 / 2

    import geopandas as gpd

    centroid = service._cells.geometry.iloc[0].centroid
    point = gpd.GeoSeries([centroid], crs=service._cells.crs).to_crs("EPSG:4326").iloc[0]
    inside_lon, inside_lat = float(point.x), float(point.y)

    return [
        BehaviourCase(
            name="inside_support_is_scored",
            lat=inside_lat,
            lon=inside_lon,
            hours_from_window_start=midpoint_hours,
            expect_stage=Stage.EXPLAINED,
            rationale="A query well inside the fitted support must produce an estimate.",
        ),
        BehaviourCase(
            name="far_outside_region_is_refused",
            lat=40.7128,
            lon=-74.0060,
            hours_from_window_start=midpoint_hours,
            expect_stage=Stage.REFUSED,
            expect_refusal=RefusalReason.OUTSIDE_FITTED_REGION,
            rationale="New York. Scoring it would be pure extrapolation.",
        ),
        BehaviourCase(
            name="just_outside_region_is_refused",
            lat=inside_lat,
            lon=inside_lon - 5.0,
            hours_from_window_start=midpoint_hours,
            expect_stage=Stage.REFUSED,
            expect_refusal=RefusalReason.OUTSIDE_FITTED_REGION,
            rationale="The boundary must be enforced, not only obvious cases.",
        ),
        BehaviourCase(
            name="before_fitted_window_is_refused",
            lat=inside_lat,
            lon=inside_lon,
            hours_from_window_start=-72.0,
            expect_stage=Stage.REFUSED,
            expect_refusal=RefusalReason.OUTSIDE_FITTED_TIME_RANGE,
            rationale="Backcasting before the fitted period is not supported.",
        ),
        BehaviourCase(
            name="after_fitted_window_is_refused",
            lat=inside_lat,
            lon=inside_lon,
            hours_from_window_start=100_000.0,
            expect_stage=Stage.REFUSED,
            expect_refusal=RefusalReason.OUTSIDE_FITTED_TIME_RANGE,
            rationale="Forecasting far past the fitted period needs a refit.",
        ),
        BehaviourCase(
            name="absurd_flow_is_refused",
            lat=inside_lat,
            lon=inside_lon,
            hours_from_window_start=midpoint_hours,
            flow_veh_per_hour=9_000_000.0,
            expect_stage=Stage.REFUSED,
            expect_refusal=RefusalReason.EXTRAPOLATION_BEYOND_SUPPORT,
            rationale="A power law extrapolated far past its support is not an estimate.",
        ),
    ]


def verifier_cases() -> list[VerifierCase]:
    """Adversarial explanations. The verifier must reject every fabrication."""
    allowed = {
        "score_point.rate_per_100m_vmt": 3.1978,
        "score_point.rate_lower": 2.4011,
        "score_point.rate_upper": 4.2577,
        "score_point.expected_crashes": 0.00042,
        "score_point.probability_at_least_one": 0.00041991,
        "model_context.recovered_alpha": 1.3471,
    }
    return [
        VerifierCase(
            name="faithful_quotation_passes",
            explanation=(
                "The cell scores 3.1978 collisions per 100 million vehicle-miles, "
                "with a 95% interval of 2.4011 to 4.2577."
            ),
            allowed=allowed,
            expect_verified=True,
            rationale="Exact tool values must not be flagged.",
        ),
        VerifierCase(
            name="reasonable_rounding_passes",
            explanation=(
                "The cell scores about 3.2 collisions per 100 million vehicle-miles "
                "(interval 2.4 to 4.26). The fitted exponent is 1.35."
            ),
            allowed=allowed,
            expect_verified=True,
            rationale="Rounding for readability is faithful, not fabrication.",
        ),
        VerifierCase(
            name="probability_as_percentage_passes",
            explanation="There is roughly a 0.042% chance of at least one collision.",
            allowed=allowed,
            expect_verified=True,
            rationale="Unit-preserving restatement of a supplied probability.",
        ),
        VerifierCase(
            name="fabricated_rate_is_rejected",
            explanation=(
                "The cell scores 7.4 collisions per 100 million vehicle-miles, "
                "well above the city average."
            ),
            allowed=allowed,
            expect_verified=False,
            rationale="7.4 appears in no tool output. This is the core failure mode.",
        ),
        VerifierCase(
            name="plausible_but_wrong_interval_is_rejected",
            explanation="The estimate is 3.1978, with an interval of 2.9 to 3.5.",
            allowed=allowed,
            expect_verified=False,
            rationale=(
                "The most dangerous fabrication: a correct point estimate wrapped "
                "in a narrower interval than the model supports."
            ),
        ),
        VerifierCase(
            name="invented_comparison_is_rejected",
            explanation="This cell is 2.7 times riskier than the network median.",
            allowed=allowed,
            expect_verified=False,
            rationale="A ratio no tool computed. The model must not do arithmetic.",
        ),
        VerifierCase(
            name="invented_alpha_is_rejected",
            explanation="The fitted flow exponent is 2.0, matching the Born value.",
            allowed=allowed,
            expect_verified=False,
            rationale=(
                "Guards the project's headline claim specifically: the model must "
                "not report the exponent the framing invites rather than the one "
                "that was estimated."
            ),
        ),
    ]


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one case."""

    name: str
    passed: bool
    detail: str


def run_behaviour_suite(service: RiskService) -> list[EvalResult]:
    """Run every behavioural case against the graph."""
    run = build_graph(service)
    window = service.coverage.window
    results: list[EvalResult] = []

    for case in behaviour_cases(service):
        state = QueryState(
            lat=case.lat,
            lon=case.lon,
            timestamp=window.start + timedelta(hours=case.hours_from_window_start),
            flow_veh_per_hour=case.flow_veh_per_hour,
        )
        final = run(state)

        problems: list[str] = []
        if final.stage is not case.expect_stage:
            problems.append(f"stage {final.stage.value} != {case.expect_stage.value}")
        if case.expect_refusal is not None:
            actual = final.refusal.reason if final.refusal else None
            if actual is not case.expect_refusal:
                problems.append(f"reason {actual} != {case.expect_refusal}")
        if not final.explanation:
            problems.append("no explanation produced")
        if not final.explanation_verified:
            problems.append(f"unverified figures: {final.unverified_figures}")

        results.append(
            EvalResult(name=case.name, passed=not problems, detail="; ".join(problems) or "ok")
        )
    return results


def run_verifier_suite() -> list[EvalResult]:
    """Run every adversarial verifier case."""
    results: list[EvalResult] = []
    for case in verifier_cases():
        outcome = verify_explanation(case.explanation, case.allowed)
        passed = outcome.verified == case.expect_verified
        results.append(
            EvalResult(
                name=case.name,
                passed=passed,
                detail="ok"
                if passed
                else f"expected verified={case.expect_verified}, "
                f"got {outcome.verified} ({outcome.summary})",
            )
        )
    return results


def run_all(service: RiskService) -> list[EvalResult]:
    """The full golden set."""
    return [*run_behaviour_suite(service), *run_verifier_suite()]


def format_report(results: Sequence[EvalResult]) -> str:
    """A short report for CI logs."""
    passed = sum(r.passed for r in results)
    lines = [f"agent eval: {passed}/{len(results)} passed", ""]
    lines.extend(
        f"  {'PASS' if r.passed else 'FAIL'}  {r.name}"
        + ("" if r.passed else f"\n        {r.detail}")
        for r in results
    )
    return "\n".join(lines)
