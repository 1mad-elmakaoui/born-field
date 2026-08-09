"""The query graph, refusal routing and the numeric verifier."""

from __future__ import annotations

import pytest

from born_field.agent import QueryState, Stage, build_graph, template_explainer
from born_field.agent.evals import run_all, run_behaviour_suite, run_verifier_suite
from born_field.agent.verification import RELATIVE_TOLERANCE, verify_explanation
from born_field.api.service import RiskService, build_demo_service, midpoint_of
from born_field.types import RefusalReason


@pytest.fixture(scope="module")
def service() -> RiskService:
    return build_demo_service(days=45)


@pytest.fixture(scope="module")
def inside(service: RiskService):
    import geopandas as gpd

    centroid = service._cells.geometry.iloc[0].centroid
    point = gpd.GeoSeries([centroid], crs=service._cells.crs).to_crs("EPSG:4326").iloc[0]
    return float(point.x), float(point.y), midpoint_of(service.coverage.window)


class TestGraphRouting:
    def test_scores_and_explains_inside_support(self, service, inside):
        lon, lat, timestamp = inside
        final = build_graph(service)(QueryState(lat=lat, lon=lon, timestamp=timestamp))
        assert final.stage is Stage.EXPLAINED
        assert final.estimate is not None
        assert final.refusal is None
        assert final.explanation

    def test_refusal_is_a_terminal_state_not_an_exception(self, service, inside):
        _, _, timestamp = inside
        final = build_graph(service)(QueryState(lat=40.7128, lon=-74.0060, timestamp=timestamp))
        assert final.stage is Stage.REFUSED
        assert final.refusal is not None
        assert final.refusal.reason is RefusalReason.OUTSIDE_FITTED_REGION
        assert final.estimate is None

    def test_refusals_still_carry_an_explanation(self, service, inside):
        """A decline the caller cannot read is barely better than a crash."""
        _, _, timestamp = inside
        final = build_graph(service)(QueryState(lat=40.7128, lon=-74.0060, timestamp=timestamp))
        assert final.explanation
        assert final.refusal is not None
        assert final.refusal.reason.value in final.explanation

    def test_every_run_records_its_tool_calls(self, service, inside):
        """No tool record means no ground truth for the verifier."""
        lon, lat, timestamp = inside
        final = build_graph(service)(QueryState(lat=lat, lon=lon, timestamp=timestamp))
        tools = {c.tool for c in final.calls}
        assert {"model_context", "locate_cell", "score_point"} <= tools
        assert final.allowed_numbers

    def test_state_is_typed_not_a_dict(self, service, inside):
        lon, lat, timestamp = inside
        final = build_graph(service)(QueryState(lat=lat, lon=lon, timestamp=timestamp))
        assert isinstance(final, QueryState)

    def test_exactly_one_outcome_arm_is_set(self, service, inside):
        lon, lat, timestamp = inside
        for state in (
            QueryState(lat=lat, lon=lon, timestamp=timestamp),
            QueryState(lat=40.7128, lon=-74.0060, timestamp=timestamp),
        ):
            final = build_graph(service)(state)
            assert (final.estimate is None) != (final.refusal is None)


class TestExplainerContainment:
    def test_a_fabricating_explainer_never_reaches_the_caller(self, service, inside):
        """The core guarantee, exercised end to end."""
        lon, lat, timestamp = inside

        def fabricator(_: QueryState) -> str:
            return "This cell scores 9999.5 per 100M VMT, 12.7 times the city average."

        final = build_graph(service, fabricator)(QueryState(lat=lat, lon=lon, timestamp=timestamp))
        assert final.explanation_verified is False
        assert final.unverified_figures
        assert "9999.5" not in (final.explanation or "")

    def test_a_faithful_explainer_is_passed_through(self, service, inside):
        lon, lat, timestamp = inside

        def faithful(state: QueryState) -> str:
            estimate = state.estimate
            assert estimate is not None
            return f"The rate is {estimate.rate_per_100m_vmt} per 100 million vehicle-miles."

        final = build_graph(service, faithful)(QueryState(lat=lat, lon=lon, timestamp=timestamp))
        assert final.explanation_verified is True
        assert "The rate is" in (final.explanation or "")

    def test_an_exploding_explainer_degrades_rather_than_failing(self, service, inside):
        """Observability and prose are never load-bearing for a risk answer."""
        lon, lat, timestamp = inside

        def broken(_: QueryState) -> str:
            raise RuntimeError("model provider is down")

        final = build_graph(service, broken)(QueryState(lat=lat, lon=lon, timestamp=timestamp))
        assert final.stage is Stage.EXPLAINED
        assert final.explanation
        assert final.error is not None

    def test_template_explainer_is_self_consistent(self, service, inside):
        """The fallback must itself pass verification, or there is no fallback."""
        lon, lat, timestamp = inside
        final = build_graph(service)(QueryState(lat=lat, lon=lon, timestamp=timestamp))
        result = verify_explanation(
            template_explainer(final), final.allowed_numbers, (final.lat, final.lon)
        )
        assert result.verified, result.summary


class TestVerifier:
    def test_exact_quotation_verifies(self):
        assert verify_explanation("The rate is 3.1978.", {"a": 3.1978}).verified

    def test_faithful_rounding_verifies(self):
        assert verify_explanation("About 3.2.", {"a": 3.1978}).verified

    def test_fabrication_is_caught(self):
        result = verify_explanation("The rate is 7.4.", {"a": 3.1978})
        assert not result.verified
        assert "7.4" in result.unverified

    def test_zero_decimal_rounding_is_not_treated_as_faithful(self):
        """Regression: enumerated roundings once authorised a 17% error."""
        assert not verify_explanation("The value is 2.0.", {"a": 2.4011}).verified

    def test_decimal_form_is_never_structurally_exempt(self):
        """Regression: "2.0" was exempted as a small integer."""
        assert not verify_explanation("Exponent is 2.0.", {"a": 1.3471}).verified
        assert verify_explanation("See item 2.", {"a": 1.3471}).verified

    def test_structural_integers_do_not_trip_the_check(self):
        assert verify_explanation(
            "A 95% interval over 100 million vehicle-miles.", {"a": 3.1978}
        ).verified

    def test_percentage_restatement_verifies(self):
        assert verify_explanation("About 4.2%.", {"p": 0.042}).verified

    def test_query_echo_is_not_fabrication(self):
        """Repeating the caller's own coordinates back is not invention."""
        assert verify_explanation(
            "At -122.2599 37.8201 the estimate applies.",
            {"a": 3.1978},
            query_values=(-122.2599, 37.8201),
        ).verified

    def test_tolerance_boundary_is_enforced(self):
        base = 100.0
        inside = base * (1 + RELATIVE_TOLERANCE / 2)
        outside = base * (1 + RELATIVE_TOLERANCE * 4)
        assert verify_explanation(f"Value {inside:.4f}.", {"a": base}).verified
        assert not verify_explanation(f"Value {outside:.4f}.", {"a": base}).verified

    def test_summary_names_the_offending_figures(self):
        result = verify_explanation("Rates of 8.1 and 9.2.", {"a": 3.1978})
        assert "8.1" in result.summary
        assert "9.2" in result.summary


class TestGoldenSet:
    def test_behaviour_suite_passes(self, service):
        results = run_behaviour_suite(service)
        failures = [r for r in results if not r.passed]
        assert not failures, [f"{r.name}: {r.detail}" for r in failures]

    def test_verifier_suite_passes(self):
        results = run_verifier_suite()
        failures = [r for r in results if not r.passed]
        assert not failures, [f"{r.name}: {r.detail}" for r in failures]

    def test_suite_covers_every_refusal_reason_the_service_can_emit(self, service):
        """A refusal path with no golden case can regress unnoticed."""
        from born_field.agent.evals import behaviour_cases

        covered = {c.expect_refusal for c in behaviour_cases(service) if c.expect_refusal}
        assert RefusalReason.OUTSIDE_FITTED_REGION in covered
        assert RefusalReason.OUTSIDE_FITTED_TIME_RANGE in covered
        assert RefusalReason.EXTRAPOLATION_BEYOND_SUPPORT in covered

    def test_full_suite_reports_all_cases(self, service):
        assert len(run_all(service)) >= 12
