"""The scoring service, the refusal contract and the HTTP surface."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from born_field.api import main as api_main
from born_field.api.service import (
    MIN_TRAINING_ROWS_PER_CELL,
    RiskService,
    build_demo_service,
    midpoint_of,
)
from born_field.types import HazardEstimate, Refusal, RefusalReason

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def service() -> RiskService:
    """One fitted service for the whole module; fitting is the slow part."""
    return build_demo_service(days=45)


@pytest.fixture(scope="module")
def client(service: RiskService):
    """A test client with the fitted service injected, skipping startup refit."""
    api_main._state["service"] = service
    with TestClient(api_main.app) as test_client:
        yield test_client
    api_main._state.clear()


@pytest.fixture
def inside(service: RiskService) -> tuple[float, float, datetime]:
    """A coordinate and timestamp known to be inside the fitted support."""
    cell = service._cells.iloc[0]
    centroid = service._cells.geometry.iloc[0].centroid
    import geopandas as gpd

    point = gpd.GeoSeries([centroid], crs=service._cells.crs).to_crs("EPSG:4326").iloc[0]
    assert cell is not None
    return float(point.x), float(point.y), midpoint_of(service.coverage.window)


class TestServiceScoring:
    def test_scores_a_point_inside_support(self, service: RiskService, inside):
        lon, lat, timestamp = inside
        result = service.score(lon, lat, timestamp)
        assert isinstance(result, HazardEstimate)
        assert result.expected_crashes >= 0
        assert result.rate_interval.lower <= result.rate_interval.upper

    def test_requires_timezone_aware_timestamps(self, service: RiskService, inside):
        """A naive timestamp is ambiguous, and risk varies strongly by hour."""
        lon, lat, _ = inside
        with pytest.raises(ValueError, match="timezone-aware"):
            service.score(lon, lat, datetime(2026, 1, 1))

    def test_supplied_flow_changes_the_estimate(self, service: RiskService, inside):
        """Exposure must actually depend on the caller's flow, not be decorative."""
        lon, lat, timestamp = inside
        low = service.score(lon, lat, timestamp, flow_veh_per_hour=200.0)
        high = service.score(lon, lat, timestamp, flow_veh_per_hour=1500.0)
        assert isinstance(low, HazardEstimate)
        assert isinstance(high, HazardEstimate)
        assert high.exposure_vehicle_miles > low.exposure_vehicle_miles
        assert high.expected_crashes > low.expected_crashes

    def test_wet_weather_raises_risk(self, service: RiskService, inside):
        lon, lat, timestamp = inside
        dry = service.score(lon, lat, timestamp, is_wet=False)
        wet = service.score(lon, lat, timestamp, is_wet=True)
        assert isinstance(dry, HazardEstimate)
        assert isinstance(wet, HazardEstimate)
        assert wet.expected_crashes > dry.expected_crashes


class TestRefusalPaths:
    """Each refusal is a different instruction to the caller."""

    def test_refuses_outside_the_region(self, service: RiskService, inside):
        _, _, timestamp = inside
        result = service.score(-74.0060, 40.7128, timestamp)  # New York
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.OUTSIDE_FITTED_REGION

    def test_refuses_before_the_fitted_window(self, service: RiskService, inside):
        lon, lat, _ = inside
        result = service.score(lon, lat, service.coverage.window.start - timedelta(days=1))
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.OUTSIDE_FITTED_TIME_RANGE

    def test_refuses_after_the_fitted_window(self, service: RiskService, inside):
        lon, lat, _ = inside
        result = service.score(lon, lat, service.coverage.window.end + timedelta(days=1))
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.OUTSIDE_FITTED_TIME_RANGE

    def test_refuses_extrapolation_beyond_fitted_flow(self, service: RiskService, inside):
        """A power law extrapolated past its support is not a risk estimate."""
        lon, lat, timestamp = inside
        result = service.score(lon, lat, timestamp, flow_veh_per_hour=5_000_000.0)
        assert isinstance(result, Refusal)
        assert result.reason is RefusalReason.EXTRAPOLATION_BEYOND_SUPPORT

    def test_refusal_carries_actionable_detail(self, service: RiskService, inside):
        _, _, timestamp = inside
        result = service.score(-74.0060, 40.7128, timestamp)
        assert isinstance(result, Refusal)
        assert "outside the fitted study area" in result.detail

    def test_coverage_threshold_is_explicit(self):
        """A product decision, not a statistical accident."""
        assert MIN_TRAINING_ROWS_PER_CELL > 0


class TestHttpContract:
    def test_health_is_unauthenticated(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_scoring_requires_an_api_key(self, client, inside):
        lon, lat, timestamp = inside
        response = client.post(
            "/v1/score",
            json={"lat": lat, "lon": lon, "timestamp": timestamp.isoformat()},
        )
        assert response.status_code == 401

    def test_score_returns_an_estimate(self, client, inside):
        lon, lat, timestamp = inside
        response = client.post(
            "/v1/score",
            headers=HEADERS,
            json={"lat": lat, "lon": lon, "timestamp": timestamp.isoformat()},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "estimate"
        assert body["rate_interval"]["lower"] <= body["rate_per_100m_vmt"]

    def test_refusal_is_http_200_not_an_error(self, client, inside):
        """A refusal is a computed answer, not a client mistake."""
        _, _, timestamp = inside
        response = client.post(
            "/v1/score",
            headers=HEADERS,
            json={"lat": 40.7128, "lon": -74.0060, "timestamp": timestamp.isoformat()},
        )
        assert response.status_code == 200
        assert response.json()["outcome"] == "refusal"
        assert response.json()["reason"] == "outside_fitted_region"

    def test_batch_mixes_estimates_and_refusals_without_failing(self, client, inside):
        """One refusal must not discard the rest of a network screening run."""
        lon, lat, timestamp = inside
        response = client.post(
            "/v1/score/batch",
            headers=HEADERS,
            json={
                "queries": [
                    {"lat": lat, "lon": lon, "timestamp": timestamp.isoformat()},
                    {"lat": 40.7128, "lon": -74.0060, "timestamp": timestamp.isoformat()},
                ]
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["n_estimates"] == 1
        assert body["n_refusals"] == 1
        assert [r["outcome"] for r in body["results"]] == ["estimate", "refusal"]

    def test_rejects_impossible_coordinates(self, client, inside):
        _, _, timestamp = inside
        response = client.post(
            "/v1/score",
            headers=HEADERS,
            json={"lat": 999.0, "lon": 0.0, "timestamp": timestamp.isoformat()},
        )
        assert response.status_code == 422

    def test_rejects_unknown_fields(self, client, inside):
        """extra='forbid': a typo in a field name must not be silently ignored."""
        lon, lat, timestamp = inside
        response = client.post(
            "/v1/score",
            headers=HEADERS,
            json={
                "lat": lat,
                "lon": lon,
                "timestamp": timestamp.isoformat(),
                "is_wett": True,
            },
        )
        assert response.status_code == 422

    def test_calibration_endpoint_reports_reliability(self, client):
        response = client.get("/v1/calibration", headers=HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["bins"]
        assert body["expected_calibration_error"] >= 0

    def test_model_endpoint_publishes_alpha_and_support(self, client):
        response = client.get("/v1/model", headers=HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert body["recovered_alpha"] > 0
        assert "bbox" in body["support"]
        assert "flow_support" in body["support"]

    def test_metrics_are_exposed_for_prometheus(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "bornfield_requests_total" in response.text

    def test_openapi_documents_both_outcomes(self, client):
        """The refusal branch must be visible in the schema, not discovered."""
        schema = client.get("/openapi.json").json()
        assert "RefusalOut" in schema["components"]["schemas"]
        assert "EstimateOut" in schema["components"]["schemas"]
        assert "/v1/score" in schema["paths"]


class TestPersistenceSchema:
    """Schema-level checks that need no live database."""

    def test_tables_are_declared(self):
        from born_field.db import Base

        assert {"cells", "segments", "score_records"} <= set(Base.metadata.tables)

    def test_cells_have_a_spatial_index(self):
        """Without GiST the point lookup is a sequential scan."""
        from born_field.db import Base

        indexes = Base.metadata.tables["cells"].indexes
        assert any(i.name == "ix_cells_geometry" for i in indexes)

    def test_score_records_can_hold_a_refusal(self):
        """Refusals are persisted, not dropped: they are the drift signal."""
        from born_field.db import Base

        columns = Base.metadata.tables["score_records"].columns
        assert "refusal_reason" in columns
        assert columns["expected_crashes"].nullable is True


class TestScoringResultUnion:
    def test_callers_must_handle_both_arms(self, service: RiskService, inside):
        """ScoringResult is a union; neither arm may be assumed away."""
        lon, lat, timestamp = inside
        results = [
            service.score(lon, lat, timestamp),
            service.score(-74.0060, 40.7128, timestamp),
        ]
        assert {type(r) for r in results} == {HazardEstimate, Refusal}


def test_utc_timestamps_round_trip(service: RiskService):
    window = service.coverage.window
    assert window.start.tzinfo is UTC or window.start.utcoffset() is not None
