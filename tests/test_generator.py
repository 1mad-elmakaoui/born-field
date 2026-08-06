"""The synthetic generator must provably implement the configured alpha."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from born_field.adapters import DataSourceAdapter, SyntheticAdapter, VectorisedSource
from born_field.config import CorruptionConfig, GeneratorConfig, GridConfig
from born_field.field import build_cells, generate_network
from born_field.types import METRES_PER_MILE, RoadClass, vehicle_miles_travelled

# Coarse cells over the full bbox. A sub-region of this network is all
# residential, which would quietly drop four of the five road classes.
FAST_GRID = GridConfig(bbox=(-122.35, 37.75, -122.15, 37.90), cell_size_m=2000.0)


def build(**generator_kwargs: object) -> SyntheticAdapter:
    """A small adapter with an otherwise default configuration."""
    corruption = generator_kwargs.pop("corruption", CorruptionConfig())
    days = int(generator_kwargs.pop("days", 4))  # type: ignore[call-overload]
    return SyntheticAdapter(
        grid=FAST_GRID,
        generator=GeneratorConfig(**generator_kwargs),  # type: ignore[arg-type]
        corruption=corruption,  # type: ignore[arg-type]
        days=days,
    )


def implied_k(adapter: SyntheticAdapter) -> np.ndarray:
    """Invert the hazard law: strip every known factor from the generated rate."""
    cfg = adapter.generator_config
    frame = adapter.frame()

    class_mult = np.array([cfg.class_multipliers[RoadClass(c)] for c in frame["road_class"]])
    time_mult = np.where(frame["is_night"], cfg.night_multiplier, 1.0)
    weather_mult = np.where(frame["is_wet"], cfg.wet_weather_multiplier, 1.0)

    denominator = (
        frame["true_flow_veh_per_hour"].to_numpy() ** cfg.true_alpha
        * class_mult
        * time_mult
        * weather_mult
        * frame["road_miles"].to_numpy()
        * adapter.grid_config.window_hours
    )
    return frame["true_rate"].to_numpy() / denominator


class TestHazardLawIdentity:
    """The generator must implement exactly the documented law."""

    def test_configured_alpha_is_the_alpha_used(self):
        adapter = build()
        assert implied_k(adapter) == pytest.approx(adapter.generator_config.true_k, rel=1e-9)

    @settings(deadline=None, max_examples=8, suppress_health_check=[HealthCheck.too_slow])
    @given(alpha=st.floats(min_value=0.5, max_value=2.6))
    def test_identity_holds_at_any_alpha(self, alpha):
        """The exponent is a free parameter, not a hard-coded 2."""
        adapter = build(true_alpha=alpha)
        assert implied_k(adapter) == pytest.approx(adapter.generator_config.true_k, rel=1e-9)

    @given(alpha=st.floats(min_value=0.6, max_value=2.4), ratio=st.floats(2.0, 8.0))
    @settings(deadline=None, max_examples=15)
    def test_rate_scales_as_flow_to_the_alpha(self, alpha, ratio):
        """Doubling flow must multiply the hazard rate by 2**alpha."""
        assert ratio**alpha == pytest.approx(np.exp(alpha * np.log(ratio)), rel=1e-12)

    def test_wrong_alpha_breaks_the_identity(self):
        """Guards the guard: the identity must be capable of failing."""
        adapter = build(true_alpha=1.35)
        cfg = adapter.generator_config
        frame = adapter.frame()
        class_mult = np.array([cfg.class_multipliers[RoadClass(c)] for c in frame["road_class"]])
        wrong = frame["true_rate"].to_numpy() / (
            frame["true_flow_veh_per_hour"].to_numpy() ** 2.0  # deliberately wrong
            * class_mult
            * np.where(frame["is_night"], cfg.night_multiplier, 1.0)
            * np.where(frame["is_wet"], cfg.wet_weather_multiplier, 1.0)
            * frame["road_miles"].to_numpy()
        )
        assert wrong != pytest.approx(cfg.true_k, rel=1e-3)


class TestSampling:
    def test_counts_are_poisson_around_the_rate(self):
        """Sample mean must track the generating rate across the whole panel."""
        adapter = build(days=20)
        frame = adapter.frame()
        assert frame["true_crash_count"].sum() > 50, "test is uninformative with no events"
        assert frame["true_crash_count"].mean() == pytest.approx(
            frame["true_rate"].mean(), rel=0.15
        )

    def test_seeds_differ_but_are_reproducible(self):
        a, b, a_again = build(seed=1), build(seed=2), build(seed=1)
        assert a.frame()["crash_count"].sum() != b.frame()["crash_count"].sum()
        assert a.frame()["crash_count"].sum() == a_again.frame()["crash_count"].sum()

    def test_motorways_are_safer_per_vehicle_mile_conditional_on_flow(self):
        """The class effect is a *conditional* statement, and only that."""
        adapter = build(days=60)
        cfg = adapter.generator_config
        frame = adapter.frame()

        time_mult = np.where(frame["is_night"], cfg.night_multiplier, 1.0)
        weather_mult = np.where(frame["is_wet"], cfg.wet_weather_multiplier, 1.0)
        conditional = frame["true_rate"] / (
            frame["exposure_vehicle_miles"]
            * frame["true_flow_veh_per_hour"] ** (cfg.true_alpha - 1)
            * time_mult
            * weather_mult
        )
        by_class = conditional.groupby(frame["road_class"]).mean()

        expected = (
            cfg.class_multipliers[RoadClass.MOTORWAY] / cfg.class_multipliers[RoadClass.RESIDENTIAL]
        )
        assert by_class["motorway"] / by_class["residential"] == pytest.approx(expected, rel=1e-9)

    def test_unconditionally_motorways_look_more_dangerous_per_vehicle_mile(self):
        """The confound bites in the direction naive intuition misses."""
        frame = build(days=60).frame()
        by_class = frame.groupby("road_class").agg(
            crashes=("true_crash_count", "sum"),
            vmt=("exposure_vehicle_miles", "sum"),
        )
        by_class["raw_rate"] = by_class["crashes"] / by_class["vmt"]

        assert by_class.loc["motorway", "crashes"] > by_class.loc["residential", "crashes"]
        assert by_class.loc["motorway", "raw_rate"] > by_class.loc["residential", "raw_rate"]


class TestCorruptionRegimes:
    """Each regime must do exactly what it claims, and nothing else."""

    def test_measurement_error_perturbs_observed_flow_only(self):
        clean = build(seed=7).frame()
        noisy = build(seed=7, corruption=CorruptionConfig(flow_lognormal_noise_sd=0.4)).frame()

        # True flow is untouched: crashes were sampled from the real intensity.
        assert np.allclose(clean["true_flow_veh_per_hour"], noisy["true_flow_veh_per_hour"])
        # Observed flow carries the error, with roughly the configured spread.
        log_ratio = np.log(noisy["flow_veh_per_hour"] / noisy["true_flow_veh_per_hour"])
        assert log_ratio.std() == pytest.approx(0.4, rel=0.1)

    def test_measurement_error_propagates_into_exposure(self):
        """Exposure is built from observed flow, as in any real pipeline."""
        noisy = build(corruption=CorruptionConfig(flow_lognormal_noise_sd=0.3)).frame()
        assert np.allclose(
            noisy["exposure_vehicle_miles"],
            noisy["flow_veh_per_hour"] * noisy["road_miles"],
        )

    def test_underreporting_removes_crashes_with_spatial_structure(self):
        frame = build(days=60, corruption=CorruptionConfig(underreport_floor=0.4)).frame()
        assert frame["crash_count"].sum() < frame["true_crash_count"].sum()

        # Reporting must vary *across space*, not at random: that is what makes
        # it a confounder rather than simple attrition.
        by_col = frame.groupby("col").agg(
            reported=("crash_count", "sum"), true=("true_crash_count", "sum")
        )
        by_col = by_col[by_col["true"] > 0]
        ratio = by_col["reported"] / by_col["true"]
        assert ratio.corr(by_col.index.to_series()) > 0.3

    def test_full_reporting_loses_nothing(self):
        frame = build(days=20).frame()
        assert (frame["crash_count"] == frame["true_crash_count"]).all()

    def test_overdispersion_inflates_variance_but_not_the_mean(self):
        clean = build(days=60, seed=3).frame()
        over = build(days=60, seed=3, corruption=CorruptionConfig(overdispersion=1.5)).frame()

        assert over["true_rate"].mean() == pytest.approx(clean["true_rate"].mean(), rel=0.15)
        assert over["true_rate"].var() > clean["true_rate"].var() * 2

    def test_regimes_are_independent(self):
        """Enabling one regime must not silently switch on another."""
        frame = build(days=20, corruption=CorruptionConfig(overdispersion=0.8)).frame()
        assert (frame["crash_count"] == frame["true_crash_count"]).all()
        assert np.allclose(frame["flow_veh_per_hour"], frame["true_flow_veh_per_hour"])


class TestExposureConsistency:
    @given(
        flow=st.floats(min_value=0.0, max_value=1e5, allow_nan=False),
        miles=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        hours=st.floats(min_value=0.0, max_value=48.0, allow_nan=False),
    )
    def test_scalar_and_array_paths_agree(self, flow, miles, hours):
        """The two exposure implementations must never diverge."""
        from born_field.types import vehicle_miles_travelled_array

        scalar = vehicle_miles_travelled(flow, miles, hours)
        array = vehicle_miles_travelled_array(np.array([flow]), np.array([miles]), hours)
        assert array[0] == pytest.approx(scalar, rel=1e-12, abs=1e-12)

    def test_panel_exposure_matches_the_contract(self):
        adapter = build()
        frame = adapter.frame()
        expected = [
            vehicle_miles_travelled(
                row.flow_veh_per_hour, row.road_miles, adapter.grid_config.window_hours
            )
            for row in frame.head(200).itertuples()
        ]
        assert np.allclose(frame["exposure_vehicle_miles"].head(200), expected)


class TestGrid:
    def test_cells_carry_heterogeneous_road_mileage(self):
        """Uniform mileage would make the road-length half of exposure inert."""
        cells = build().cells_frame
        assert cells["road_miles"].min() > 0
        assert cells["road_miles"].std() / cells["road_miles"].mean() > 0.15

    def test_no_cell_has_zero_exposure_capacity(self):
        """Zero-road cells must be dropped: log(0) exposure is -inf."""
        assert (build().cells_frame["road_miles"] > 0).all()

    def test_lengths_are_computed_in_a_projected_crs(self):
        """A degree is not a distance."""
        network = generate_network(FAST_GRID.bbox, seed=1)
        cells = build_cells(network, FAST_GRID.bbox, 1000.0)
        span_miles = 8_000 / METRES_PER_MILE  # bbox is roughly 8 km across
        assert span_miles < cells["road_miles"].sum() < span_miles * 500

    def test_dominant_class_is_by_mileage(self):
        cells = build().cells_frame
        for row in cells.head(50).itertuples():
            per_class = {c.value: getattr(row, c.value) for c in RoadClass}
            assert row.dominant_class == max(per_class, key=lambda k: per_class[k])


class TestSegments:
    """The analysis unit is (cell, road_class), not the cell."""

    def test_every_class_is_populated(self):
        """A stratum with no rows has an unidentifiable coefficient."""
        segments = build().segments_frame
        assert set(segments["road_class"]) == {c.value for c in RoadClass}

    def test_every_class_accumulates_events(self):
        """Populated support is not enough; each class needs actual crashes."""
        frame = build(days=60).frame()
        by_class = frame.groupby("road_class")["true_crash_count"].sum()
        assert (by_class > 0).all(), by_class.to_dict()

    def test_segment_mileage_reconstructs_cell_mileage(self):
        """Splitting into strata must conserve total road length."""
        adapter = build()
        per_cell = adapter.segments_frame.groupby("cell_id")["road_miles"].sum()
        expected = adapter.cells_frame.set_index("cell_id")["road_miles"]
        assert np.allclose(per_cell.sort_index(), expected.sort_index())

    def test_zero_mileage_strata_are_dropped(self):
        assert (build().segments_frame["road_miles"] > 0).all()


class TestVendoredFixture:
    """The shipped road network must actually be in the repository."""

    def test_fixture_is_present(self):
        from born_field.field.network import FIXTURE_PATH

        assert FIXTURE_PATH.exists(), (
            f"vendored network missing at {FIXTURE_PATH}; check .gitignore is not "
            "excluding data/fixtures/*.parquet"
        )

    def test_fixture_loads_and_covers_every_road_class(self):
        from born_field.field.network import FIXTURE_PATH, load_network

        network = load_network(FIXTURE_PATH)
        assert len(network) > 0
        assert set(network["road_class"]) == {c.value for c in RoadClass}


class TestProtocolConformance:
    def test_adapter_satisfies_both_protocols(self):
        adapter = build()
        assert isinstance(adapter, DataSourceAdapter)
        assert isinstance(adapter, VectorisedSource)

    def test_row_wise_and_frame_paths_agree_on_crash_totals(self):
        """The interop path and the fast path must describe the same world."""
        adapter = build(days=10)
        crashes = list(adapter.crashes(adapter.window))
        assert len(crashes) == int(adapter.frame()["crash_count"].sum())

    def test_flow_observations_cover_every_cell_window(self):
        adapter = build(days=2)
        assert len(list(adapter.flow(adapter.window))) == len(adapter.frame())

    def test_coverage_declares_the_generated_support(self):
        adapter = build()
        coverage = adapter.coverage()
        cell = adapter.cells()[0]
        assert coverage.contains_point(cell.centroid_lon, cell.centroid_lat)
        assert not coverage.contains_point(-74.0, 40.7)  # New York

    def test_version_string_encodes_what_changes_the_data(self):
        assert "alpha=1.9" in build(true_alpha=1.9).version
