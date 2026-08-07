"""The hazard model and the alpha-recovery experiment."""

from __future__ import annotations

import numpy as np
import pytest

from born_field.adapters import SyntheticAdapter
from born_field.config import CorruptionConfig, GeneratorConfig
from born_field.experiments.recovery import (
    DEFAULT_GRID,
    Regime,
    aggregate_panel,
    calibrate_k,
    default_regimes,
    run_one,
    summarise,
)
from born_field.models import HazardModel, HazardPanel, PoissonHazardGlm
from born_field.models.glm import REFERENCE_CLASS
from born_field.types import RoadClass

RECOVERY_DAYS = 120


def fitted(**kwargs: object) -> tuple[PoissonHazardGlm, HazardPanel, float]:
    """Fit the hazard model to a fresh world, returning the model and truth."""
    true_alpha = kwargs.pop("true_alpha", 1.35)
    corruption = kwargs.pop("corruption", CorruptionConfig())
    seed = kwargs.pop("seed", 11)
    adapter = SyntheticAdapter(
        grid=DEFAULT_GRID,
        generator=GeneratorConfig(true_alpha=true_alpha, true_k=2e-7, seed=seed),
        corruption=corruption,
        days=RECOVERY_DAYS,
    )
    panel = HazardPanel(adapter.frame())
    model = PoissonHazardGlm(**kwargs)
    model.fit(panel)
    return model, panel, true_alpha


class TestIdentification:
    def test_satisfies_the_hazard_model_protocol(self):
        assert isinstance(PoissonHazardGlm(), HazardModel)

    def test_alpha_is_one_plus_the_log_flow_coefficient(self):
        """The ``+ 1`` is the offset's contribution and must not be dropped."""
        model, panel, _ = fitted()
        diagnostics = model.fit(panel)
        assert model.alpha == pytest.approx(1.0 + diagnostics.params["log_flow"])

    def test_exposure_is_never_a_fitted_coefficient(self):
        """An estimated exposure coefficient would mean modelling counts."""
        model, panel, _ = fitted()
        diagnostics = model.fit(panel)
        assert not any("exposure" in name for name in diagnostics.params)

    def test_offset_is_log_exposure(self):
        _, panel, _ = fitted()
        offset = PoissonHazardGlm._offset(panel)
        assert np.allclose(offset, np.log(panel.exposure))

    def test_reference_class_is_excluded_from_the_design(self):
        """Keeping every level alongside an intercept is rank-deficient."""
        model, panel, _ = fitted()
        diagnostics = model.fit(panel)
        assert f"class_{REFERENCE_CLASS.value}" not in diagnostics.params
        assert "class_motorway" in diagnostics.params


class TestCleanRecovery:
    """Guaranteed by MLE consistency; a failure here is a wiring bug (F2)."""

    def test_recovers_the_configured_alpha(self):
        model, _, true_alpha = fitted()
        interval = model.alpha_interval()
        assert interval.lower <= true_alpha <= interval.upper
        assert model.alpha == pytest.approx(true_alpha, rel=0.05)

    @pytest.mark.parametrize("true_alpha", [1.1, 1.35, 1.8])
    def test_recovers_across_the_range(self, true_alpha):
        model, _, _ = fitted(true_alpha=true_alpha)
        assert model.alpha == pytest.approx(true_alpha, rel=0.08)

    def test_recovers_the_class_multipliers(self):
        """Includes the ordering that distinguishes risk from exposure."""
        model, _, _ = fitted()
        multipliers = model.class_multipliers()
        truth = GeneratorConfig().class_multipliers
        reference = truth[REFERENCE_CLASS]

        for road_class, fitted_value in multipliers.items():
            assert fitted_value == pytest.approx(truth[road_class] / reference, rel=0.15)
        assert multipliers[RoadClass.MOTORWAY] < multipliers[RoadClass.RESIDENTIAL]

    def test_rejects_the_exposure_null_when_alpha_exceeds_one(self):
        model, _, _ = fitted(true_alpha=1.35)
        assert model.exposure_null_rejected

    def test_the_exposure_null_has_approximately_nominal_size(self):
        """F1 must be able to fire, and must not fire spuriously."""
        seeds = (5, 11, 17, 23, 29, 31, 37, 41)
        k = calibrate_k(1.0, target_events=6000.0, days=60)
        rejections = 0
        for seed in seeds:
            adapter = SyntheticAdapter(
                grid=DEFAULT_GRID,
                generator=GeneratorConfig(true_alpha=1.0, true_k=k, seed=seed),
                days=60,
            )
            model = PoissonHazardGlm()
            model.fit(HazardPanel(adapter.frame()))
            rejections += int(model.exposure_null_rejected)

        assert rejections / len(seeds) <= 0.3, (
            f"rejected a true null in {rejections}/{len(seeds)} draws; "
            "F1's false-positive rate is far above nominal"
        )


class TestAttenuation:
    """Measurement error is the regime that actually threatens deployment."""

    def test_measurement_error_attenuates_alpha_toward_zero(self):
        clean, _, _ = fitted(seed=5)
        noisy, _, _ = fitted(seed=5, corruption=CorruptionConfig(flow_lognormal_noise_sd=0.3))
        assert noisy.alpha < clean.alpha

    def test_attenuation_matches_the_predicted_reliability_ratio(self):
        """The bias is a mechanism, not an observation."""
        sigma = 0.5
        adapter = SyntheticAdapter(
            grid=DEFAULT_GRID,
            generator=GeneratorConfig(true_alpha=1.35, true_k=2e-7, seed=5),
            corruption=CorruptionConfig(flow_lognormal_noise_sd=sigma),
            days=RECOVERY_DAYS,
        )
        frame = adapter.frame()

        # Residual variance of log true flow after the model's own covariates.
        log_flow = np.log(frame["true_flow_veh_per_hour"].to_numpy())
        design = [np.ones(len(frame))]
        for level in RoadClass:
            if level is not REFERENCE_CLASS:
                design.append((frame["road_class"].to_numpy() == level.value).astype(float))
        design.append(frame["is_night"].to_numpy(dtype=float))
        design.append(frame["is_wet"].to_numpy(dtype=float))
        matrix = np.column_stack(design)
        residual = log_flow - matrix @ np.linalg.lstsq(matrix, log_flow, rcond=None)[0]

        reliability = residual.var() / (residual.var() + sigma**2)
        predicted = 1.35 * reliability

        model = PoissonHazardGlm()
        model.fit(HazardPanel(frame))
        assert model.alpha == pytest.approx(predicted, rel=0.10)

    def test_severe_measurement_error_can_invert_the_conclusion(self):
        """Pre-registered F5, demonstrated."""
        model, _, _ = fitted(corruption=CorruptionConfig(flow_lognormal_noise_sd=0.6))
        assert model.alpha < 1.0


class TestIntervals:
    def test_every_estimate_carries_an_ordered_interval(self):
        model, panel, _ = fitted()
        small = panel.subset(np.arange(len(panel)) < 300)
        for estimate in model.predict(small)[:50]:
            assert estimate.rate_interval.lower <= estimate.rate_per_100m_vmt
            assert estimate.rate_per_100m_vmt <= estimate.rate_interval.upper
            assert 0.0 <= estimate.probability_at_least_one <= 1.0

    def test_intervals_are_asymmetric_on_the_rate_scale(self):
        """Log-link propagation, not a normal approximation on counts."""
        model, panel, _ = fitted()
        small = panel.subset(np.arange(len(panel)) < 200)
        estimate = model.predict(small)[0]
        below = estimate.rate_per_100m_vmt - estimate.rate_interval.lower
        above = estimate.rate_interval.upper - estimate.rate_per_100m_vmt
        assert above > below
        assert estimate.rate_interval.lower > 0


class TestBeatsBaselines:
    def test_beats_every_baseline_on_holdout_deviance(self, panel):
        """Pre-registered F3. If this fails, the baseline ships instead."""
        from born_field.evaluation import evaluate
        from born_field.models import all_baselines

        candidate = evaluate(PoissonHazardGlm(), panel, n_folds=3, block_cells=2)
        for baseline in all_baselines():
            result = evaluate(baseline, panel, n_folds=3, block_cells=2)
            assert candidate.mean_deviance < result.mean_deviance, (
                f"hazard model failed to beat {baseline.name}"
            )


class TestAggregation:
    def test_aggregation_conserves_counts_and_exposure(self):
        """Both are extensive; aggregation must not create or destroy either."""
        _, panel, _ = fitted()
        coarse = aggregate_panel(panel, 2)
        assert coarse.counts.sum() == pytest.approx(panel.counts.sum())
        assert coarse.exposure.sum() == pytest.approx(panel.exposure.sum())
        assert len(coarse) < len(panel)

    def test_aggregated_flow_preserves_the_exposure_identity(self):
        """``VMT = flow x miles x hours`` must still hold at the coarse scale."""
        _, panel, _ = fitted()
        coarse = aggregate_panel(panel, 2).data
        assert np.allclose(
            coarse["exposure_vehicle_miles"],
            coarse["flow_veh_per_hour"] * coarse["road_miles"],
            rtol=1e-9,
        )

    def test_factor_one_is_the_identity(self):
        _, panel, _ = fitted()
        assert len(aggregate_panel(panel, 1)) == len(panel)

    def test_rejects_invalid_factor(self):
        _, panel, _ = fitted()
        with pytest.raises(ValueError, match="aggregation factor"):
            aggregate_panel(panel, 0)


class TestExperimentPlumbing:
    def test_calibrate_k_hits_the_target_event_count(self):
        """Without this, sweeping alpha compares sample sizes, not estimators."""
        target = 4000.0
        k = calibrate_k(1.6, target, days=40)
        adapter = SyntheticAdapter(
            grid=DEFAULT_GRID,
            generator=GeneratorConfig(true_alpha=1.6, true_k=k, seed=0),
            days=40,
        )
        assert adapter.frame()["true_rate"].sum() == pytest.approx(target, rel=0.01)

    def test_a_run_records_everything_needed_to_reproduce_it(self):
        run = run_one(Regime(name="clean"), seed=3, true_alpha=1.35, true_k=2e-7, days=30)
        assert run.regime == "clean"
        assert run.converged
        assert run.n_events > 0
        assert run.ci_lower < run.recovered_alpha < run.ci_upper

    def test_summary_reports_coverage_separately_from_bias(self):
        """They are different failures: wrong estimate vs wrong uncertainty."""
        runs = [
            run_one(Regime(name="clean"), seed=s, true_alpha=1.35, true_k=2e-7, days=30)
            for s in (1, 2)
        ]
        summary = summarise(runs)
        assert {"mean_bias", "coverage"} <= set(summary.columns)
        assert 0.0 <= float(summary["coverage"].iloc[0]) <= 1.0

    def test_regime_definitions_are_distinct_and_named(self):
        regimes = default_regimes()
        assert len({r.name for r in regimes}) == len(regimes)
        assert regimes[0].name == "clean", "the control must be first"
        assert all(r.description for r in regimes)
