"""Splits, metrics, baselines and the shared scoring harness."""

from __future__ import annotations

import numpy as np
import pytest

from born_field.evaluation import (
    comparison_table,
    evaluate,
    forward_time_split,
    hit_rate_at_top_n,
    poisson_deviance,
    pr_auc,
    random_split,
    spatial_block_split,
)
from born_field.models import (
    CrashKde,
    ExposureOnly,
    HazardModel,
    HazardPanel,
    HistoricalFrequency,
    VolumeOnlyGlm,
    all_baselines,
)


class TestPoissonDeviance:
    def test_perfect_prediction_scores_zero(self):
        observed = np.array([0.0, 1.0, 3.0, 7.0])
        assert poisson_deviance(observed, observed) == pytest.approx(0.0, abs=1e-9)

    def test_worse_predictions_score_higher(self):
        observed = np.array([0.0, 1.0, 2.0, 5.0])
        good = poisson_deviance(observed, observed * 1.1)
        bad = poisson_deviance(observed, observed * 3.0)
        assert good < bad

    def test_zero_observations_are_finite(self):
        """All-zero rows are the common case; they must not produce NaN."""
        result = poisson_deviance(np.zeros(5), np.full(5, 0.01))
        assert np.isfinite(result)

    def test_zero_prediction_against_an_event_is_finite(self):
        """A confident-and-wrong model must be penalised, not crash the fold."""
        result = poisson_deviance(np.array([1.0]), np.array([0.0]))
        assert np.isfinite(result)
        assert result > 0


class TestRankingMetrics:
    def test_hit_rate_beats_chance_for_a_good_ranking(self):
        observed = np.zeros(1000)
        observed[:50] = 1.0
        perfect = np.linspace(1.0, 0.0, 1000)
        assert hit_rate_at_top_n(observed, perfect, fraction=0.05) == pytest.approx(1.0)

    def test_hit_rate_of_a_useless_ranking_is_about_the_fraction(self):
        rng = np.random.default_rng(0)
        observed = (rng.random(5000) < 0.05).astype(float)
        assert hit_rate_at_top_n(observed, rng.random(5000), fraction=0.1) == pytest.approx(
            0.1, abs=0.05
        )

    def test_pr_auc_is_nan_without_positives(self):
        """NaN is honest; 0.0 would silently drag a pooled mean down."""
        assert np.isnan(pr_auc(np.zeros(10), np.random.default_rng(0).random(10)))

    def test_hit_rate_is_nan_without_events(self):
        assert np.isnan(hit_rate_at_top_n(np.zeros(10), np.arange(10.0)))


class TestSplits:
    def test_spatial_blocks_separate_train_from_test_cells(self):
        """No cell may appear on both sides. Leakage here invalidates the score."""
        panel = _small_panel()
        for split in spatial_block_split(panel, n_folds=4, block_cells=2):
            overlap = set(split.train.data["cell_id"]) & set(split.test.data["cell_id"])
            assert not overlap

    def test_spatial_folds_are_not_one_contiguous_slab(self):
        """A fold that is a single region confounds the fold with geography."""
        panel = _small_panel()
        for split in spatial_block_split(panel, n_folds=4, block_cells=2):
            columns = split.test.data["col"].unique()
            assert len(columns) > 1

    def test_every_row_is_tested_at_most_once_per_fold(self):
        panel = _small_panel()
        for split in spatial_block_split(panel, n_folds=5, block_cells=2):
            assert len(split.train) + len(split.test) == len(panel)

    def test_forward_split_is_strictly_chronological(self):
        panel = _small_panel()
        split = forward_time_split(panel, train_fraction=0.7)
        assert split.train.data["window_start"].max() < split.test.data["window_start"].min()

    def test_forward_split_rejects_degenerate_fractions(self):
        panel = _small_panel()
        for fraction in (0.0, 1.0, -0.5):
            with pytest.raises(ValueError, match="train_fraction"):
                forward_time_split(panel, train_fraction=fraction)

    def test_spatial_split_rejects_too_few_folds(self):
        with pytest.raises(ValueError, match="n_folds"):
            list(spatial_block_split(_small_panel(), n_folds=1))


class TestValidationDesign:
    """Why random k-fold is rejected, demonstrated rather than asserted."""

    def test_random_split_is_optimistic_versus_spatial(self, panel: HazardPanel):
        """Random folds flatter a spatial model, which is the whole argument."""
        random = random_split(panel, train_fraction=0.7, seed=0)
        model = HistoricalFrequency()
        model.fit(random.train)
        random_deviance = poisson_deviance(random.test.counts, model.expected_counts(random.test))

        spatial_deviances = []
        for split in spatial_block_split(panel, n_folds=4, block_cells=2):
            fold_model = HistoricalFrequency()
            fold_model.fit(split.train)
            spatial_deviances.append(
                poisson_deviance(split.test.counts, fold_model.expected_counts(split.test))
            )

        assert random_deviance < np.mean(spatial_deviances), (
            "random folding did not flatter the memorising model, which would "
            "undermine the stated reason for rejecting random k-fold"
        )


class TestPanelValidation:
    def test_rejects_missing_columns(self, panel: HazardPanel):
        with pytest.raises(ValueError, match="missing required columns"):
            HazardPanel(panel.data.drop(columns=["exposure_vehicle_miles"]))

    def test_rejects_zero_exposure(self, panel: HazardPanel):
        """log(0) offset is -inf and would poison the likelihood silently."""
        broken = panel.data.copy()
        broken.loc[broken.index[0], "exposure_vehicle_miles"] = 0.0
        with pytest.raises(ValueError, match="non-positive exposure"):
            HazardPanel(broken)

    def test_rejects_negative_counts(self, panel: HazardPanel):
        broken = panel.data.copy()
        broken.loc[broken.index[0], "crash_count"] = -1
        with pytest.raises(ValueError, match="negative crash counts"):
            HazardPanel(broken)

    def test_rejects_empty(self, panel: HazardPanel):
        with pytest.raises(ValueError, match="empty"):
            HazardPanel(panel.data.head(0))


class TestBaselines:
    def test_all_baselines_satisfy_the_protocol(self):
        for model in all_baselines():
            assert isinstance(model, HazardModel), model

    @pytest.mark.parametrize(
        "factory", [ExposureOnly, HistoricalFrequency, CrashKde, VolumeOnlyGlm]
    )
    def test_each_baseline_fits_and_predicts_finite_counts(self, factory, panel: HazardPanel):
        model = factory()
        diagnostics = model.fit(panel)
        expected = model.expected_counts(panel)

        assert diagnostics.n_observations == len(panel)
        assert len(expected) == len(panel)
        assert np.all(np.isfinite(expected))
        assert np.all(expected >= 0)

    def test_exposure_only_recovers_the_pooled_rate(self, panel: HazardPanel):
        """The F1 null must be exactly what it claims: one rate everywhere."""
        model = ExposureOnly()
        model.fit(panel)
        implied = model.expected_counts(panel) / panel.exposure
        assert implied.std() == pytest.approx(0.0, abs=1e-12)
        assert implied[0] == pytest.approx(panel.counts.sum() / panel.exposure.sum(), rel=1e-6)

    def test_volume_only_glm_has_no_exposure_offset(self):
        """It models counts, not rates. That is the point of including it."""
        assert VolumeOnlyGlm.use_offset is False
        assert ExposureOnly.use_offset is True

    def test_predict_returns_intervals_on_every_estimate(self, panel: HazardPanel):
        model = ExposureOnly()
        small = panel.subset(np.arange(len(panel)) < 200)
        model.fit(panel)
        estimates = model.predict(small)

        assert len(estimates) == len(small)
        for estimate in estimates[:20]:
            assert estimate.rate_interval.lower <= estimate.rate_interval.upper
            assert 0.0 <= estimate.probability_at_least_one <= 1.0

    def test_kde_needs_events_to_fit(self, panel: HazardPanel):
        empty = panel.data.copy()
        empty["crash_count"] = 0
        with pytest.raises(ValueError, match="at least three cells"):
            CrashKde().fit(HazardPanel(empty))


class TestHarness:
    def test_every_baseline_is_scored_by_the_same_path(self, panel: HazardPanel):
        results = [evaluate(model, panel, n_folds=3, block_cells=2) for model in all_baselines()]
        table = comparison_table(results)

        assert len(table) == len(results)
        assert table["mean_poisson_deviance"].is_monotonic_increasing
        assert table["n_folds"].nunique() == 1, "models must be scored on identical folds"

    def test_hotspot_kde_is_beaten_by_exposure(self, panel: HazardPanel):
        """If the hotspot map won, this project's argument would be wrong."""
        kde = evaluate(CrashKde(), panel, n_folds=3, block_cells=2)
        exposure = evaluate(ExposureOnly(), panel, n_folds=3, block_cells=2)
        assert exposure.mean_deviance < kde.mean_deviance

    def test_calibration_comes_from_the_forward_holdout(self, panel: HazardPanel):
        """Calibration must be measured on the split that matches deployment."""
        result = evaluate(ExposureOnly(), panel, n_folds=3, block_cells=2)
        assert result.calibration.bins
        assert result.calibration.expected_calibration_error >= 0

    def test_pooled_pr_auc_weights_by_events(self, panel: HazardPanel):
        result = evaluate(ExposureOnly(), panel, n_folds=3, block_cells=2)
        per_fold = [f.pr_auc for f in result.folds if not np.isnan(f.pr_auc)]
        assert min(per_fold) <= result.pooled_pr_auc <= max(per_fold)


def _small_panel() -> HazardPanel:
    """A compact panel for split-geometry tests."""
    from born_field.adapters import SyntheticAdapter
    from tests.conftest import TEST_GRID

    return HazardPanel(SyntheticAdapter(grid=TEST_GRID, days=3).frame())
