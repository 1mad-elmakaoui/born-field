"""Baselines."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm
from scipy.stats import gaussian_kde

from born_field.evaluation.metrics import poisson_deviance
from born_field.models.base import FitDiagnostics, HazardPanel
from born_field.types import HazardEstimate, Interval, TimeWindow

_EPSILON = 1e-12


def _estimates_from_counts(
    panel: HazardPanel,
    expected: npt.NDArray[np.float64],
    name: str,
    version: str,
) -> list[HazardEstimate]:
    """Wrap point predictions as domain estimates with Poisson intervals."""
    lower = np.maximum(0.0, expected - 1.96 * np.sqrt(expected))
    upper = expected + 1.96 * np.sqrt(expected)
    exposure = panel.exposure
    per_100m = expected / np.maximum(exposure, _EPSILON) * 1e8

    estimates: list[HazardEstimate] = []
    for i, row in enumerate(panel.data.itertuples()):
        start = cast("datetime", row.window_start)
        window = TimeWindow(start=start, end=start + timedelta(hours=1))
        estimates.append(
            HazardEstimate(
                cell_id=str(row.cell_id),
                window=window,
                expected_crashes=float(expected[i]),
                rate_per_100m_vmt=float(per_100m[i]),
                rate_interval=Interval(
                    lower=float(lower[i] / max(exposure[i], _EPSILON) * 1e8),
                    upper=float(upper[i] / max(exposure[i], _EPSILON) * 1e8),
                ),
                probability_at_least_one=float(-np.expm1(-expected[i])),
                probability_interval=Interval(
                    lower=float(-np.expm1(-lower[i])),
                    upper=float(-np.expm1(-upper[i])),
                ),
                exposure_vehicle_miles=float(exposure[i]),
                model_name=name,
                model_version=version,
            )
        )
    return estimates


class HistoricalFrequency:
    """Per-cell crash rate per vehicle-mile, carried forward."""

    name = "baseline_historical_frequency"
    version = "1"

    def __init__(self) -> None:
        """Initialise an unfitted baseline."""
        self._rate_by_cell: dict[str, float] = {}
        self._global_rate = 0.0

    def fit(self, panel: HazardPanel) -> FitDiagnostics:
        """Compute per-cell rates per vehicle-mile."""
        grouped = panel.data.groupby("cell_id").agg(
            crashes=("crash_count", "sum"), vmt=("exposure_vehicle_miles", "sum")
        )
        self._global_rate = float(grouped["crashes"].sum() / max(grouped["vmt"].sum(), _EPSILON))
        self._rate_by_cell = cast(
            "dict[str, float]",
            (grouped["crashes"] / grouped["vmt"].clip(lower=_EPSILON)).to_dict(),
        )

        expected = self.expected_counts(panel)
        return FitDiagnostics(
            converged=True,
            n_observations=len(panel),
            n_events=panel.n_events,
            log_likelihood=float("nan"),
            deviance=poisson_deviance(panel.counts, expected),
            dispersion=float("nan"),
            params={"global_rate_per_vmt": self._global_rate},
            param_std_errors={},
        )

    def expected_counts(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        """Per-cell rate times test exposure."""
        rates = panel.data["cell_id"].map(self._rate_by_cell).fillna(self._global_rate).to_numpy()
        return np.asarray(rates * panel.exposure, dtype=np.float64)

    def predict(self, panel: HazardPanel) -> Sequence[HazardEstimate]:
        """Estimates with Poisson intervals."""
        return _estimates_from_counts(panel, self.expected_counts(panel), self.name, self.version)


class CrashKde:
    """Kernel density over past crash locations: the hotspot map."""

    name = "baseline_crash_kde"
    version = "1"

    def __init__(self, bandwidth: float | str = "scott") -> None:
        """Configure the kernel bandwidth."""
        self.bandwidth = bandwidth
        self._kde: gaussian_kde | None = None
        self._scale = 0.0
        self._train_windows = 1

    def fit(self, panel: HazardPanel) -> FitDiagnostics:
        """Fit a Gaussian KDE weighted by observed crash counts."""
        events = panel.data[panel.data["crash_count"] > 0]
        if len(events) < 3:
            msg = "KDE baseline needs at least three cells with crashes"
            raise ValueError(msg)

        points = np.vstack([events["col"].to_numpy(), events["row"].to_numpy()]).astype(float)
        # Jitter breaks the collinearity from events sharing integer grid
        # coordinates, which would make the kernel covariance singular.
        points = points + np.random.default_rng(0).normal(0, 1e-3, points.shape)

        self._kde = gaussian_kde(
            points, weights=events["crash_count"].to_numpy(dtype=float), bw_method=self.bandwidth
        )
        self._train_windows = int(panel.data["window_start"].nunique())
        self._scale = float(panel.counts.sum())

        return FitDiagnostics(
            converged=True,
            n_observations=len(panel),
            n_events=panel.n_events,
            log_likelihood=float("nan"),
            deviance=poisson_deviance(panel.counts, self.expected_counts(panel)),
            dispersion=float("nan"),
            params={"n_kernel_points": float(points.shape[1])},
            param_std_errors={},
        )

    def expected_counts(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        """Spatial density, rescaled to the test period's expected total."""
        if self._kde is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)

        # The kernel is purely spatial, so it takes one value per cell. Evaluating
        # per unique location and broadcasting back is equivalent and far cheaper.
        coordinates = panel.data[["col", "row"]].to_numpy(dtype=float)
        unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
        density_at_unique = np.asarray(self._kde(unique.T), dtype=np.float64)
        density = np.maximum(density_at_unique[inverse], _EPSILON)

        test_windows = max(int(panel.data["window_start"].nunique()), 1)
        expected_total = self._scale * test_windows / max(self._train_windows, 1)
        result: npt.NDArray[np.float64] = density / density.sum() * expected_total
        return result

    def predict(self, panel: HazardPanel) -> Sequence[HazardEstimate]:
        """Estimates with Poisson intervals."""
        return _estimates_from_counts(panel, self.expected_counts(panel), self.name, self.version)


class _GlmBaseline:
    """Shared fitting machinery for the two GLM baselines."""

    name = "baseline_glm"
    version = "1"
    use_offset = False

    def __init__(self) -> None:
        """Initialise an unfitted baseline."""
        # Untyped statsmodels results; see PoissonHazardGlm for the same note.
        self._result: Any = None

    def _design(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def _offset(self, panel: HazardPanel) -> npt.NDArray[np.float64] | None:
        if not self.use_offset:
            return None
        return np.log(np.maximum(panel.exposure, _EPSILON))

    def fit(self, panel: HazardPanel) -> FitDiagnostics:
        """Fit by iteratively reweighted least squares."""
        model = sm.GLM(
            panel.counts,
            self._design(panel),
            family=sm.families.Poisson(),
            offset=self._offset(panel),
        )
        result = model.fit()
        self._result = result

        names = self._param_names()
        return FitDiagnostics(
            converged=bool(result.converged),
            n_observations=len(panel),
            n_events=panel.n_events,
            log_likelihood=float(result.llf),
            deviance=float(result.deviance),
            dispersion=float(result.pearson_chi2 / result.df_resid),
            params=dict(zip(names, map(float, result.params), strict=True)),
            param_std_errors=dict(zip(names, map(float, result.bse), strict=True)),
        )

    def _param_names(self) -> list[str]:
        raise NotImplementedError

    def expected_counts(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        """Predicted mean counts."""
        if self._result is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        predicted = self._result.predict(self._design(panel), offset=self._offset(panel))
        return np.asarray(predicted, dtype=np.float64)

    def predict(self, panel: HazardPanel) -> Sequence[HazardEstimate]:
        """Estimates with Poisson intervals."""
        return _estimates_from_counts(panel, self.expected_counts(panel), self.name, self.version)


class VolumeOnlyGlm(_GlmBaseline):
    """Poisson GLM on log traffic volume, with **no exposure offset**."""

    name = "baseline_volume_only_glm"
    version = "1"
    use_offset = False

    def _design(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        log_flow = np.log(np.maximum(panel.data["flow_veh_per_hour"].to_numpy(), _EPSILON))
        return np.asarray(sm.add_constant(log_flow, has_constant="add"), dtype=np.float64)

    def _param_names(self) -> list[str]:
        return ["intercept", "log_flow"]


class ExposureOnly(_GlmBaseline):
    """Intercept plus ``log(exposure)`` offset: constant risk per vehicle-mile."""

    name = "baseline_exposure_only"
    version = "1"
    use_offset = True

    def _design(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        return np.ones((len(panel), 1), dtype=np.float64)

    def _param_names(self) -> list[str]:
        return ["intercept"]


def all_baselines() -> list[HistoricalFrequency | CrashKde | VolumeOnlyGlm | ExposureOnly]:
    """Every baseline, in the order they are reported."""
    return [ExposureOnly(), HistoricalFrequency(), CrashKde(), VolumeOnlyGlm()]
