"""The primary hazard model: Poisson/negative-binomial GLM with an exposure offset."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm

from born_field.evaluation.metrics import poisson_deviance
from born_field.models.base import FitDiagnostics, HazardPanel
from born_field.types import HazardEstimate, Interval, RoadClass, TimeWindow

_EPSILON = 1e-12
_Z95 = 1.959963984540054

REFERENCE_CLASS = RoadClass.PRIMARY
"""Dropped dummy level. Class effects are read relative to primary roads."""


class PoissonHazardGlm:
    """Poisson (or negative-binomial) hazard model on vehicle-mile exposure."""

    version = "1"

    def __init__(
        self,
        *,
        negative_binomial: bool = False,
        dropped_classes: Sequence[RoadClass] = (),
    ) -> None:
        """Configure the model family and design."""
        self.negative_binomial = negative_binomial
        self.dropped_classes = tuple(dropped_classes)
        # statsmodels ships no type information, so its results wrapper is Any.
        # The untyped boundary stops here: everything this class returns is typed.
        self._result: Any = None
        self._columns: list[str] = []

    @property
    def name(self) -> str:
        """Model identifier, reflecting the family actually fitted."""
        return "hazard_negbin_glm" if self.negative_binomial else "hazard_poisson_glm"

    # -- design -------------------------------------------------------------

    def _active_classes(self) -> list[RoadClass]:
        """Dummy levels present in the design, excluding the reference."""
        return [c for c in RoadClass if c is not REFERENCE_CLASS and c not in self.dropped_classes]

    def _design(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        data = panel.data
        columns: list[npt.NDArray[np.float64]] = [np.ones(len(data))]
        names = ["intercept"]

        log_flow = np.log(np.maximum(data["flow_veh_per_hour"].to_numpy(dtype=float), _EPSILON))
        columns.append(log_flow)
        names.append("log_flow")

        road_class = data["road_class"].to_numpy()
        for level in self._active_classes():
            columns.append((road_class == level.value).astype(float))
            names.append(f"class_{level.value}")

        columns.append(data["is_night"].to_numpy(dtype=float))
        names.append("is_night")
        columns.append(data["is_wet"].to_numpy(dtype=float))
        names.append("is_wet")

        self._columns = names
        return np.column_stack(columns)

    @staticmethod
    def _offset(panel: HazardPanel) -> npt.NDArray[np.float64]:
        """``log(VMT)``. The coefficient on this is fixed at 1, never fitted."""
        return np.log(np.maximum(panel.exposure, _EPSILON))

    # -- fitting ------------------------------------------------------------

    def fit(self, panel: HazardPanel) -> FitDiagnostics:
        """Fit by IRLS and return diagnostics including convergence."""
        design = self._design(panel)
        offset = self._offset(panel)

        if self.negative_binomial:
            # statsmodels defaults the NB dispersion to 1.0 instead of fitting it,
            # so estimate it by method of moments from a first Poisson pass.
            family = sm.families.NegativeBinomial(
                alpha=self._estimate_dispersion(design, offset, panel)
            )
        else:
            family = sm.families.Poisson()

        result = sm.GLM(panel.counts, design, family=family, offset=offset).fit()
        self._result = result

        return FitDiagnostics(
            converged=bool(result.converged),
            n_observations=len(panel),
            n_events=panel.n_events,
            log_likelihood=float(result.llf),
            deviance=float(result.deviance),
            dispersion=float(result.pearson_chi2 / result.df_resid),
            params=dict(zip(self._columns, map(float, result.params), strict=True)),
            param_std_errors=dict(zip(self._columns, map(float, result.bse), strict=True)),
        )

    @staticmethod
    def _estimate_dispersion(
        design: npt.NDArray[np.float64],
        offset: npt.NDArray[np.float64],
        panel: HazardPanel,
    ) -> float:
        """Method-of-moments negative-binomial dispersion from a Poisson pass."""
        counts = panel.counts
        poisson_fit = sm.GLM(counts, design, family=sm.families.Poisson(), offset=offset).fit()
        mu = np.asarray(poisson_fit.fittedvalues, dtype=np.float64)
        excess = (counts - mu) ** 2 - mu
        estimate = float(np.sum(excess * mu**2) / np.sum(mu**4))
        return max(estimate, 1e-6)

    # -- the headline quantity ----------------------------------------------

    @property
    def alpha(self) -> float:
        """Recovered flow exponent: ``1 + coefficient on log(flow)``."""
        return 1.0 + self._coefficient("log_flow")

    def alpha_interval(self, level: float = 0.95) -> Interval:
        """Confidence interval for alpha, from the analytic standard error."""
        z = _Z95 if level == 0.95 else float(abs(np.round(level, 6)))
        estimate = self._coefficient("log_flow")
        se = self._std_error("log_flow")
        return Interval(lower=1.0 + estimate - z * se, upper=1.0 + estimate + z * se, level=level)

    @property
    def exposure_null_rejected(self) -> bool:
        """Whether the 95% interval for alpha excludes 1."""
        interval = self.alpha_interval()
        return not (interval.lower <= 1.0 <= interval.upper)

    def class_multipliers(self) -> dict[RoadClass, float]:
        """Fitted risk multipliers relative to the reference class."""
        multipliers = {REFERENCE_CLASS: 1.0}
        for level in self._active_classes():
            multipliers[level] = float(np.exp(self._coefficient(f"class_{level.value}")))
        return multipliers

    def _coefficient(self, name: str) -> float:
        result = self._require_fitted()
        return float(result.params[self._columns.index(name)])

    def _std_error(self, name: str) -> float:
        result = self._require_fitted()
        return float(result.bse[self._columns.index(name)])

    def _require_fitted(self) -> Any:  # noqa: ANN401 - untyped statsmodels result
        if self._result is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        return self._result

    # -- prediction ---------------------------------------------------------

    def expected_counts(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        """Predicted mean counts on the panel's own exposure."""
        result = self._require_fitted()
        predicted = result.predict(self._design(panel), offset=self._offset(panel))
        return np.asarray(predicted, dtype=np.float64)

    def _linear_predictor_std_error(
        self, design: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Standard error of the linear predictor, row by row."""
        result = self._require_fitted()
        covariance = np.asarray(result.cov_params(), dtype=np.float64)
        variance = np.einsum("ij,jk,ik->i", design, covariance, design)
        errors: npt.NDArray[np.float64] = np.sqrt(np.maximum(variance, 0.0))
        return errors

    def predict(self, panel: HazardPanel) -> Sequence[HazardEstimate]:
        """Calibrated estimates with intervals propagated from the fit."""
        design = self._design(panel)
        offset = self._offset(panel)
        result = self._require_fitted()

        eta = design @ np.asarray(result.params, dtype=np.float64) + offset
        se = self._linear_predictor_std_error(design)

        expected = np.exp(eta)
        lower = np.exp(eta - _Z95 * se)
        upper = np.exp(eta + _Z95 * se)

        exposure = panel.exposure
        scale = 1e8 / np.maximum(exposure, _EPSILON)

        estimates: list[HazardEstimate] = []
        for i, row in enumerate(panel.data.itertuples()):
            start = cast("datetime", row.window_start)
            estimates.append(
                HazardEstimate(
                    cell_id=str(row.cell_id),
                    window=TimeWindow(start=start, end=start + timedelta(hours=1)),
                    expected_crashes=float(expected[i]),
                    rate_per_100m_vmt=float(expected[i] * scale[i]),
                    rate_interval=Interval(
                        lower=float(lower[i] * scale[i]), upper=float(upper[i] * scale[i])
                    ),
                    probability_at_least_one=float(-np.expm1(-expected[i])),
                    probability_interval=Interval(
                        lower=float(-np.expm1(-lower[i])), upper=float(-np.expm1(-upper[i]))
                    ),
                    exposure_vehicle_miles=float(exposure[i]),
                    model_name=self.name,
                    model_version=self.version,
                )
            )
        return estimates

    def deviance_on(self, panel: HazardPanel) -> float:
        """Held-out mean Poisson deviance."""
        return poisson_deviance(panel.counts, self.expected_counts(panel))
