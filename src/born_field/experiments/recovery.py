"""The alpha-recovery experiment: the project's flagship result."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from born_field.adapters.synthetic import SyntheticAdapter
from born_field.config import CorruptionConfig, GeneratorConfig, GridConfig
from born_field.evaluation.splits import forward_time_split
from born_field.logging import get_logger
from born_field.models.base import HazardPanel
from born_field.models.glm import PoissonHazardGlm
from born_field.types import RoadClass

log = get_logger(__name__)

DEFAULT_GRID = GridConfig(bbox=(-122.35, 37.75, -122.15, 37.90), cell_size_m=2000.0)
DEFAULT_DAYS = 120


@dataclass(frozen=True)
class Regime:
    """One misspecification condition."""

    name: str
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    negative_binomial: bool = False
    dropped_classes: tuple[RoadClass, ...] = ()
    description: str = ""


def default_regimes() -> list[Regime]:
    """The sweep reported in the README."""
    return [
        Regime(
            name="clean",
            description="Control: no corruption. Recovery is guaranteed a priori.",
        ),
        Regime(
            name="measurement_error",
            corruption=CorruptionConfig(flow_lognormal_noise_sd=0.30),
            description="30% lognormal error on observed flow (detector noise).",
        ),
        Regime(
            name="measurement_error_severe",
            corruption=CorruptionConfig(flow_lognormal_noise_sd=0.60),
            description="60% lognormal error on observed flow.",
        ),
        Regime(
            name="underreporting",
            corruption=CorruptionConfig(underreport_floor=0.40),
            description="Reporting probability varies 0.4-1.0 across space.",
        ),
        Regime(
            name="aggregation",
            corruption=CorruptionConfig(aggregation_factor=2),
            description="Fit at 2x the generating cell size (MAUP).",
        ),
        Regime(
            name="omitted_covariate",
            dropped_classes=(RoadClass.RESIDENTIAL, RoadClass.SECONDARY),
            description="Two road-class strata unavailable to the model.",
        ),
        Regime(
            name="overdispersion",
            corruption=CorruptionConfig(overdispersion=1.0),
            description="Gamma-Poisson counts, fitted as Poisson (overconfident).",
        ),
        Regime(
            name="overdispersion_negbin",
            corruption=CorruptionConfig(overdispersion=1.0),
            negative_binomial=True,
            description="Same data, fitted as negative binomial (the correct fix).",
        ),
    ]


def aggregate_panel(panel: HazardPanel, factor: int) -> HazardPanel:
    """Collapse the panel onto cells ``factor`` times larger on a side."""
    if factor < 1:
        msg = f"aggregation factor must be >= 1, got {factor}"
        raise ValueError(msg)
    if factor == 1:
        return panel

    data = panel.data.copy()
    data["row"] = data["row"] // factor
    data["col"] = data["col"] // factor
    data["cell_id"] = [f"r{r:04d}c{c:04d}" for r, c in zip(data["row"], data["col"], strict=True)]

    grouped = data.groupby(
        ["cell_id", "row", "col", "road_class", "window_start"], as_index=False
    ).agg(
        crash_count=("crash_count", "sum"),
        exposure_vehicle_miles=("exposure_vehicle_miles", "sum"),
        road_miles=("road_miles", "sum"),
        is_wet=("is_wet", "first"),
        is_night=("is_night", "first"),
    )
    grouped["flow_veh_per_hour"] = grouped["exposure_vehicle_miles"] / grouped["road_miles"].clip(
        lower=1e-12
    )
    return HazardPanel(grouped)


def calibrate_k(
    true_alpha: float,
    target_events: float,
    grid: GridConfig = DEFAULT_GRID,
    days: int = DEFAULT_DAYS,
    seed: int = 0,
) -> float:
    """Choose ``true_k`` so the expected event count hits ``target_events``."""
    probe_k = 1e-7
    probe = SyntheticAdapter(
        grid=grid,
        generator=GeneratorConfig(true_alpha=true_alpha, true_k=probe_k, seed=seed),
        days=days,
    )
    expected_at_probe = float(probe.frame()["true_rate"].sum())
    if expected_at_probe <= 0:
        msg = "probe run produced zero expected events; check the grid configuration"
        raise ValueError(msg)
    return probe_k * target_events / expected_at_probe


@dataclass(frozen=True)
class RecoveryRun:
    """One fit: a single (regime, seed) cell of the experiment."""

    regime: str
    seed: int
    true_alpha: float
    recovered_alpha: float
    ci_lower: float
    ci_upper: float
    n_events: int
    n_observations: int
    converged: bool
    dispersion: float
    holdout_deviance: float
    exposure_null_rejected: bool

    @property
    def covers_truth(self) -> bool:
        """Whether the interval contains the true alpha."""
        return self.ci_lower <= self.true_alpha <= self.ci_upper

    @property
    def bias(self) -> float:
        """Signed error in the point estimate."""
        return self.recovered_alpha - self.true_alpha

    @property
    def relative_bias(self) -> float:
        """Bias as a fraction of the truth, for the F5 threshold."""
        return self.bias / self.true_alpha


def run_one(
    regime: Regime,
    seed: int,
    true_alpha: float,
    true_k: float,
    grid: GridConfig = DEFAULT_GRID,
    days: int = DEFAULT_DAYS,
) -> RecoveryRun:
    """Generate one world under ``regime`` and try to recover alpha from it."""
    adapter = SyntheticAdapter(
        grid=grid,
        generator=GeneratorConfig(true_alpha=true_alpha, true_k=true_k, seed=seed),
        corruption=regime.corruption,
        days=days,
    )
    panel = HazardPanel(adapter.frame())
    panel = aggregate_panel(panel, regime.corruption.aggregation_factor)

    model = PoissonHazardGlm(
        negative_binomial=regime.negative_binomial,
        dropped_classes=regime.dropped_classes,
    )
    diagnostics = model.fit(panel)
    interval = model.alpha_interval()

    # Held-out deviance on the forward split, so the comparison shows what
    # misspecification costs in prediction as well as in parameter bias.
    split = forward_time_split(panel, train_fraction=0.7)
    holdout_model = PoissonHazardGlm(
        negative_binomial=regime.negative_binomial,
        dropped_classes=regime.dropped_classes,
    )
    holdout_model.fit(split.train)
    holdout_deviance = holdout_model.deviance_on(split.test)

    return RecoveryRun(
        regime=regime.name,
        seed=seed,
        true_alpha=true_alpha,
        recovered_alpha=model.alpha,
        ci_lower=interval.lower,
        ci_upper=interval.upper,
        n_events=panel.n_events,
        n_observations=len(panel),
        converged=diagnostics.converged,
        dispersion=diagnostics.dispersion,
        holdout_deviance=holdout_deviance,
        exposure_null_rejected=model.exposure_null_rejected,
    )


def sweep(
    regimes: Sequence[Regime],
    seeds: Sequence[int],
    true_alpha: float = 1.35,
    target_events: float = 5000.0,
    grid: GridConfig = DEFAULT_GRID,
    days: int = DEFAULT_DAYS,
) -> Iterator[RecoveryRun]:
    """Run every ``(regime, seed)`` cell, yielding results as they complete."""
    true_k = calibrate_k(true_alpha, target_events, grid=grid, days=days)
    log.info("calibrated_k", true_alpha=true_alpha, true_k=true_k, target_events=target_events)

    for regime in regimes:
        for seed in seeds:
            run = run_one(regime, seed, true_alpha, true_k, grid=grid, days=days)
            log.info(
                "recovery_run",
                regime=run.regime,
                seed=seed,
                recovered=round(run.recovered_alpha, 4),
                covers=run.covers_truth,
                events=run.n_events,
            )
            yield run


def alpha_sweep(
    alphas: Sequence[float],
    seeds: Sequence[int],
    target_events: float = 5000.0,
    grid: GridConfig = DEFAULT_GRID,
    days: int = DEFAULT_DAYS,
) -> Iterator[RecoveryRun]:
    """Recover alpha across a range of true values, at constant event count."""
    clean = Regime(name="clean")
    for alpha in alphas:
        true_k = calibrate_k(alpha, target_events, grid=grid, days=days)
        for seed in seeds:
            run = run_one(clean, seed, alpha, true_k, grid=grid, days=days)
            log.info(
                "alpha_sweep_run",
                true_alpha=alpha,
                seed=seed,
                recovered=round(run.recovered_alpha, 4),
                covers=run.covers_truth,
            )
            yield run


def summarise(runs: Sequence[RecoveryRun]) -> pd.DataFrame:
    """Aggregate runs into the table reported in the README."""
    frame = pd.DataFrame(
        [r.__dict__ | {"covers_truth": r.covers_truth, "bias": r.bias} for r in runs]
    )
    return (
        frame.groupby("regime")
        .agg(
            n_runs=("seed", "count"),
            true_alpha=("true_alpha", "first"),
            mean_recovered_alpha=("recovered_alpha", "mean"),
            sd_recovered_alpha=("recovered_alpha", "std"),
            mean_bias=("bias", "mean"),
            coverage=("covers_truth", "mean"),
            mean_events=("n_events", "mean"),
            mean_holdout_deviance=("holdout_deviance", "mean"),
            null_rejected_rate=("exposure_null_rejected", "mean"),
            converged=("converged", "all"),
        )
        .reset_index()
        .sort_values("mean_bias", key=np.abs)
        .reset_index(drop=True)
    )
