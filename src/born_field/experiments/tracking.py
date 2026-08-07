"""MLflow tracking and the flagship plot."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # No display in CI; must be set before pyplot is imported.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from born_field.experiments.recovery import RecoveryRun, summarise
from born_field.logging import get_logger

log = get_logger(__name__)


def log_to_mlflow(
    runs: Sequence[RecoveryRun],
    experiment_name: str = "alpha_recovery",
    tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    """Record every run, plus one parent run holding the summary."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    summary = summarise(runs)
    with mlflow.start_run(run_name="sweep"):
        mlflow.log_param("n_runs", len(runs))
        mlflow.log_param("regimes", ",".join(sorted({r.regime for r in runs})))
        mlflow.log_param("seeds", ",".join(str(s) for s in sorted({r.seed for r in runs})))

        for row in summary.itertuples():
            regime = str(row.regime)
            for metric in (
                "mean_recovered_alpha",
                "mean_bias",
                "coverage",
                "mean_holdout_deviance",
            ):
                mlflow.log_metric(f"{regime}/{metric}", float(cast("float", getattr(row, metric))))

        for run in runs:
            with mlflow.start_run(run_name=f"{run.regime}_seed{run.seed}", nested=True):
                mlflow.log_params(
                    {
                        "regime": run.regime,
                        "seed": run.seed,
                        "true_alpha": run.true_alpha,
                    }
                )
                mlflow.log_metrics(
                    {
                        "recovered_alpha": run.recovered_alpha,
                        "ci_lower": run.ci_lower,
                        "ci_upper": run.ci_upper,
                        "bias": run.bias,
                        "relative_bias": run.relative_bias,
                        "covers_truth": float(run.covers_truth),
                        "n_events": float(run.n_events),
                        "dispersion": run.dispersion,
                        "holdout_deviance": run.holdout_deviance,
                        "exposure_null_rejected": float(run.exposure_null_rejected),
                    }
                )


def plot_regime_recovery(runs: Sequence[RecoveryRun], output: Path) -> Path:
    """The flagship figure: recovered alpha by misspecification regime."""
    summary = summarise(runs)
    true_alpha = float(summary["true_alpha"].iloc[0])

    figure, axes = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(summary))

    axes.axhline(
        true_alpha, color="#111", linestyle="--", linewidth=1.2, label=f"true α = {true_alpha:g}"
    )
    axes.axhline(
        1.0,
        color="#b3261e",
        linestyle=":",
        linewidth=1.2,
        label="α = 1 (exposure explains everything)",
    )

    for i, row in enumerate(summary.itertuples()):
        subset = [r for r in runs if r.regime == row.regime]
        values = np.array([r.recovered_alpha for r in subset])
        axes.scatter(
            np.full(len(values), i) + np.random.default_rng(0).normal(0, 0.045, len(values)),
            values,
            s=18,
            alpha=0.45,
            color="#3b6ea5",
            zorder=3,
        )
        # Interval across seeds. A reader should judge the spread of the
        # estimator rather than one lucky fit.
        axes.errorbar(
            i,
            values.mean(),
            yerr=1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0,
            fmt="o",
            color="#12355b",
            markersize=7,
            capsize=4,
            zorder=4,
        )

    axes.set_xticks(positions)
    axes.set_xticklabels(summary["regime"], rotation=30, ha="right")
    axes.set_ylabel("recovered α")
    axes.set_title(
        "Flow-exponent recovery under misspecification\n"
        f"{len({r.seed for r in runs})} seeds per regime; clean case is the control"
    )
    axes.legend(loc="best", fontsize=9)
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    log.info("wrote_plot", path=str(output))
    return output


def plot_alpha_sweep(runs: Sequence[RecoveryRun], output: Path) -> Path:
    """Recovered alpha against true alpha, with the identity line."""
    frame = pd.DataFrame(
        [
            {
                "true_alpha": r.true_alpha,
                "recovered_alpha": r.recovered_alpha,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
            }
            for r in runs
        ]
    )
    grouped = frame.groupby("true_alpha").agg(
        mean=("recovered_alpha", "mean"),
        sd=("recovered_alpha", "std"),
        ci_lower=("ci_lower", "mean"),
        ci_upper=("ci_upper", "mean"),
    )

    figure, axes = plt.subplots(figsize=(7.5, 6))
    limits = [float(grouped.index.min()) - 0.15, float(grouped.index.max()) + 0.15]
    axes.plot(limits, limits, color="#111", linestyle="--", linewidth=1.1, label="perfect recovery")

    axes.errorbar(
        grouped.index,
        grouped["mean"],
        yerr=[
            grouped["mean"] - grouped["ci_lower"],
            grouped["ci_upper"] - grouped["mean"],
        ],
        fmt="o",
        color="#12355b",
        capsize=4,
        markersize=6,
        label="recovered (mean 95% CI)",
    )
    axes.scatter(
        frame["true_alpha"] + np.random.default_rng(1).normal(0, 0.012, len(frame)),
        frame["recovered_alpha"],
        s=14,
        alpha=0.35,
        color="#3b6ea5",
        zorder=2,
    )

    axes.axvline(2.0, color="#b3261e", linestyle=":", linewidth=1.1, label="Born exponent (2)")
    axes.set_xlabel("true α")
    axes.set_ylabel("recovered α")
    axes.set_title("Flow-exponent recovery, clean sampling\nevent count held constant across α")
    axes.legend(loc="best", fontsize=9)
    axes.grid(alpha=0.25)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    log.info("wrote_plot", path=str(output))
    return output
