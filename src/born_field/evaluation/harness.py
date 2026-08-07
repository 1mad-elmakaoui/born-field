"""The evaluation harness: one code path for every model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from born_field.evaluation.metrics import FoldMetrics, calibration_report, score_fold
from born_field.evaluation.splits import Split, forward_time_split, spatial_block_split
from born_field.logging import get_logger
from born_field.models.base import HazardModel, HazardPanel
from born_field.types import CalibrationReport, Frozen

log = get_logger(__name__)


class EvaluationResult(Frozen):
    """Everything one model's evaluation produced."""

    model_name: str
    model_version: str
    folds: Sequence[FoldMetrics]
    calibration: CalibrationReport

    @property
    def mean_deviance(self) -> float:
        """Primary headline metric: mean held-out Poisson deviance."""
        return float(np.mean([f.poisson_deviance for f in self.folds]))

    @property
    def pooled_pr_auc(self) -> float:
        """Event-weighted PR-AUC."""
        values = [(f.pr_auc, f.n_test_events) for f in self.folds if not np.isnan(f.pr_auc)]
        if not values:
            return float("nan")
        total = sum(w for _, w in values)
        return float(sum(v * w for v, w in values) / total) if total else float("nan")


def evaluate(
    model: HazardModel,
    panel: HazardPanel,
    n_folds: int = 5,
    block_cells: int = 4,
    train_fraction: float = 0.7,
) -> EvaluationResult:
    """Score a model under spatially blocked CV and a forward-in-time holdout."""
    splits: list[Split] = [
        *spatial_block_split(panel, n_folds=n_folds, block_cells=block_cells),
        forward_time_split(panel, train_fraction=train_fraction),
    ]

    folds: list[FoldMetrics] = []
    calibration: CalibrationReport | None = None

    for split in splits:
        if not split.is_informative:
            log.warning(
                "skipping_uninformative_fold",
                fold=split.name,
                model=model.name,
                test_events=split.test.n_events,
            )
            continue

        model.fit(split.train)
        expected = model.expected_counts(split.test)
        observed = split.test.counts
        folds.append(score_fold(split.name, observed, expected))

        if split.name.startswith("forward_time"):
            calibration = calibration_report(observed, expected, model.name, model.version)

    if not folds:
        msg = f"no informative folds for model {model.name}"
        raise ValueError(msg)
    if calibration is None:
        msg = "forward-in-time holdout produced no calibration report"
        raise ValueError(msg)

    return EvaluationResult(
        model_name=model.name,
        model_version=model.version,
        folds=folds,
        calibration=calibration,
    )


def comparison_table(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    """Side-by-side leaderboard, sorted by the primary metric."""
    frame = pd.DataFrame(
        [
            {
                "model": r.model_name,
                "mean_poisson_deviance": r.mean_deviance,
                "pooled_pr_auc": r.pooled_pr_auc,
                "mean_hit_rate_top_5pct": float(np.nanmean([f.hit_rate_top_5pct for f in r.folds])),
                "calibration_error": r.calibration.expected_calibration_error,
                "n_folds": len(r.folds),
            }
            for r in results
        ]
    )
    return frame.sort_values("mean_poisson_deviance").reset_index(drop=True)
