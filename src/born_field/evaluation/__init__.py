"""Baselines, splits, metrics, and the shared scoring harness."""

from born_field.evaluation.harness import EvaluationResult, comparison_table, evaluate
from born_field.evaluation.metrics import (
    FoldMetrics,
    calibration_report,
    hit_rate_at_top_n,
    poisson_deviance,
    poisson_log_likelihood,
    pr_auc,
    score_fold,
)
from born_field.evaluation.splits import (
    Split,
    forward_time_split,
    random_split,
    spatial_block_split,
)

__all__ = [
    "EvaluationResult",
    "FoldMetrics",
    "Split",
    "calibration_report",
    "comparison_table",
    "evaluate",
    "forward_time_split",
    "hit_rate_at_top_n",
    "poisson_deviance",
    "poisson_log_likelihood",
    "pr_auc",
    "random_split",
    "score_fold",
    "spatial_block_split",
]
