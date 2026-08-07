"""Scoring metrics, with a deliberate hierarchy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score

from born_field.types import (
    CalibrationBin,
    CalibrationReport,
    Frozen,
    NonNegative,
    Probability,
)

_EPSILON = 1e-12


def poisson_deviance(
    observed: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
) -> float:
    """Mean Poisson deviance."""
    expected = np.maximum(expected, _EPSILON)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(observed > 0, observed * np.log(observed / expected), 0.0)
    return float(2.0 * np.mean(term - (observed - expected)))


def poisson_log_likelihood(
    observed: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
) -> float:
    """Mean Poisson log-likelihood, dropping the constant ``log(y!)`` term."""
    expected = np.maximum(expected, _EPSILON)
    return float(np.mean(observed * np.log(expected) - expected))


def pr_auc(
    observed: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
) -> float:
    """Average precision for "at least one crash"."""
    labels = (observed > 0).astype(int)
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, expected))


def hit_rate_at_top_n(
    observed: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
    fraction: float = 0.05,
) -> float:
    """Share of all crashes falling in the top ``fraction`` of ranked rows."""
    if not 0.0 < fraction <= 1.0:
        msg = f"fraction must lie in (0, 1], got {fraction}"
        raise ValueError(msg)
    total = observed.sum()
    if total == 0:
        return float("nan")

    n_top = max(1, int(len(expected) * fraction))
    top = np.argsort(expected)[::-1][:n_top]
    return float(observed[top].sum() / total)


class FoldMetrics(Frozen):
    """Scores for one train/test split."""

    fold: str
    n_test_rows: int
    n_test_events: int
    poisson_deviance: NonNegative
    log_likelihood: float
    pr_auc: float
    hit_rate_top_5pct: float


def score_fold(
    fold: str,
    observed: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
) -> FoldMetrics:
    """Score one fold under every metric."""
    return FoldMetrics(
        fold=fold,
        n_test_rows=len(observed),
        n_test_events=int(observed.sum()),
        poisson_deviance=poisson_deviance(observed, expected),
        log_likelihood=poisson_log_likelihood(observed, expected),
        pr_auc=pr_auc(observed, expected),
        hit_rate_top_5pct=hit_rate_at_top_n(observed, expected),
    )


def calibration_report(
    observed: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
    model_name: str,
    model_version: str,
    n_bins: int = 10,
) -> CalibrationReport:
    """Reliability of the predicted rate, binned by predicted value."""
    order = np.argsort(expected)
    observed, expected = observed[order], expected[order]
    bins = np.array_split(np.arange(len(expected)), n_bins)

    reported: list[CalibrationBin] = []
    weighted_error = 0.0
    within_interval = 0
    counted = 0

    for index in bins:
        if len(index) == 0:
            continue
        predicted_mean = float(expected[index].mean())
        observed_total = float(observed[index].sum())
        n = len(index)
        observed_mean = observed_total / n

        # Normal approximation to the Poisson total, which is adequate here
        # because bins hold thousands of rows even when events are rare.
        standard_error = float(np.sqrt(max(observed_total, 1.0))) / n
        lower = max(0.0, observed_mean - 1.96 * standard_error)
        upper = observed_mean + 1.96 * standard_error

        reported.append(
            CalibrationBin(
                predicted_mean=predicted_mean,
                observed_mean=observed_mean,
                observed_interval_lower=lower,
                observed_interval_upper=upper,
                n_observations=n,
            )
        )
        weighted_error += abs(predicted_mean - observed_mean) * n
        counted += n
        if lower <= predicted_mean <= upper:
            within_interval += 1

    return CalibrationReport(
        model_name=model_name,
        model_version=model_version,
        bins=reported,
        expected_calibration_error=weighted_error / counted if counted else 0.0,
        coverage_of_nominal_95=Probability(within_interval / len(reported)) if reported else 0.0,
    )
