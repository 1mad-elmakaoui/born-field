"""Validation splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from born_field.models.base import HazardPanel


@dataclass(frozen=True)
class Split:
    """One train/test partition of a panel."""

    name: str
    train: HazardPanel
    test: HazardPanel

    @property
    def is_informative(self) -> bool:
        """Whether the test fold contains enough events to score."""
        return self.test.n_events > 0 and self.train.n_events > 0


def spatial_block_split(
    panel: HazardPanel,
    n_folds: int = 5,
    block_cells: int = 4,
) -> Iterator[Split]:
    """Spatially blocked cross-validation."""
    if n_folds < 2:
        msg = f"n_folds must be at least 2, got {n_folds}"
        raise ValueError(msg)
    if block_cells < 1:
        msg = f"block_cells must be at least 1, got {block_cells}"
        raise ValueError(msg)

    block_row = panel.data["row"].to_numpy() // block_cells
    block_col = panel.data["col"].to_numpy() // block_cells
    fold_of_row = (block_row + block_col * 2) % n_folds

    for fold in range(n_folds):
        test_mask = fold_of_row == fold
        if not test_mask.any() or test_mask.all():
            continue
        yield Split(
            name=f"spatial_block_{fold}",
            train=panel.subset(~test_mask),
            test=panel.subset(test_mask),
        )


def forward_time_split(
    panel: HazardPanel,
    train_fraction: float = 0.7,
) -> Split:
    """Train on the past, score the future."""
    if not 0.0 < train_fraction < 1.0:
        msg = f"train_fraction must lie in (0, 1), got {train_fraction}"
        raise ValueError(msg)

    starts = panel.data["window_start"]
    ordered = np.sort(starts.unique())
    cutoff = ordered[int(len(ordered) * train_fraction)]

    train_mask: npt.NDArray[np.bool_] = (starts < cutoff).to_numpy()
    return Split(
        name=f"forward_time_{train_fraction:g}",
        train=panel.subset(train_mask),
        test=panel.subset(~train_mask),
    )


def random_split(panel: HazardPanel, train_fraction: float = 0.7, seed: int = 0) -> Split:
    """A deliberately invalid random split, provided only for contrast."""
    rng = np.random.default_rng(seed)
    mask: npt.NDArray[np.bool_] = rng.random(len(panel)) < train_fraction
    return Split(
        name="random_INVALID",
        train=panel.subset(mask),
        test=panel.subset(~mask),
    )
