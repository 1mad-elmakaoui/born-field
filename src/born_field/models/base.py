"""The hazard-model extension point.

Same reasoning as the adapter Protocol: a buyer swaps the model without
touching validation or serving. The evaluation harness in Stage 2 is written
against this interface, so a baseline and the production model are scored by
identical code -- which is the only way a baseline comparison means anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from born_field.types import (
    Frozen,
    HazardEstimate,
    NonNegative,
    RoadClass,
    TimeWindow,
)


class HazardSample(Frozen):
    """One cell-window row of the fitting table.

    ``exposure_vehicle_miles`` is mandatory and separate from ``flow_veh_per_hour``
    even though one is derived from the other, because they play different roles:
    the former is the Poisson offset (coefficient fixed at 1), the latter is a
    covariate whose coefficient is the quantity of scientific interest.
    """

    cell_id: str
    window: TimeWindow
    crash_count: int
    exposure_vehicle_miles: NonNegative
    flow_veh_per_hour: NonNegative
    road_class: RoadClass
    is_wet: bool
    is_night: bool


REQUIRED_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "cell_id",
    "window_start",
    "crash_count",
    "exposure_vehicle_miles",
    "flow_veh_per_hour",
    "road_class",
    "road_miles",
    "is_wet",
    "is_night",
    "row",
    "col",
)


@dataclass(frozen=True)
class HazardPanel:
    """A validated training table: the unit models are fitted on.

    :class:`HazardSample` is the right shape for scoring one cell-window
    through the API, but the wrong shape for fitting: the recovery experiment
    fits hundreds of models over panels of ~10^6 rows, and constructing that
    many validated objects per fit would dominate its runtime while proving
    nothing extra.

    So validation moves from per-row to per-panel. The schema is checked once,
    on construction, and every model then works against columnar data. This is
    the same trade every production feature store makes, and it keeps the type
    system meaningful without paying per-row for it.
    """

    data: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate the schema once, at the boundary."""
        missing = set(REQUIRED_PANEL_COLUMNS) - set(self.data.columns)
        if missing:
            msg = f"panel is missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        if len(self.data) == 0:
            msg = "panel is empty"
            raise ValueError(msg)
        if (self.data["exposure_vehicle_miles"] <= 0).any():
            # log(exposure) is the offset; a non-positive value is -inf and
            # would silently poison the likelihood rather than raise.
            msg = "panel contains non-positive exposure, which has no log offset"
            raise ValueError(msg)
        if (self.data["crash_count"] < 0).any():
            msg = "panel contains negative crash counts"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.data)

    @property
    def counts(self) -> npt.NDArray[np.float64]:
        """Observed crash counts."""
        return self.data["crash_count"].to_numpy(dtype=np.float64)

    @property
    def exposure(self) -> npt.NDArray[np.float64]:
        """Vehicle-miles travelled: the Poisson offset, before the log."""
        return self.data["exposure_vehicle_miles"].to_numpy(dtype=np.float64)

    @property
    def n_events(self) -> int:
        """Total observed crashes; the real sample size for a rare-event model."""
        return int(self.data["crash_count"].sum())

    def subset(self, mask: npt.NDArray[np.bool_]) -> HazardPanel:
        """A panel restricted to ``mask``, re-validated."""
        return HazardPanel(self.data.loc[mask].reset_index(drop=True))

    @classmethod
    def from_samples(cls, samples: Sequence[HazardSample]) -> HazardPanel:
        """Build a panel from validated rows, for callers holding domain objects."""
        return cls(
            pd.DataFrame(
                [
                    {
                        "cell_id": s.cell_id,
                        "window_start": s.window.start,
                        "crash_count": s.crash_count,
                        "exposure_vehicle_miles": s.exposure_vehicle_miles,
                        "flow_veh_per_hour": s.flow_veh_per_hour,
                        "road_class": s.road_class.value,
                        "road_miles": 0.0,
                        "is_wet": s.is_wet,
                        "is_night": s.is_night,
                        "row": 0,
                        "col": 0,
                    }
                    for s in samples
                ]
            )
        )


class FitDiagnostics(Frozen):
    """What a fit reports about itself, for MLflow and for refusing to serve.

    ``converged`` is not cosmetic: a non-converged GLM will still happily emit
    predictions, and serving from one is worse than refusing.
    """

    converged: bool
    n_observations: int
    n_events: int
    log_likelihood: float
    deviance: float
    dispersion: float
    params: Mapping[str, float]
    param_std_errors: Mapping[str, float]

    @property
    def events_per_observation(self) -> float:
        """Event density. Very low values make Poisson intervals unreliable."""
        return self.n_events / self.n_observations if self.n_observations else 0.0


@runtime_checkable
class HazardModel(Protocol):
    """Fits a collision-hazard model and emits calibrated estimates.

    Implementations must honour two invariants:

    1. **Exposure is an offset, not a feature.** ``log(exposure_vehicle_miles)``
       enters with its coefficient fixed at 1. A model that estimates a free
       coefficient on exposure is modelling counts, not risk, and will rank
       busy roads as dangerous roads.
    2. **Every estimate carries an interval.** Point estimates without
       uncertainty cannot be priced against and must not be returned.
    """

    @property
    def name(self) -> str:
        """Model identifier, recorded in the registry."""
        ...

    @property
    def version(self) -> str:
        """Model version, recorded in the registry."""
        ...

    def fit(self, panel: HazardPanel) -> FitDiagnostics:
        """Fit to a training panel and return convergence diagnostics."""
        ...

    def predict(self, panel: HazardPanel) -> Sequence[HazardEstimate]:
        """Emit calibrated estimates with intervals for each input row."""
        ...

    def expected_counts(self, panel: HazardPanel) -> npt.NDArray[np.float64]:
        """Expected crash counts, for deviance and calibration scoring.

        Kept separate from :meth:`predict` so the evaluation harness can score
        the mean function without paying for interval construction on every row.
        Every baseline and every candidate model is scored through this one
        method, which is what makes a baseline comparison mean anything.
        """
        ...
