"""The scoring service: where refusal is enforced."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast

import geopandas as gpd
import pandas as pd

from born_field.adapters.base import Coverage
from born_field.logging import get_logger
from born_field.models.base import HazardPanel
from born_field.models.glm import PoissonHazardGlm
from born_field.types import (
    STORAGE_CRS,
    WORKING_CRS,
    CalibrationReport,
    HazardEstimate,
    Refusal,
    RefusalReason,
    RoadClass,
    ScoringResult,
    TimeWindow,
    vehicle_miles_travelled,
)

log = get_logger(__name__)

MIN_TRAINING_ROWS_PER_CELL = 24
"""Below this, a cell's contribution to the fit is too thin to score from.

One day of hourly observations. The threshold is deliberately explicit and
configurable rather than implicit in the data: "we do not have enough data here"
is a product decision, not a statistical accident.
"""

FLOW_SUPPORT_QUANTILES = (0.001, 0.999)
"""Fitted flow range. Outside it the model extrapolates a power law, which is
exactly where a power law is least trustworthy."""


@dataclass(frozen=True)
class CellContext:
    """Everything known about one cell, needed to score a query against it."""

    cell_id: str
    road_class: RoadClass
    road_miles: float
    typical_flow: float
    training_rows: int


class RiskService:
    """Scores point-in-time queries against a fitted hazard model."""

    def __init__(
        self,
        model: PoissonHazardGlm,
        panel: HazardPanel,
        cells: gpd.GeoDataFrame,
        coverage: Coverage,
    ) -> None:
        """Precompute the support checks so scoring is a lookup, not a scan."""
        self.model = model
        self.coverage = coverage
        self._cells = cells.set_index("cell_id")

        data = panel.data
        self._flow_low, self._flow_high = (
            float(data["flow_veh_per_hour"].quantile(FLOW_SUPPORT_QUANTILES[0])),
            float(data["flow_veh_per_hour"].quantile(FLOW_SUPPORT_QUANTILES[1])),
        )

        grouped = data.groupby(["cell_id", "road_class"], as_index=False).agg(
            road_miles=("road_miles", "first"),
            typical_flow=("flow_veh_per_hour", "median"),
            training_rows=("crash_count", "size"),
        )
        # One context per cell: the stratum carrying the most road mileage,
        # which is the one a point query most likely lands on.
        best = grouped.sort_values("road_miles", ascending=False).drop_duplicates("cell_id")
        self._context: dict[str, CellContext] = {
            str(row.cell_id): CellContext(
                cell_id=str(row.cell_id),
                road_class=RoadClass(str(row.road_class)),
                road_miles=float(cast("float", row.road_miles)),
                typical_flow=float(cast("float", row.typical_flow)),
                training_rows=int(cast("int", row.training_rows)),
            )
            for row in best.itertuples()
        }
        self._calibration: CalibrationReport | None = None

    # -- lookup -------------------------------------------------------------

    def locate(self, lon: float, lat: float) -> str | None:
        """The cell containing a coordinate, or None if outside the grid."""
        point = gpd.GeoSeries.from_xy([lon], [lat], crs=STORAGE_CRS).to_crs(WORKING_CRS)
        hits = self._cells[self._cells.geometry.contains(point.iloc[0])]
        return str(hits.index[0]) if len(hits) else None

    # -- scoring ------------------------------------------------------------

    def score(
        self,
        lon: float,
        lat: float,
        timestamp: datetime,
        flow_veh_per_hour: float | None = None,
        is_wet: bool = False,
    ) -> ScoringResult:
        """Score one point in space and time, or refuse with a reason."""
        if timestamp.tzinfo is None:
            msg = "timestamp must be timezone-aware"
            raise ValueError(msg)

        if not self.coverage.contains_point(lon, lat):
            return Refusal(
                reason=RefusalReason.OUTSIDE_FITTED_REGION,
                detail=(
                    f"({lat:.5f}, {lon:.5f}) lies outside the fitted study area "
                    f"{self.coverage.bbox}"
                ),
            )

        window = self.coverage.window
        if not (window.start <= timestamp < window.end):
            return Refusal(
                reason=RefusalReason.OUTSIDE_FITTED_TIME_RANGE,
                detail=(
                    f"{timestamp.isoformat()} lies outside the fitted range "
                    f"{window.start.isoformat()} to {window.end.isoformat()}"
                ),
            )

        cell_id = self.locate(lon, lat)
        if cell_id is None or cell_id not in self._context:
            return Refusal(
                reason=RefusalReason.INSUFFICIENT_COVERAGE,
                detail=(
                    f"({lat:.5f}, {lon:.5f}) falls in no cell carrying road mileage; "
                    "no exposure denominator exists here"
                ),
                cell_id=cell_id,
            )

        context = self._context[cell_id]
        if context.training_rows < MIN_TRAINING_ROWS_PER_CELL:
            return Refusal(
                reason=RefusalReason.INSUFFICIENT_COVERAGE,
                detail=(
                    f"cell {cell_id} contributed only {context.training_rows} training "
                    f"observations, below the minimum of {MIN_TRAINING_ROWS_PER_CELL}"
                ),
                cell_id=cell_id,
            )

        flow = context.typical_flow if flow_veh_per_hour is None else flow_veh_per_hour
        if not self._flow_low <= flow <= self._flow_high:
            return Refusal(
                reason=RefusalReason.EXTRAPOLATION_BEYOND_SUPPORT,
                detail=(
                    f"flow {flow:.1f} veh/h lies outside the fitted support "
                    f"[{self._flow_low:.1f}, {self._flow_high:.1f}]; a power law "
                    "extrapolated beyond its support is not a risk estimate"
                ),
                cell_id=cell_id,
            )

        exposure = vehicle_miles_travelled(flow, context.road_miles, 1.0)
        if exposure <= 0:
            return Refusal(
                reason=RefusalReason.EXPOSURE_UNAVAILABLE,
                detail=f"cell {cell_id} has no vehicle-miles in this window",
                cell_id=cell_id,
            )

        query = HazardPanel(
            pd.DataFrame(
                [
                    {
                        "cell_id": cell_id,
                        "window_start": timestamp,
                        "crash_count": 0,
                        "exposure_vehicle_miles": exposure,
                        "flow_veh_per_hour": flow,
                        "road_class": context.road_class.value,
                        "road_miles": context.road_miles,
                        "is_wet": is_wet,
                        "is_night": timestamp.hour >= 20 or timestamp.hour < 6,
                        "row": 0,
                        "col": 0,
                    }
                ]
            )
        )
        return self.model.predict(query)[0]

    def score_batch(self, panel: HazardPanel) -> list[HazardEstimate]:
        """Score a whole panel. Used by the batch job, not the point endpoint."""
        return list(self.model.predict(panel))

    # -- reporting ----------------------------------------------------------

    def set_calibration(self, report: CalibrationReport) -> None:
        """Attach the calibration report served by the API."""
        self._calibration = report

    @property
    def calibration(self) -> CalibrationReport | None:
        """The attached calibration report, if any."""
        return self._calibration

    @property
    def fitted_alpha(self) -> float:
        """Recovered flow exponent, exposed for the explanation endpoint."""
        return self.model.alpha

    def support_summary(self) -> dict[str, object]:
        """What the service will and will not answer, in one object."""
        return {
            "bbox": list(self.coverage.bbox),
            "window_start": self.coverage.window.start.isoformat(),
            "window_end": self.coverage.window.end.isoformat(),
            "n_cells": len(self._context),
            "flow_support": [self._flow_low, self._flow_high],
            "min_training_rows_per_cell": MIN_TRAINING_ROWS_PER_CELL,
        }


def build_demo_service(days: int = 60) -> RiskService:
    """Fit a service on synthetic data, for the demo and for tests."""
    from born_field.adapters.synthetic import SyntheticAdapter
    from born_field.evaluation.metrics import calibration_report
    from born_field.evaluation.splits import forward_time_split
    from born_field.experiments.recovery import DEFAULT_GRID

    adapter = SyntheticAdapter(grid=DEFAULT_GRID, days=days)
    panel = HazardPanel(adapter.frame())

    model = PoissonHazardGlm()
    model.fit(panel)

    split = forward_time_split(panel, train_fraction=0.7)
    holdout = PoissonHazardGlm()
    holdout.fit(split.train)
    report = calibration_report(
        split.test.counts,
        holdout.expected_counts(split.test),
        model.name,
        model.version,
    )

    service = RiskService(model, panel, adapter.cells_frame, adapter.coverage())
    service.set_calibration(report)
    log.info(
        "demo_service_ready",
        alpha=round(model.alpha, 4),
        cells=len(adapter.cells_frame),
        events=panel.n_events,
    )
    return service


def midpoint_of(window: TimeWindow) -> datetime:
    """A timestamp guaranteed to be inside a window. Used by docs and tests."""
    return window.start + timedelta(seconds=window.duration.total_seconds() / 2)


__all__ = [
    "FLOW_SUPPORT_QUANTILES",
    "MIN_TRAINING_ROWS_PER_CELL",
    "CellContext",
    "RiskService",
    "build_demo_service",
    "midpoint_of",
]
