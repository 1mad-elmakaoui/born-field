"""Synthetic data source: the only adapter that ships working."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd

from born_field.adapters.base import Coverage
from born_field.config import CorruptionConfig, GeneratorConfig, GridConfig
from born_field.field.grid import build_cells, build_segments, cells_to_models
from born_field.field.network import default_network
from born_field.types import (
    CrashObservation,
    FlowObservation,
    GridCell,
    RoadClass,
    TimeWindow,
    vehicle_miles_travelled_array,
)

# Free-flow throughput by class, vehicles per hour, whole-cell aggregate.
_BASE_FLOW: dict[str, float] = {
    RoadClass.MOTORWAY.value: 5800.0,
    RoadClass.TRUNK.value: 3100.0,
    RoadClass.PRIMARY.value: 1450.0,
    RoadClass.SECONDARY.value: 690.0,
    RoadClass.RESIDENTIAL.value: 130.0,
}

_NIGHT_START_HOUR = 20
_NIGHT_END_HOUR = 6


def _diurnal_multiplier(hours_of_day: np.ndarray) -> np.ndarray:
    """Twin-peaked commute profile, normalised to a daily mean near 1."""
    morning = np.exp(-(((hours_of_day - 8.0) / 1.6) ** 2))
    evening = np.exp(-(((hours_of_day - 17.5) / 2.0) ** 2))
    overnight = 0.12
    return overnight + 0.55 + 1.35 * morning + 1.55 * evening


def _is_night(hours_of_day: np.ndarray) -> np.ndarray:
    """Night flag used by both the flow model and the hazard law."""
    return (hours_of_day >= _NIGHT_START_HOUR) | (hours_of_day < _NIGHT_END_HOUR)


class SyntheticAdapter:
    """Generates a complete synthetic world over the configured study area."""

    def __init__(
        self,
        grid: GridConfig | None = None,
        generator: GeneratorConfig | None = None,
        corruption: CorruptionConfig | None = None,
        network: gpd.GeoDataFrame | None = None,
        start: datetime | None = None,
        days: int = 30,
    ) -> None:
        """Build the study area and fix the sampling parameters."""
        self.grid_config = grid or GridConfig()
        self.generator_config = generator or GeneratorConfig()
        self.corruption = corruption or CorruptionConfig()
        self.days = days
        self.start = start or datetime(2026, 1, 1, tzinfo=UTC)

        self._network = (
            network
            if network is not None
            else default_network(self.grid_config.bbox, seed=self.generator_config.seed)
        )
        self.cells_frame = build_cells(
            self._network, self.grid_config.bbox, self.grid_config.cell_size_m
        )
        self.segments_frame = build_segments(self.cells_frame)
        self._frame: pd.DataFrame | None = None

    # -- adapter identity ---------------------------------------------------

    @property
    def name(self) -> str:
        """Stable identifier recorded with every fitted model."""
        return "synthetic"

    @property
    def version(self) -> str:
        """Encodes the parameters that change the data, for run provenance."""
        cfg = self.generator_config
        return f"alpha={cfg.true_alpha:g};k={cfg.true_k:g};seed={cfg.seed};days={self.days}"

    def coverage(self) -> Coverage:
        """Declared spatial and temporal support."""
        return Coverage(bbox=self.grid_config.bbox, window=self.window)

    @property
    def window(self) -> TimeWindow:
        """Full generated time range."""
        return TimeWindow(start=self.start, end=self.start + timedelta(days=self.days))

    # -- vectorised path ----------------------------------------------------

    def frame(self) -> pd.DataFrame:
        """Materialise the full cell-window panel, sampling once and caching."""
        if self._frame is None:
            self._frame = self._sample()
        return self._frame

    def _sample(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.generator_config.seed)
        cfg = self.generator_config
        # One row per (cell, road_class). The hazard law applies per stratum, each
        # with its own flow and multiplier.
        segments = self.segments_frame

        window_hours = self.grid_config.window_hours
        n_windows = int(self.days * 24 / window_hours)
        offsets = np.arange(n_windows) * window_hours
        starts = np.array([self.start + timedelta(hours=float(h)) for h in offsets], dtype=object)
        hours_of_day = (offsets + self.start.hour) % 24
        day_index = (offsets // 24).astype(int)
        is_weekend = ((day_index + self.start.weekday()) % 7) >= 5

        n_segments = len(segments)
        cell_ids = segments["cell_id"].to_numpy()
        road_miles = segments["road_miles"].to_numpy(dtype=float)
        road_class = segments["road_class"].to_numpy()

        # Spatial demand gradient. Uniform flow would leave the flow exponent
        # barely identified.
        cx, cy = segments["col"].to_numpy(), segments["row"].to_numpy()
        radial = np.hypot(
            (cx - cx.mean()) / max(cx.std(), 1.0), (cy - cy.mean()) / max(cy.std(), 1.0)
        )
        demand = 0.45 + 0.95 * np.exp(-0.5 * (radial / 1.4) ** 2)

        base = np.array([_BASE_FLOW[c] for c in road_class])

        # Outer product over (window, segment).
        temporal = _diurnal_multiplier(hours_of_day) * np.where(is_weekend, 0.72, 1.0)
        true_flow = np.outer(temporal, base * demand)

        # Real segment-to-segment variability, present in the flow that causes
        # crashes. Measurement error is applied later.
        true_flow *= np.exp(rng.normal(0.0, 0.22, size=true_flow.shape) - 0.22**2 / 2)
        true_flow = np.clip(true_flow, 1.0, None)

        night = _is_night(hours_of_day)
        # Weather is shared across the study area within a window, inducing the
        # cross-cell correlation that makes naive standard errors optimistic.
        wet = rng.random(n_windows) < cfg.wet_weather_fraction

        class_mult = np.array([cfg.class_multipliers[RoadClass(c)] for c in road_class])
        time_mult = np.where(night, cfg.night_multiplier, 1.0)
        weather_mult = np.where(wet, cfg.wet_weather_multiplier, 1.0)

        # The hazard law. flow enters with exponent alpha; road_miles and hours
        # enter linearly, which is what makes them part of exposure.
        rate = (
            cfg.true_k
            * true_flow**cfg.true_alpha
            * class_mult[None, :]
            * (time_mult * weather_mult)[:, None]
            * road_miles[None, :]
            * window_hours
        )

        if self.corruption.overdispersion > 0:
            # Gamma-Poisson mixture gives negative binomial counts. Mean holds,
            # variance inflates, and a Poisson fit then understates the interval.
            shape = 1.0 / self.corruption.overdispersion
            rate = rate * rng.gamma(shape=shape, scale=1.0 / shape, size=rate.shape)

        true_counts = rng.poisson(rate)

        # Reporting probability varies smoothly across space, so underreporting is
        # spatially structured. That is what makes it a confounder.
        floor = self.corruption.underreport_floor
        if floor < 1.0:
            gradient = (cx - cx.min()) / max(cx.max() - cx.min(), 1)
            report_p = floor + (1.0 - floor) * gradient
        else:
            report_p = np.ones(n_segments)
        reported_counts = rng.binomial(true_counts, report_p[None, :])

        # Measurement error on the *observed* flow only, applied after sampling.
        sigma = self.corruption.flow_lognormal_noise_sd
        observed_flow = (
            true_flow * np.exp(rng.normal(0.0, sigma, size=true_flow.shape) - sigma**2 / 2)
            if sigma > 0
            else true_flow
        )

        n_rows = n_windows * n_segments
        frame = pd.DataFrame(
            {
                "cell_id": np.tile(cell_ids, n_windows),
                "window_start": np.repeat(starts, n_segments),
                "row": np.tile(segments["row"].to_numpy(), n_windows),
                "col": np.tile(cx, n_windows),
                "road_class": np.tile(road_class, n_windows),
                "road_miles": np.tile(road_miles, n_windows),
                "flow_veh_per_hour": observed_flow.reshape(n_rows),
                "true_flow_veh_per_hour": true_flow.reshape(n_rows),
                "is_night": np.repeat(night, n_segments),
                "is_wet": np.repeat(wet, n_segments),
                "crash_count": reported_counts.reshape(n_rows),
                "true_crash_count": true_counts.reshape(n_rows),
                "true_rate": rate.reshape(n_rows),
            }
        )
        frame["exposure_vehicle_miles"] = vehicle_miles_travelled_array(
            frame["flow_veh_per_hour"].to_numpy(),
            frame["road_miles"].to_numpy(),
            window_hours,
        )
        return frame

    # -- row-wise Protocol path ---------------------------------------------

    def cells(self) -> Sequence[GridCell]:
        """The analysis grid as validated domain objects."""
        return cells_to_models(self.cells_frame)

    def flow(self, window: TimeWindow) -> Iterable[FlowObservation]:
        """Observed flow per cell-window, clipped to ``window``."""
        frame = self._within(window)
        imputed = self.corruption.flow_lognormal_noise_sd > 0
        # `itertuples` cannot carry per-column types, so each value is narrowed
        # explicitly at the boundary rather than trusting the frame's contents.
        for row in frame.itertuples():
            yield FlowObservation(
                cell_id=str(row.cell_id),
                window_start=cast("datetime", row.window_start),
                flow_veh_per_hour=float(cast("float", row.flow_veh_per_hour)),
                imputed=imputed,
            )

    def crashes(self, window: TimeWindow) -> Iterable[CrashObservation]:
        """Individual collision records, expanded from cell-window counts."""
        frame = self._within(window)
        rng = np.random.default_rng(self.generator_config.seed + 1)
        centroids = self.cells_frame.set_index("cell_id")[["centroid_lon", "centroid_lat"]]
        half_deg = self.grid_config.cell_size_m / 111_320.0 / 2

        for row in frame.itertuples():
            count = int(cast("int", row.crash_count))
            if count == 0:
                continue
            occurred_at = cast("datetime", row.window_start)
            lon0, lat0 = centroids.loc[row.cell_id]
            for k in range(count):
                yield CrashObservation(
                    crash_id=f"{row.cell_id}-{occurred_at:%Y%m%d%H}-{k}",
                    lon=float(np.clip(lon0 + rng.uniform(-half_deg, half_deg), -180, 180)),
                    lat=float(np.clip(lat0 + rng.uniform(-half_deg, half_deg), -90, 90)),
                    occurred_at=occurred_at,
                    reported=True,
                )

    def _within(self, window: TimeWindow) -> pd.DataFrame:
        frame = self.frame()
        starts = frame["window_start"]
        return frame[(starts >= window.start) & (starts < window.end)]
