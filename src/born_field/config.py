"""Configuration schema."""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from born_field.types import RoadClass

DEFAULT_BBOX: Final[tuple[float, float, float, float]] = (
    -122.35,
    37.75,
    -122.15,
    37.90,
)
"""Default study area: a slice of the East Bay (lon_min, lat_min, lon_max, lat_max).

California specifically, because it is the only region where both shipped
adapter targets are coherent -- Caltrans PeMS for flow and TIMS/SWITRS for
collisions. A study area in a state neither source covers would make the
adapters decorative.
"""


class GridConfig(BaseModel):
    """Spatial and temporal discretisation of the study area."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bbox: tuple[float, float, float, float] = DEFAULT_BBOX
    cell_size_m: Annotated[float, Field(gt=0)] = 500.0
    window_hours: Annotated[float, Field(gt=0)] = 1.0

    @model_validator(mode="after")
    def _check_bbox(self) -> GridConfig:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        if lon_max <= lon_min or lat_max <= lat_min:
            msg = (
                "bbox must be (lon_min, lat_min, lon_max, lat_max) with min < max, "
                f"got {self.bbox!r}"
            )
            raise ValueError(msg)
        return self


class GeneratorConfig(BaseModel):
    """Ground truth for the synthetic world."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    true_alpha: Annotated[float, Field(gt=0.0, le=4.0)] = 1.35
    true_k: Annotated[float, Field(gt=0.0)] = 1e-7
    seed: int = 20260804

    class_multipliers: dict[RoadClass, Annotated[float, Field(gt=0)]] = Field(
        default_factory=lambda: {
            RoadClass.MOTORWAY: 0.6,
            RoadClass.TRUNK: 0.8,
            RoadClass.PRIMARY: 1.0,
            RoadClass.SECONDARY: 1.3,
            RoadClass.RESIDENTIAL: 1.8,
        }
    )
    """Per-mile risk multipliers. Motorways are below 1.0 deliberately: grade
    separation makes them safer *per vehicle-mile* even though they carry the
    most traffic. If the model cannot recover that ordering it has learned
    exposure, not risk."""

    wet_weather_multiplier: Annotated[float, Field(gt=0)] = 1.6
    wet_weather_fraction: Annotated[float, Field(ge=0, le=1)] = 0.15
    night_multiplier: Annotated[float, Field(gt=0)] = 1.9


class CorruptionConfig(BaseModel):
    """Controlled misspecification applied *after* sampling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    flow_lognormal_noise_sd: Annotated[float, Field(ge=0)] = 0.0
    """Multiplicative measurement error on observed flow. Classical
    errors-in-variables attenuates the fitted log-flow coefficient toward zero,
    i.e. biases alpha toward 1. Loop detectors are noisy and heavily imputed,
    so this is the single most consequential regime for a real deployment."""

    underreport_floor: Annotated[float, Field(ge=0, le=1)] = 1.0
    """Lowest crash-reporting probability across the study area (1.0 = full
    reporting). Reporting rates vary by neighbourhood; this regime is what
    connects the ethics section to a measured bias rather than a caveat."""

    aggregation_factor: Annotated[int, Field(ge=1)] = 1
    """Fit at ``aggregation_factor`` x the generating cell size (MAUP)."""

    drop_covariate: RoadClass | None = None
    """Omit a known covariate stratum to observe confounding load onto alpha."""

    overdispersion: Annotated[float, Field(ge=0)] = 0.0
    """Gamma-mixture scale producing negative-binomial counts. Affects interval
    width far more than the point estimate; a Poisson fit here yields intervals
    that are too narrow, which is a calibration failure, not a bias."""

    @property
    def is_clean(self) -> bool:
        """True when no corruption is active (the control condition)."""
        return (
            self.flow_lognormal_noise_sd == 0.0
            and self.underreport_floor == 1.0
            and self.aggregation_factor == 1
            and self.drop_covariate is None
            and self.overdispersion == 0.0
        )


class Settings(BaseSettings):
    """Runtime settings, environment-driven with safe local defaults."""

    model_config = SettingsConfigDict(
        env_prefix="BORNFIELD_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    log_level: str = "INFO"
    log_json: bool = False
    database_url: str = "postgresql+psycopg://bornfield:bornfield@localhost:5432/bornfield"
    # SQLite, not a file store: MLflow has put the filesystem backend into
    # maintenance mode and it now raises rather than degrading. SQLite keeps
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    grid: GridConfig = Field(default_factory=GridConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
