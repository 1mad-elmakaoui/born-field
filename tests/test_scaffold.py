"""Stage 0 smoke tests.

Deliberately not ``assert True``. These pin the two things Stage 0 actually
claims: that the exposure denominator is computed correctly, and that the
domain types reject invalid states rather than propagating them into a model.
The heavy property tests for the generator arrive in Stage 1.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from born_field import __version__, vehicle_miles_travelled
from born_field.adapters import Coverage, DataSourceAdapter
from born_field.config import GeneratorConfig, GridConfig, Settings
from born_field.models import HazardModel
from born_field.types import METRES_PER_MILE, RoadClass, TimeWindow


def test_version_is_exported():
    assert __version__


def test_protocols_are_runtime_checkable():
    # A buyer's own class must be able to satisfy these structurally.
    assert not isinstance(object(), DataSourceAdapter)
    assert not isinstance(object(), HazardModel)


class TestExposure:
    """VMT is the Poisson offset; an error here silently corrupts every result."""

    def test_known_value(self):
        # 1000 veh/h over 2 miles of road for 3 hours = 6000 vehicle-miles.
        assert vehicle_miles_travelled(1000.0, 2.0, 3.0) == pytest.approx(6000.0)

    def test_zero_flow_gives_zero_exposure(self):
        assert vehicle_miles_travelled(0.0, 5.0, 24.0) == 0.0

    @pytest.mark.parametrize(
        ("flow", "miles", "hours"),
        [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)],
    )
    def test_rejects_negative_inputs(self, flow, miles, hours):
        with pytest.raises(ValueError, match="non-negative"):
            vehicle_miles_travelled(flow, miles, hours)

    @given(
        flow=st.floats(min_value=0, max_value=1e5, allow_nan=False),
        miles=st.floats(min_value=0, max_value=1e3, allow_nan=False),
        hours=st.floats(min_value=0, max_value=1e4, allow_nan=False),
        scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
    )
    def test_linear_in_flow(self, flow, miles, hours, scale):
        """Exposure must be exactly linear in flow.

        This is the property that makes ``offset = log(VMT)`` meaningful: any
        nonlinearity in the *offset* would be indistinguishable from the flow
        exponent the model is trying to estimate.
        """
        assert vehicle_miles_travelled(flow * scale, miles, hours) == pytest.approx(
            scale * vehicle_miles_travelled(flow, miles, hours)
        )

    def test_mile_constant_is_exact(self):
        assert METRES_PER_MILE == 1609.344


class TestTimeWindow:
    def test_rejects_reversed_bounds(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        with pytest.raises(ValidationError):
            TimeWindow(start=now, end=now - timedelta(hours=1))

    def test_rejects_zero_length(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        with pytest.raises(ValidationError):
            TimeWindow(start=now, end=now)

        def test_rejects_naive_datetimes(self):
            naive = datetime(2026, 1, 1)  # noqa: DTZ001 - intentionally naive: that is the point
            with pytest.raises(ValidationError):
                TimeWindow(start=naive, end=naive + timedelta(hours=1))

    def test_hours(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        window = TimeWindow(start=start, end=start + timedelta(minutes=90))
        assert window.hours == pytest.approx(1.5)


class TestConfig:
    def test_defaults_load(self):
        settings = Settings()
        assert settings.grid.cell_size_m > 0
        assert settings.generator.true_alpha > 0

    def test_bbox_ordering_enforced(self):
        with pytest.raises(ValidationError):
            GridConfig(bbox=(-122.0, 37.9, -122.3, 37.7))

    def test_generator_covers_every_road_class(self):
        """A missing stratum would make the class effect unidentifiable."""
        multipliers = GeneratorConfig().class_multipliers
        assert set(multipliers) == set(RoadClass)

    def test_motorways_are_safer_per_vehicle_mile(self):
        """Encodes the exposure confound in the ground truth itself.

        Motorways carry the most traffic and the most crashes, but fewer
        crashes *per vehicle-mile*. A model that ranks them most dangerous has
        recovered exposure, not risk.
        """
        multipliers = GeneratorConfig().class_multipliers
        assert multipliers[RoadClass.MOTORWAY] < multipliers[RoadClass.RESIDENTIAL]

    def test_config_is_frozen(self):
        with pytest.raises(ValidationError):
            GridConfig().cell_size_m = 1000.0  # type: ignore[misc]


class TestCorruptionDefaults:
    def test_default_is_the_clean_control(self):
        from born_field.config import CorruptionConfig

        assert CorruptionConfig().is_clean

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"flow_lognormal_noise_sd": 0.3},
            {"underreport_floor": 0.6},
            {"aggregation_factor": 2},
            {"drop_covariate": RoadClass.RESIDENTIAL},
            {"overdispersion": 0.5},
        ],
    )
    def test_each_regime_is_detected_as_dirty(self, kwargs):
        from born_field.config import CorruptionConfig

        assert not CorruptionConfig(**kwargs).is_clean


class TestCoverage:
    def test_point_containment(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        coverage = Coverage(
            bbox=(-122.35, 37.75, -122.15, 37.90),
            window=TimeWindow(start=start, end=start + timedelta(days=1)),
        )
        assert coverage.contains_point(-122.25, 37.80)
        assert not coverage.contains_point(-118.24, 34.05)  # Los Angeles
