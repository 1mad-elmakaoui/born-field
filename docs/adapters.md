# Writing a data-source adapter

A buyer forking this repository replaces exactly one thing: the adapter.
Everything downstream, intensity field, hazard model, validation, serving, is
written against [`DataSourceAdapter`](./src/born_field/adapters/base.py) and
nothing else.

## The contract

```python
class DataSourceAdapter(Protocol):
    @property
    def name(self) -> str: ..
    @property
    def version(self) -> str: ..
    def coverage(self) -> Coverage: ..
    def cells(self) -> Sequence[GridCell]: ..
    def flow(self, window: TimeWindow) -> Iterable[FlowObservation]: ..
    def crashes(self, window: TimeWindow) -> Iterable[CrashObservation]: ..
```

It is a `Protocol`, not an abstract base class: an existing internal telematics
client satisfies it structurally, with no import from this package.

## Four requirements that are not negotiable

**1. `cells()` must supply road mileage.** It is half the exposure denominator.
An adapter that cannot supply centre-line miles per cell cannot support a
per-vehicle-mile model, and the honest response is to fail at construction rather
than substitute a constant.

**2. `coverage()` must be truthful.** It is what the refusal path is checked
against. A source that overstates its support turns a refusal into a silent
extrapolation, which is the worst failure this system can have.

**3. Preserve `imputed` and `reported`.** Both are bias sources and both are
unrecoverable once dropped. PeMS imputes a large share of samples; crash
reporting varies by neighbourhood.

**4. Compute lengths in a projected CRS.** Storage is EPSG:4326; every length and
area computation happens in EPSG:3310. A degree is not a distance, and computing
segment length in geographic coordinates silently corrupts every result while
producing a plausible-looking risk surface.

## Optional: the vectorised fast path

Implement `VectorisedSource.frame()` if your source can produce a columnar panel.
Callers detect it with `isinstance` and fall back to the row-wise path otherwise,
so it is genuinely optional. It matters for batch refits over millions of rows
and does nothing for point queries.

## Validating a new adapter

```python
from born_field.adapters import DataSourceAdapter
from born_field.models import HazardPanel

assert isinstance(my_adapter, DataSourceAdapter)   # structural conformance
panel = HazardPanel(my_adapter.frame())            # schema + invariants
```

`HazardPanel` construction is the real gate: it rejects non-positive exposure
(whose log offset is `-inf`), negative counts, and missing columns.

Then check identifiability before trusting any exponent:

```python
import numpy as np
log_flow = np.log(panel.data["flow_veh_per_hour"])
assert log_flow.std() > 0.3, "flow has too little spread to identify alpha"
```

Note that what matters is the *within-class* spread of log flow, not its
marginal spread, the road-class dummies absorb most of the latter. See the
attenuation analysis in the README.

Read [model card](./MODEL_CARD.md) §3 before connecting a real source. If your
flow feed is imputed or noisy, the measured attenuation is severe enough to
invert the sign of the conclusion.

## Shipped adapters

| Adapter | Status | Notes |
| --- | --- | --- |
| `SyntheticAdapter` | **working** | Samples from a known hazard law. Zero credentials. |
| `PeMSAdapter` | documented stub | Caltrans loop detectors. Needs an errors-in-variables correction, see F5. |
| `SWITRSAdapter` | documented stub | California collision records via TIMS. Carries the reporting bias in model card §3. |
