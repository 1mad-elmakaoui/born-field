# Running Born Field

Every command below was executed against this repository and the output shown is
what it actually printed. No credentials, no network access, no external data.

- [0. Prerequisites](#0-prerequisites)
- [1. Install](#1-install)
- [2. Verify the install](#2-verify-the-install)
- [3. Run the test suite](#3-run-the-test-suite)
- [4. Reproduce the flagship result](#4-reproduce-the-flagship-result)
- [5. Run the API](#5-run-the-api)
- [6. Run the agent evals](#6-run-the-agent-evals)
- [7. Full stack with PostGIS](#7-full-stack-with-postgis)
- [8. Swap in real data](#8-swap-in-real-data)
- [9. Troubleshooting](#9-troubleshooting)

---

## 0. Prerequisites

| Need | Why |
| --- | --- |
| Python 3.11 or 3.12 | Tested on both in CI. |
| [uv](https://docs.astral.sh/uv/) | Dependency and venv management. `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker *(optional)* | Only for §7. Everything else runs without it. |

You do **not** need PostgreSQL, an API key, or any dataset. The road network
ships in `data/fixtures/` and the collision data is generated.

```bash
git clone <repo-url> born-field && cd born-field
```

## 1. Install

```bash
uv sync --all-extras --dev
```

Roughly 90 seconds cold. This pulls the full stack, geospatial, modelling,
serving, and agent. To install less:

```bash
uv sync --dev                                  # core only, ~10s
uv sync --extra geo --extra model --dev        # enough for §4
uv sync --extra geo --extra model --extra serve --dev   # adds the API
```

Extras are separated so a dependency-resolution failure surfaces one layer at a
time instead of as a wall.

## 2. Verify the install

```bash
uv run bornfield --version
uv run bornfield config
```

`config` prints the resolved settings to **stderr**; `--json` writes machine-
readable output to **stdout**, so this is safe:

```bash
uv run bornfield config --json | jq .generator
```

```json
{
  "true_alpha": 1.35,
  "true_k": 1e-07,
  "seed": 20260804,
  "class_multipliers": { "motorway": 0.6, .. }
}
```

Any setting is overridable by environment variable, nested with a double
underscore:

```bash
BORNFIELD_GENERATOR__TRUE_ALPHA=1.9 uv run bornfield config --json | jq .generator.true_alpha
# 1.9
```

## 3. Run the test suite

```bash
make check        # lint + type-check + tests, exactly what CI runs
```

Or individually:

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
uv run pytest -q
```

**Expect 168 tests passing, ~4 minutes.** Most of that is fitting models inside
tests, which is deliberate, the tests that matter assert statistical
properties, not mocks.

Faster subsets while iterating:

```bash
uv run pytest tests/test_generator.py -q      # generator properties, ~40s
uv run pytest tests/test_agent.py -q          # graph + verifier, ~90s
uv run pytest -q -k "verifier or refusal"     # the guarantees, ~30s
```

## 4. Reproduce the flagship result

This is the headline: how flow-exponent recovery degrades under each realistic
data defect.

```bash
uv run bornfield experiment --seeds 12
```

**~10 minutes** (96 model fits). Fewer seeds for a quick look:

```bash
uv run bornfield experiment --seeds 3 --no-mlflow    # ~2.5 min
```

Prints the regime table and writes:

- `experiments/alpha_recovery_regimes.png`, the flagship plot
- `experiments/regime_sweep.parquet`, every run
- `mlflow.db`, tracked runs and metrics

```
                  regime  n_runs  true_alpha  mean_recovered_alpha  mean_bias  coverage
                   clean      12        1.35                1.3509     0.0009    1.0000
          overdispersion      12        1.35                1.3485    -0.0015    0.9167
          underreporting      12        1.35                1.3551     0.0051    0.9167
             aggregation      12        1.35                1.3569     0.0069    1.0000
       omitted_covariate      12        1.35                1.1912    -0.1588    0.0000
       measurement_error      12        1.35                0.9139    -0.4361    0.0000
measurement_error_severe      12        1.35                0.4739    -0.8761    0.0000
```

**Read the bottom two rows.** Detector measurement error drives α below 1,
meaning the pipeline would report risk *decreasing* with traffic density, the
opposite of the truth. That is pre-registered failure condition F5 firing, and
it is the most useful output here.

Browse the tracked runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # localhost:5000
```

## 5. Run the API

```bash
uv run bornfield serve
```

The demo model fits at startup, so **wait ~40 seconds** for readiness. Watch for
`api_ready` in the log, or poll:

```bash
curl -s localhost:8000/health
# {"status":"ok","model_loaded":true,"version":"0.1.0"}
```

Interactive docs: <http://localhost:8000/docs>

### Authentication

Every `/v1` endpoint needs an `X-API-Key` header. The stub accepts **any
non-empty value** and uses it as the caller identity for rate limiting:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"lat":37.82,"lon":-122.26,"timestamp":"2026-01-15T08:00:00Z"}'
# 401
```

### First: ask what the model will answer

```bash
curl -s localhost:8000/v1/model -H 'X-API-Key: demo' | jq
```

```json
{
  "model_name": "hazard_poisson_glm",
  "recovered_alpha": 1.3123585096971535,
  "exposure_null_rejected": true,
  "support": {
    "bbox": [-122.35, 37.75, -122.15, 37.9],
    "window_start": "2026-01-01T00:00:00+00:00",
    "window_end": "2026-03-02T00:00:00+00:00",
    "n_cells": 81,
    "flow_support": [36.46, 17609.90],
    "min_training_rows_per_cell": 24
  }
}
```

`support` is published so you can avoid queries that will be refused instead of
finding the boundary by trial and error. **Your timestamp must fall inside
`window_start`–`window_end`**, which is the most common first-run mistake.

### Score a point

```bash
curl -s -X POST localhost:8000/v1/score \
  -H 'Content-Type: application/json' -H 'X-API-Key: demo' \
  -d '{"lat":37.8200,"lon":-122.2600,"timestamp":"2026-01-15T08:00:00Z"}' | jq
```

```json
{
  "outcome": "estimate",
  "cell_id": "r0003c0004",
  "expected_crashes": 0.0018100807320095074,
  "rate_per_100m_vmt": 107.23863796138997,
  "rate_interval": { "lower": 93.38641910663358, "upper": 123.14558778276546, "level": 0.95 },
  "probability_at_least_one": 0.0018084435238566077,
  "exposure_vehicle_miles": 1687.899777933776,
  "model_name": "hazard_poisson_glm",
  "model_version": "1"
}
```

Note the interval is **asymmetric** around the estimate (−13.9 / +15.9): it is
propagated through the log link rather than approximated on the count scale, so
it stays strictly positive, the right shape for a rate near zero.

Optional fields: `flow_veh_per_hour` (omit to use the cell's fitted typical
flow) and `is_wet`.

### Every refusal path

All four return **HTTP 200** with `"outcome": "refusal"`. A refusal is an answer
the service computed, not a client error to retry.

```bash
# Outside the fitted region
curl -s -X POST localhost:8000/v1/score -H 'Content-Type: application/json' \
  -H 'X-API-Key: demo' \
  -d '{"lat":40.7128,"lon":-74.0060,"timestamp":"2026-01-15T08:00:00Z"}' | jq -c
# {"outcome":"refusal","reason":"outside_fitted_region","detail":"(40.71280, -74.00600) lies outside the fitted study area (-122.35, 37.75, -122.15, 37.9)","cell_id":null}

# Outside the fitted time range
.. -d '{"lat":37.82,"lon":-122.26,"timestamp":"2027-06-01T08:00:00Z"}'
# reason: outside_fitted_time_range

# Flow beyond the fitted support
.. -d '{"lat":37.82,"lon":-122.26,"timestamp":"2026-01-15T08:00:00Z","flow_veh_per_hour":900000}'
# reason: extrapolation_beyond_support

# Inside the bbox but in a cell with no road mileage
.. -d '{"lat":37.7520,"lon":-122.2430,"timestamp":"2026-01-15T08:00:00Z"}'
# reason: insufficient_coverage
# detail: "falls in no cell carrying road mileage; no exposure denominator exists here"
```

`insufficient_coverage` also fires when a cell contributed fewer than
`min_training_rows_per_cell` observations, though not in this demo, where a
60-day hourly fit gives every populated cell 1,440 rows. The reachable form here
is the one above: a point inside the bbox that lands where the network has no
roads, so there is no exposure denominator at all.

### Batch scoring

A refusal does not fail the batch, a network screening run legitimately
contains cells the model declines to score.

```bash
curl -s -X POST localhost:8000/v1/score/batch \
  -H 'Content-Type: application/json' -H 'X-API-Key: demo' \
  -d '{"queries":[
        {"lat":37.82,"lon":-122.26,"timestamp":"2026-01-15T08:00:00Z"},
        {"lat":40.7128,"lon":-74.0060,"timestamp":"2026-01-15T08:00:00Z"}]}' | jq -c
# n_estimates 1, n_refusals 1, results: ["estimate","refusal"]
```

### Calibration and metrics

```bash
curl -s localhost:8000/v1/calibration -H 'X-API-Key: demo' | jq '{expected_calibration_error, coverage_of_nominal_95}'
# { "expected_calibration_error": 0.00039521, "coverage_of_nominal_95": 0.9 }

curl -s localhost:8000/metrics | grep bornfield_refusals_total
```

`bornfield_refusals_total` is the operational signal worth alerting on: a rising
refusal rate in one region means fitted support has drifted from live traffic.

### Bruno collection

The [`bruno/`](bruno) directory has the same calls with assertions. Open it in
[Bruno](https://usebruno.com) and set `baseUrl` to `http://localhost:8000`.

## 6. Run the agent evals

```bash
uv run python scripts/run_agent_evals.py
```

```
agent eval: 13/13 passed

  PASS  inside_support_is_scored
  PASS  far_outside_region_is_refused
  ..
  PASS  plausible_but_wrong_interval_is_rejected
  PASS  invented_alpha_is_rejected
```

**No API key required**, which is the point: the properties under test are that
refusal routing holds and that no fabricated figure survives verification.
Neither depends on which language model writes the prose.

To use a real explainer:

```bash
export ANTHROPIC_API_KEY=sk-ant-..
```

```python
from born_field.agent.graph import build_anthropic_explainer, build_graph
from born_field.api.service import build_demo_service

service = build_demo_service()
run = build_graph(service, build_anthropic_explainer())
```

The verifier runs identically. Prose quoting an untraceable number is discarded
and replaced by the deterministic template, with the rejected figures recorded
on `state.unverified_figures`.

## 7. Full stack with PostGIS

```bash
docker compose up --build
```

Builds the image (~3 min cold), starts PostGIS, waits for `pg_isready`, then
starts the API. The container healthcheck allows a 90-second start period
because the demo model fits at startup.

```bash
curl -s localhost:8000/health
docker compose exec api bornfield db-init      # PostGIS extension + schema
docker compose down -v                         # -v also drops the volume
```

Against a database you are running yourself:

```bash
export BORNFIELD_DATABASE_URL='postgresql+psycopg://user:pass@host:5432/db'
uv run bornfield db-init
```

## 8. Swap in real data

### A real road network

```bash
uv run python scripts/fetch_osm_extract.py
```

Fetches OSM ways for the configured bbox and overwrites
`data/fixtures/study_area_roads.parquet`. **Nothing downstream changes**, the
grid, adapter, model, and API all read whatever is in that file. Requires
outbound access to an Overpass endpoint; the shipped fixture is a procedurally
generated network precisely so the default path needs neither.

Change the study area first if you want somewhere else:

```bash
export BORNFIELD_GRID__BBOX='[-118.35,33.95,-118.15,34.10]'   # Los Angeles
```

### A real data source

Implement the `DataSourceAdapter` Protocol, see [docs/adapters.md](docs/adapters.md)
for the contract and the four non-negotiable requirements.

**Read [MODEL_CARD.md](MODEL_CARD.md) §3 first.** If your flow feed is imputed
or noisy, the measured attenuation is severe enough to invert the sign of the
conclusion.

## 9. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: geopandas` | Missing extra. `uv sync --all-extras --dev`. |
| `/v1/*` returns 401 | Add `-H 'X-API-Key: anything'`. |
| `/health` shows `model_loaded: false` | Startup still fitting; wait ~40s. |
| Everything refuses with `outside_fitted_time_range` | Your timestamp is outside the fitted window. Check `GET /v1/model` → `support.window_start`. This is the most common first-run mistake. |
| `outside_fitted_region` for a coordinate you expect | `lat`/`lon` transposed, or outside the bbox in `support`. |
| `MlflowException: filesystem tracking backend .. maintenance mode` | A stale `BORNFIELD_MLFLOW_TRACKING_URI=file:..`. Use `sqlite:///mlflow.db`. |
| `COPY data ./data` fails in Docker | `data/fixtures/*.parquet` is missing. Regenerate with `uv run python scripts/build_fixture.py`. |
| Tests slow | Expected, they fit real models. Use `-k` to select a subset. |
| `docker compose` cannot reach the DB | Compose waits on `pg_isready`; check `docker compose logs db`. |

### Useful environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `BORNFIELD_LOG_LEVEL` | `INFO` | Log verbosity. |
| `BORNFIELD_LOG_JSON` | `false` | JSON logs (containers set this true). |
| `BORNFIELD_DATABASE_URL` | local Postgres | PostGIS connection. |
| `BORNFIELD_MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Tracking backend. |
| `BORNFIELD_GRID__CELL_SIZE_M` | `500` | Cell edge. Changing it invalidates fitted models. |
| `BORNFIELD_GENERATOR__TRUE_ALPHA` | `1.35` | Ground-truth flow exponent. |

See [`.env.example`](.env.example). All logs go to **stderr**, so stdout stays a
clean data channel.
