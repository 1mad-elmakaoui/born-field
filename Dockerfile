# Multi-stage: the geospatial stack pulls in a large build toolchain that has no
# business in a serving image.
FROM python:3.11-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies are installed from the lockfile before the source is copied, so a
# code change does not invalidate the (slow) dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra geo --extra model --extra serve --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev --extra geo --extra model --extra serve

FROM python:3.11-slim AS runtime

# GEOS/PROJ are runtime shared libraries for shapely and pyproj; without them
# the wheels import but fail on first geometry operation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgeos-c1v5 libproj25 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root: the service reads a model and answers queries, and needs nothing else.
RUN useradd --create-home --uid 10001 bornfield
WORKDIR /app

COPY --from=builder --chown=bornfield:bornfield /app/.venv /app/.venv
COPY --from=builder --chown=bornfield:bornfield /app/src /app/src
COPY --from=builder --chown=bornfield:bornfield /app/data /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BORNFIELD_LOG_JSON=true

USER bornfield
EXPOSE 8000

# Fitting the demo model at startup takes ~30s, so the start period is generous;
# without it the container would be killed before it ever became ready.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "born_field.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
