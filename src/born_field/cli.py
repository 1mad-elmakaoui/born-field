"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from born_field import __version__
from born_field.config import Settings
from born_field.logging import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="bornfield", description="Born Field pipeline CLI.")
    parser.add_argument("--version", action="version", version=f"born-field {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("config", help="Print the effective resolved configuration.")
    show.add_argument("--json", action="store_true", help="Emit JSON instead of a summary.")

    serve = sub.add_parser("serve", help="Run the API locally.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    sub.add_parser("db-init", help="Enable PostGIS and create the schema.")

    experiment = sub.add_parser("experiment", help="Run the alpha-recovery sweep and write plots.")
    experiment.add_argument("--seeds", type=int, default=12)
    experiment.add_argument("--output", type=Path, default=Path("experiments"))
    experiment.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    if args.command == "config":
        if args.json:
            try:
                json.dump(settings.model_dump(mode="json"), sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
                sys.stdout.flush()
            except BrokenPipeError:
                # `bornfield config --json | head` is normal usage; a downstream
                # reader closing early is not an error worth a traceback.
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
                return 0
        else:
            log = get_logger("cli")
            log.info(
                "effective_config",
                bbox=settings.grid.bbox,
                cell_size_m=settings.grid.cell_size_m,
                true_alpha=settings.generator.true_alpha,
                mlflow_tracking_uri=settings.mlflow_tracking_uri,
            )
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("born_field.api.main:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "db-init":
        from born_field.db import build_engine, init_schema

        init_schema(build_engine(settings.database_url))
        return 0

    if args.command == "experiment":
        return _run_experiment(args.seeds, args.output, mlflow=not args.no_mlflow)

    return 1


def _run_experiment(n_seeds: int, output: Path, *, mlflow: bool) -> int:
    """Run the recovery sweep, write artefacts, and print the summary table."""
    import pandas as pd

    from born_field.experiments.recovery import default_regimes, summarise, sweep
    from born_field.experiments.tracking import log_to_mlflow, plot_regime_recovery

    runs = list(sweep(default_regimes(), seeds=list(range(1, n_seeds + 1))))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in runs]).to_parquet(output / "regime_sweep.parquet")
    plot_regime_recovery(runs, output / "alpha_recovery_regimes.png")
    if mlflow:
        log_to_mlflow(runs)

    sys.stdout.write(summarise(runs).round(4).to_string(index=False) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
