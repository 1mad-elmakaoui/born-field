"""Run the agent golden set."""

from __future__ import annotations

import sys

from born_field.agent.evals import format_report, run_all
from born_field.api.service import build_demo_service


def main() -> int:
    """Run the suite and report."""
    results = run_all(build_demo_service(days=45))
    sys.stdout.write(format_report(results) + "\n")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
