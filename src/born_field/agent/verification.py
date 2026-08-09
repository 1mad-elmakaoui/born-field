"""Numeric traceability: the guarantee that the LLM does not invent figures."""

from __future__ import annotations

import re
from dataclasses import dataclass

# The fractional part needs a digit after the point. Without it the pattern eats
# a sentence-ending full stop and "See item 2." parses as "2.".
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Small bare integers are structural: ordinals, the 95 in "95% interval", hours.
# Keyed on written form, so "2" is exempt and "2.0" is a measurement that is not.
_STRUCTURAL_MAX = 100.0
_BARE_INTEGER = re.compile(r"^[-+]?[\d,]+$")

RELATIVE_TOLERANCE = 0.02
"""Allowed rounding slack. An explanation saying "about 3.2" for 3.1978 is
faithful; one saying 4.5 is not."""


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of checking one explanation."""

    verified: bool
    unverified: tuple[str, ...]
    checked: int

    @property
    def summary(self) -> str:
        """One line for logs and traces."""
        if self.verified:
            return f"all {self.checked} figures traceable to tool outputs"
        return f"{len(self.unverified)} of {self.checked} figures untraceable: " + ", ".join(
            self.unverified
        )


def _candidate_forms(value: float) -> set[float]:
    """Representations a faithful writer might use for one tool output."""
    return {f for f in (value, value * 100.0) if f == f}  # drop NaN


def verify_explanation(
    explanation: str,
    allowed: dict[str, float],
    query_values: tuple[float, ...] = (),
) -> VerificationResult:
    """Check that every figure in ``explanation`` traces to a tool output."""
    permitted: set[float] = set()
    for value in allowed.values():
        permitted |= _candidate_forms(value)
    for value in query_values:
        permitted |= _candidate_forms(value)

    unverified: list[str] = []
    checked = 0

    for match in _NUMBER.finditer(explanation):
        raw = match.group()
        try:
            number = float(raw.replace(",", ""))
        except ValueError:  # pragma: no cover - regex admits only parseable forms
            continue
        checked += 1

        # Small bare integers are structural (ordinals, clock hours, list
        # counts, "95% interval"). A decimal point disqualifies the exemption.
        if _BARE_INTEGER.match(raw) and abs(number) <= _STRUCTURAL_MAX:
            continue

        if any(
            abs(number - candidate) <= RELATIVE_TOLERANCE * max(abs(candidate), 1e-9)
            or number == candidate
            for candidate in permitted
        ):
            continue

        unverified.append(raw)

    return VerificationResult(
        verified=not unverified, unverified=tuple(unverified), checked=checked
    )
