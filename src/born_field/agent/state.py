"""Typed graph state."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from born_field.types import HazardEstimate, Latitude, Longitude, Refusal


class Stage(StrEnum):
    """Where the run got to. Useful in traces and in failure triage."""

    RECEIVED = "received"
    LOCATED = "located"
    SCORED = "scored"
    EXPLAINED = "explained"
    REFUSED = "refused"
    FAILED = "failed"


class ToolCall(BaseModel):
    """One deterministic tool invocation and its result."""

    model_config = ConfigDict(frozen=True)

    tool: str
    arguments: dict[str, Any]
    numeric_outputs: dict[str, float] = Field(default_factory=dict)
    text_outputs: dict[str, str] = Field(default_factory=dict)


class QueryState(BaseModel):
    """State threaded through the query-time graph."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # -- input
    lat: Latitude
    lon: Longitude
    timestamp: datetime
    flow_veh_per_hour: float | None = None
    is_wet: bool = False
    question: str | None = Field(
        default=None, description="Optional free-text question to answer in the explanation."
    )

    # -- progress
    stage: Stage = Stage.RECEIVED
    cell_id: str | None = None
    calls: list[ToolCall] = Field(default_factory=list)

    # -- outcome: exactly one of these is set
    estimate: HazardEstimate | None = None
    refusal: Refusal | None = None
    explanation: str | None = None

    # -- verification
    unverified_figures: list[str] = Field(default_factory=list)
    explanation_verified: bool = False
    error: str | None = None

    @property
    def is_refused(self) -> bool:
        """Whether the run ended in a refusal."""
        return self.refusal is not None

    @property
    def allowed_numbers(self) -> dict[str, float]:
        """Every number any tool produced, keyed by ``tool.field``."""
        allowed: dict[str, float] = {}
        for call in self.calls:
            for key, value in call.numeric_outputs.items():
                allowed[f"{call.tool}.{key}"] = value
        return allowed
