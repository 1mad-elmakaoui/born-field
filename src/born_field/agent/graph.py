"""The query-time agent graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from langgraph.graph import END, StateGraph

from born_field.agent.state import QueryState, Stage
from born_field.agent.tools import locate_cell, model_context, score_point
from born_field.agent.verification import verify_explanation
from born_field.api.service import RiskService
from born_field.logging import get_logger
from born_field.types import Refusal

log = get_logger(__name__)


class Explainer(Protocol):
    """Turns a scored state into prose. May phrase; may not compute."""

    def __call__(self, state: QueryState) -> str:
        """Return an explanation of the state's outcome."""
        ...


def template_explainer(state: QueryState) -> str:
    """Deterministic explanation."""
    if state.refusal is not None:
        return (
            f"No estimate was produced. Reason: {state.refusal.reason.value}. "
            f"{state.refusal.detail}"
        )

    estimate = state.estimate
    if estimate is None:  # pragma: no cover - unreachable via the graph
        return "No estimate is available for this query."

    return (
        f"Cell {estimate.cell_id} is estimated at "
        f"{estimate.rate_per_100m_vmt:.3g} collisions per 100 million vehicle-miles "
        f"(95% interval {estimate.rate_interval.lower:.3g} to "
        f"{estimate.rate_interval.upper:.3g}). Over this one-hour window the expected "
        f"count is {estimate.expected_crashes:.3g}, a "
        f"{estimate.probability_at_least_one:.3g} probability of at least one collision, "
        f"against exposure of {estimate.exposure_vehicle_miles:.4g} vehicle-miles. "
        f"Model {estimate.model_name} version {estimate.model_version}."
    )


EXPLAIN_SYSTEM_PROMPT = """You explain collision-risk estimates to transport \
engineers and insurance analysts.

Rules, without exception:
- Use ONLY the numbers given to you in the tool outputs below. Never compute a \
new figure, never convert units, never estimate, never round to a value that \
changes the digit before the decimal point.
- If a number you want is not in the tool outputs, describe it in words instead.
- Always state the uncertainty interval alongside any rate you quote.
- Risk is per vehicle-mile. Never describe a location as dangerous on the basis \
of collision counts alone; a busy road has more collisions and may still be \
safer per vehicle-mile.
- Be concise: three or four sentences.
- This estimate is for infrastructure prioritisation. Do not suggest using it \
to price an individual policy or to target enforcement.
"""


def build_anthropic_explainer(model: str = "claude-sonnet-5") -> Explainer:
    """An LLM explainer."""

    def explain(state: QueryState) -> str:
        import anthropic

        client = anthropic.Anthropic()
        facts = "\n".join(
            f"{call.tool}.{key} = {value!r}"
            for call in state.calls
            for key, value in {**call.numeric_outputs, **call.text_outputs}.items()
        )
        question = state.question or "Explain this collision-risk estimate."
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=EXPLAIN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Tool outputs:\n{facts}\n\n{question}"}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return explain


def build_graph(
    service: RiskService,
    explainer: Explainer | None = None,
) -> Callable[[QueryState], QueryState]:
    """Compile the query graph and return a callable that runs it."""
    explain_with = explainer or template_explainer

    def receive(_: QueryState) -> dict[str, Any]:
        return {"stage": Stage.RECEIVED, "calls": [model_context(service)]}

    def locate(state: QueryState) -> dict[str, Any]:
        call = locate_cell(service, state)
        cell_id = call.text_outputs.get("cell_id")
        return {
            "stage": Stage.LOCATED,
            "calls": [*state.calls, call],
            "cell_id": None if cell_id == "none" else cell_id,
        }

    def score(state: QueryState) -> dict[str, Any]:
        call, result = score_point(service, state)
        calls = [*state.calls, call]
        if isinstance(result, Refusal):
            return {"stage": Stage.REFUSED, "calls": calls, "refusal": result}
        return {"stage": Stage.SCORED, "calls": calls, "estimate": result}

    def explain(state: QueryState) -> dict[str, Any]:
        try:
            text = explain_with(state)
        except Exception as exc:
            log.warning("explainer_failed", error=str(exc))
            return {
                "stage": Stage.EXPLAINED,
                "explanation": template_explainer(state),
                "explanation_verified": True,
                "error": f"explainer unavailable: {exc}",
            }
        return {"stage": Stage.EXPLAINED, "explanation": text}

    def verify(state: QueryState) -> dict[str, Any]:
        if state.explanation_verified:
            return {}

        query_values = (state.lat, state.lon, state.flow_veh_per_hour or 0.0)
        result = verify_explanation(state.explanation or "", state.allowed_numbers, query_values)
        if result.verified:
            return {"explanation_verified": True}

        # An untraceable figure is never returned. The deterministic template
        # replaces it, and the violation is recorded rather than swallowed.
        log.warning(
            "explanation_rejected",
            unverified=list(result.unverified),
            cell_id=state.cell_id,
        )
        return {
            "explanation": template_explainer(state),
            "explanation_verified": False,
            "unverified_figures": list(result.unverified),
        }

    def refuse(state: QueryState) -> dict[str, Any]:
        return {
            "stage": Stage.REFUSED,
            "explanation": template_explainer(state),
            "explanation_verified": True,
        }

    def route(state: QueryState) -> str:
        return "refuse" if state.is_refused else "explain"

    builder: StateGraph[QueryState, None, QueryState, QueryState] = StateGraph(QueryState)

    # LangGraph's add_node overloads reject a callable returning a partial state
    # update under mypy --strict, so the ignore sits here and not at six sites.
    nodes: dict[str, Callable[[QueryState], dict[str, Any]]] = {
        "receive": receive,
        "locate": locate,
        "score": score,
        "explain": explain,
        "verify": verify,
        "refuse": refuse,
    }
    for name, action in nodes.items():
        builder.add_node(name, action)  # type: ignore[call-overload]

    builder.set_entry_point("receive")
    builder.add_edge("receive", "locate")
    builder.add_edge("locate", "score")
    builder.add_conditional_edges("score", route, {"explain": "explain", "refuse": "refuse"})
    builder.add_edge("explain", "verify")
    builder.add_edge("verify", END)
    builder.add_edge("refuse", END)

    compiled = builder.compile()

    def run(state: QueryState) -> QueryState:
        return QueryState.model_validate(compiled.invoke(state))

    return run


def build_traced_graph(
    service: RiskService, explainer: Explainer | None = None
) -> Callable[[QueryState], QueryState]:
    """The graph with Langfuse tracing attached when it is configured."""
    runner = build_graph(service, explainer)

    try:
        from langfuse import Langfuse

        client = Langfuse()
    except Exception as exc:
        log.info("tracing_disabled", reason=str(exc))
        return runner

    def traced(state: QueryState) -> QueryState:
        try:
            with client.start_as_current_observation(
                name="query_risk",
                as_type="span",
                input=state.model_dump(mode="json", exclude={"calls"}),
            ) as span:
                final = runner(state)
                span.update(
                    output={
                        "stage": final.stage.value,
                        "refused": final.is_refused,
                        "explanation_verified": final.explanation_verified,
                        "unverified_figures": final.unverified_figures,
                    },
                    metadata={
                        "cell_id": final.cell_id,
                        "n_tool_calls": len(final.calls),
                    },
                )
                return final
        except Exception as exc:
            log.warning("trace_emit_failed", error=str(exc))
            return runner(state)

    return traced
