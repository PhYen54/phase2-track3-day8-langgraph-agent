"""State schema for the Day 08 LangGraph lab.

Students should extend the schema only when needed. Keep state lean and serializable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, TypedDict

from operator import add
from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph state with clear append-only vs overwrite strategy.

    APPEND-ONLY fields (use `add` reducer):
    - messages: accumulate conversation history for audit trail
    - tool_results: retain all tool calls for reproducibility and retry decisions
    - errors: build error log across retries for debugging and dead-letter escalation
    - events: append-only audit log of every node transition with metadata

    OVERWRITE fields (no reducer - represents current state):
    - thread_id: unique run identifier (constant, set at start)
    - scenario_id: scenario reference (constant, set at start)
    - query: normalized user query (immutable after intake normalization)
    - route: current classified route (updated by classify node)
    - risk_level: current risk assessment (updated by classify node)
    - attempt: current retry attempt count (incremented by retry node)
    - max_attempts: maximum retry limit (constant from scenario)
    - final_answer: final response to user (computed by answer/clarify/dead_letter nodes)
    - pending_question: clarification question if needed (set by clarify node)
    - proposed_action: risky action proposal awaiting approval (set by risky_action node)
    - approval: approval decision with reviewer metadata (set by approval node)
    - evaluation_result: evaluation of latest tool result (set by evaluate node)

    This design ensures:
    1. Auditability: append-only lists preserve full execution history
    2. Reproducibility: tool_results allow deterministic retry and replay
    3. Clarity: overwrite fields show current state (route, attempt, etc.)
    4. Efficiency: state stays lean; no redundant copies of transient values
    """

    thread_id: str
    scenario_id: str
    query: str
    route: str
    risk_level: str
    attempt: int
    max_attempts: int
    final_answer: str | None
    pending_question: str | None
    proposed_action: str | None
    approval: dict[str, Any] | None
    evaluation_result: str | None
    messages: Annotated[list[str], add]
    tool_results: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict[str, Any]], add]


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario."""
    return {
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        "final_answer": None,
        "pending_question": None,
        "proposed_action": None,
        "approval": None,
        "evaluation_result": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: Any) -> dict[str, Any]:
    """Create a normalized event payload."""
    return LabEvent(node=node, event_type=event_type, message=message, metadata=metadata).model_dump()
