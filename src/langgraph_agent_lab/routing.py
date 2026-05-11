"""Routing functions for conditional edges."""

from __future__ import annotations

from .state import AgentState, Route


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node.

    Handles unknown routes safely with validation and edge cases.
    """
    route = state.get("route", Route.SIMPLE.value)
    
    # Define the routing mapping
    mapping = {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",
        Route.DEAD_LETTER.value: "dead_letter",
        Route.DONE.value: "finalize",
    }
    
    # Validate route is valid enum value
    valid_routes = {r.value for r in Route}
    if route not in valid_routes:
        # Unknown route: log and fallback to SIMPLE
        print(f"WARNING: Unknown route '{route}', defaulting to '{Route.SIMPLE.value}'")
        return mapping[Route.SIMPLE.value]
    
    # Return mapped node name
    next_node = mapping.get(route, "answer")
    return next_node


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry, fallback, or dead-letter.

    Implements bounded retry with dead-letter escalation when max attempts exceeded.
    """
    # Extract attempt counters with validation
    try:
        attempt = int(state.get("attempt", 0))
        max_attempts = int(state.get("max_attempts", 3))
    except (ValueError, TypeError):
        # Invalid attempt count: escalate to dead-letter
        print("WARNING: Invalid attempt count, escalating to dead-letter")
        return "dead_letter"
    
    # Validate reasonable bounds
    if attempt < 0:
        print(f"WARNING: Negative attempt count {attempt}, resetting to 0")
        attempt = 0
    if max_attempts < 1:
        print(f"WARNING: max_attempts {max_attempts} < 1, setting to 3")
        max_attempts = 3
    
    # Bounded retry logic
    if attempt >= max_attempts:
        # Max attempts exceeded: escalate
        return "dead_letter"
    
    # Still have attempts left: retry tool
    return "tool"


def route_after_evaluate(state: AgentState) -> str:
    """Decide whether tool result is satisfactory or needs retry.

    This is the 'done?' check that enables retry loops - a key LangGraph advantage over LCEL.
    Replaces heuristic with structured evaluation validation.
    """
    evaluation_result = state.get("evaluation_result", "unknown")
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    
    # Structured routing based on evaluation result
    if evaluation_result == "needs_retry":
        # Tool result indicates failure and is retryable
        if attempt < max_attempts:
            # Still have attempts: retry
            return "retry"
        else:
            # Max attempts reached but result still bad: escalate
            return "dead_letter"
    
    elif evaluation_result == "success":
        # Tool result is satisfactory: proceed to answer
        return "answer"
    
    elif evaluation_result == "failed":
        # Tool result indicates non-retryable failure: escalate immediately
        return "dead_letter"
    
    else:
        # Unknown evaluation result (e.g., corrupted state)
        # Default to safe path: attempt to answer
        print(f"WARNING: Unknown evaluation_result '{evaluation_result}', proceeding to answer")
        return "answer"


def route_after_approval(state: AgentState) -> str:
    """Route based on approval decision.

    Supports approve, reject, and edit/request outcomes.
    """
    approval = state.get("approval") or {}
    
    # Extract approval decision with validation
    approved = approval.get("approved", False)
    approval_status = approval.get("status", "unknown")
    comment = approval.get("comment", "")
    
    # Route based on approval outcome
    if approved:
        # Approval granted: proceed to execute tool
        return "tool"
    
    elif approval_status == "rejected":
        # Approval explicitly rejected: ask for clarification or alternative
        print(f"Approval rejected: {comment}")
        return "clarify"
    
    elif approval_status == "edit_requested":
        # Reviewer requested clarification/editing: ask for more details
        print(f"Clarification requested: {comment}")
        return "clarify"
    
    elif approval_status == "timeout":
        # Approval window expired: escalate to dead-letter
        print(f"Approval timeout: request waiting too long")
        return "dead_letter"
    
    else:
        # Default: treat as not approved, ask for clarification
        print(f"Approval status '{approval_status}' not recognized, requesting clarification")
        return "clarify"

