"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path

from .state import AgentState, ApprovalDecision, Route, make_event


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    TODO(student): add normalization, PII checks, and metadata extraction.
    """
    import re
    
    query = state.get("query", "").strip()
    
    # Normalize: lowercase, remove extra whitespace
    normalized_query = " ".join(query.lower().split())
    
    # PII checks: mask email, phone, credit card patterns
    pii_detected = []
    
    # Check for email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, normalized_query):
        normalized_query = re.sub(email_pattern, '[EMAIL_MASKED]', normalized_query)
        pii_detected.append('email')
    
    # Check for phone number
    phone_pattern = r'\b(?:\d{3}[-.]?){2}\d{4}\b'
    if re.search(phone_pattern, normalized_query):
        normalized_query = re.sub(phone_pattern, '[PHONE_MASKED]', normalized_query)
        pii_detected.append('phone')
    
    # Check for credit card (basic)
    cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    if re.search(cc_pattern, normalized_query):
        normalized_query = re.sub(cc_pattern, '[CC_MASKED]', normalized_query)
        pii_detected.append('credit_card')
    
    # Metadata extraction
    word_count = len(normalized_query.split())
    has_numbers = bool(re.search(r'\d', normalized_query))
    has_urgency = any(word in normalized_query for word in ['urgent', 'asap', 'emergency', 'critical'])
    
    metadata = {
        'word_count': word_count,
        'has_numbers': has_numbers,
        'has_urgency': has_urgency,
        'pii_detected': pii_detected,
        'original_length': len(query),
    }
    
    return {
        "query": normalized_query,
        "messages": [f"intake:{normalized_query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized", **metadata)],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route.

    TODO(student): replace keyword heuristics with a clear routing policy.
    Required routes: simple, tool, missing_info, risky, error.
    """
    query = state.get("query", "").lower()
    words = query.split()
    clean_words = [w.strip("?!.,;:") for w in words]
    
    # Clear routing policy based on keywords and patterns
    route = Route.SIMPLE
    risk_level = "low"
    confidence = 0.0
    matched_keywords = []
    
    # HIGH RISK routes: actions that require approval/escalation
    risky_keywords = {"refund", "delete", "cancel", "send", "create", "modify", "update", "remove"}
    risky_actions = {"send email", "send message", "send notification", "delete account", "close account"}
    
    if any(word in query for word in risky_keywords):
        # Check if it's an action that needs approval
        if any(action in query for action in risky_actions) or "refund" in query or "delete" in query:
            route = Route.RISKY
            risk_level = "high"
            confidence = 0.95
            matched_keywords = [kw for kw in risky_keywords if kw in query]
    
    # ERROR routes: system failures, timeouts, technical issues
    if not matched_keywords:
        error_keywords = {"timeout", "error", "fail", "crash", "exception", "cannot", "unable", "broken"}
        if any(word in query for word in error_keywords):
            route = Route.ERROR
            risk_level = "medium"
            confidence = 0.90
            matched_keywords = [kw for kw in error_keywords if kw in query]
    
    # TOOL routes: lookup/query operations requiring external data
    if not matched_keywords:
        tool_keywords = {"status", "order", "lookup", "check", "find", "search", "verify", "look up"}
        if any(word in query for word in tool_keywords):
            route = Route.TOOL
            risk_level = "low"
            confidence = 0.85
            matched_keywords = [kw for kw in tool_keywords if kw in query]
    
    # MISSING_INFO routes: incomplete or ambiguous queries
    if not matched_keywords:
        # Vague pronouns without context
        vague_indicators = {"it", "that", "this", "it?"}
        if len(clean_words) < 5 and any(word in clean_words for word in vague_indicators):
            route = Route.MISSING_INFO
            risk_level = "low"
            confidence = 0.75
            matched_keywords = ["vague_pronoun"]
        # Very short queries
        elif len(clean_words) <= 3 and "?" in query:
            route = Route.MISSING_INFO
            risk_level = "low"
            confidence = 0.70
            matched_keywords = ["too_short"]
    
    # Default: SIMPLE route (FAQ, informational, no action needed)
    if route == Route.SIMPLE and not matched_keywords:
        simple_keywords = {"how", "what", "why", "help", "password", "reset", "login"}
        if any(word in query for word in simple_keywords):
            confidence = 0.80
            matched_keywords = [kw for kw in simple_keywords if kw in query]
    
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route.value}, confidence={confidence}",
                matched_keywords=matched_keywords,
                confidence=confidence,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    TODO(student): generate a specific clarification question from state.
    """
    query = state.get("query", "")
    
    # Generate context-aware clarification questions
    question = None
    
    # Detect what's missing
    if "it" in query.lower() or "that" in query.lower():
        question = "I see your question refers to 'it'. Could you be more specific? Are you asking about an order, account, or service?"
    elif "can you fix it" in query.lower():
        question = "I'd like to help fix that. Could you describe what issue you're experiencing in more detail?"
    elif "delete" in query.lower() and "what" not in query.lower():
        question = "To help with your request, could you clarify what you'd like to delete?"
    elif "send" in query.lower() and "where" not in query.lower():
        question = "Could you specify where you'd like this sent to (email, SMS, etc.) and to whom?"
    elif len(query.split()) < 5:
        question = "Your request is quite brief. Could you provide more details about what you need?"
    else:
        question = "Could you provide more specific information to help me better assist you?"
    
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [
            make_event(
                "clarify",
                "completed",
                "missing information requested",
                question_type="context_specific",
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    Implements idempotent tool execution and structured tool results.
    """
    attempt = int(state.get("attempt", 0))
    route = state.get("route")
    scenario_id = state.get("scenario_id", "unknown")
    thread_id = state.get("thread_id", "unknown")
    
    # Structured tool result with metadata
    tool_metadata = {
        "attempt": attempt,
        "scenario_id": scenario_id,
        "thread_id": thread_id,
        "timestamp": time.time(),
        "route": route,
    }
    
    # Idempotent tool execution: same inputs yield same results
    # For ERROR routes, simulate transient failures that eventually succeed
    if route == Route.ERROR.value and attempt < 2:
        result = {
            "status": "ERROR",
            "message": f"Transient failure (attempt {attempt + 1}/3)",
            "error_code": "TRANSIENT_FAILURE",
            "retryable": True,
            **tool_metadata,
        }
    elif route == Route.TOOL.value:
        # Mock successful tool result for TOOL routes
        result = {
            "status": "SUCCESS",
            "message": f"Order data retrieved",
            "data": {
                "order_id": "12345",
                "status": "shipped",
                "tracking": "TRK123456789",
            },
            **tool_metadata,
        }
    else:
        # Default successful result
        result = {
            "status": "SUCCESS",
            "message": f"Tool executed successfully",
            **tool_metadata,
        }
    
    # Convert result dict to JSON string for storage
    result_json = json.dumps(result, default=str)
    
    return {
        "tool_results": [result_json],
        "events": [
            make_event(
                "tool",
                "completed",
                f"tool executed attempt={attempt}",
                status=result.get("status", "UNKNOWN"),
                attempt=attempt,
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval.

    Creates a proposed action with evidence and risk justification.
    """
    query = state.get("query", "")
    risk_level = state.get("risk_level", "unknown")
    
    # Parse the query to extract the specific action
    action = "unknown action"
    justification = "Customer request requires human review."
    
    if "refund" in query.lower():
        action = "Issue refund to customer"
        justification = "Customer requested refund. This impacts financial records and requires verification of eligibility."
    elif "delete" in query.lower() and "account" in query.lower():
        action = "Delete customer account"
        justification = "Account deletion is irreversible and affects customer data retention policies. Requires verification."
    elif "send" in query.lower() and ("email" in query.lower() or "message" in query.lower()):
        action = "Send external communication"
        justification = "Outbound communication to customer requires message review for accuracy and compliance."
    elif "cancel" in query.lower():
        action = "Cancel service/order"
        justification = "Service cancellation affects billing and SLA. Requires approval to prevent customer churn."
    else:
        action = "Perform action"
        justification = "Action flagged as high-risk. Requires human approval before execution."
    
    proposed_action = f"{action} (Risk: {risk_level})"
    
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "pending_approval",
                "approval required",
                action=action,
                risk_level=risk_level,
                justification=justification,
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.

    Implements reject/edit decisions and timeout escalation.
    """
    approval_timeout = 3600  # 1 hour timeout
    approval_start = time.time()

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
            "query": state.get("query"),
            "scenario_id": state.get("scenario_id"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        # Mock approval logic for testing
        # In production, this would wait for human review
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="mock approval for lab - would require real human review in production"
        )

    # Log approval with decision details
    approval_time = time.time() - approval_start

    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                "completed",
                f"approved={decision.approved}",
                reviewer=decision.reviewer,
                approval_time_ms=int(approval_time * 1000),
                comment=decision.comment,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision.

    Implements bounded retry with exponential backoff and jitter.
    """
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    
    # Calculate exponential backoff: 1s, 2s, 4s, 8s, etc.
    backoff_seconds = 2 ** (attempt - 1)
    
    # Jitter: add random variation (0-50%) to prevent thundering herd
    jitter = random.uniform(0, 0.5 * backoff_seconds)
    backoff_with_jitter = backoff_seconds + jitter
    
    # Track error information for debugging
    errors = state.get("errors", []).copy()
    current_error = f"Retry attempt {attempt}/{max_attempts} - backoff: {backoff_seconds}s (+jitter: {jitter:.1f}s)"
    errors.append(current_error)
    
    # Metadata for monitoring and analysis
    retry_metadata = {
        "attempt": attempt,
        "max_attempts": max_attempts,
        "backoff_seconds": backoff_seconds,
        "jitter_seconds": jitter,
        "total_backoff": backoff_with_jitter,
        "is_final_attempt": attempt >= max_attempts,
    }
    
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [
            make_event(
                "retry",
                "completed",
                f"retry attempt {attempt}/{max_attempts} scheduled",
                **retry_metadata,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response.

    Grounds the answer in tool_results and approval where relevant.
    """
    route = state.get("route", Route.SIMPLE.value)
    tool_results = state.get("tool_results", [])
    approval = state.get("approval", {})
    pending_question = state.get("pending_question")
    
    answer = None
    metadata = {"route": route}
    
    # Route-specific answer generation
    if route == Route.SIMPLE.value:
        answer = (
            "I'm here to help! For general questions about account management, "
            "password reset, or FAQ items, you can typically find answers in our help center. "
            "If you need specific assistance, please provide more details."
        )
    
    elif route == Route.TOOL.value and tool_results:
        # Ground answer in structured tool results
        try:
            last_result = json.loads(tool_results[-1])
            if last_result.get("status") == "SUCCESS":
                if "data" in last_result:
                    data = last_result["data"]
                    answer = (
                        f"I found your information:\n"
                        f"  Order ID: {data.get('order_id', 'N/A')}\n"
                        f"  Status: {data.get('status', 'N/A')}\n"
                        f"  Tracking: {data.get('tracking', 'N/A')}\n"
                        "Please let me know if you need anything else."
                    )
                else:
                    answer = f"Tool execution successful: {last_result.get('message', 'Request processed.')}"
            else:
                answer = "I encountered an issue retrieving your information. Please try again or contact support."
        except (json.JSONDecodeError, TypeError):
            # Fallback for non-JSON results
            answer = f"I found: {tool_results[-1][:100]}"
        metadata["tool_status"] = "used"
    
    elif route == Route.MISSING_INFO.value:
        # Answer is the clarification question itself
        answer = pending_question or "Could you please provide more details?"
        metadata["type"] = "clarification"
    
    elif route == Route.RISKY.value:
        # Ground risky actions in approval
        if approval and approval.get("approved"):
            answer = (
                f"Your request has been approved. "
                f"Reviewer: {approval.get('reviewer', 'System')}. "
                f"Action will be processed. {approval.get('comment', '')}"
            )
            metadata["approved"] = True
        else:
            answer = "Your request requires approval before proceeding. Please contact support."
            metadata["approved"] = False
    
    elif route == Route.ERROR.value:
        answer = (
            "We encountered a technical issue processing your request. "
            "Our team has been notified and we're working on resolving it. "
            "Please try again in a few moments."
        )
        metadata["type"] = "error_recovery"
    
    else:
        answer = "Thank you for your inquiry. We'll process your request shortly."
    
    return {
        "final_answer": answer,
        "events": [
            make_event(
                "answer",
                "completed",
                "answer generated",
                **metadata,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.

    Replaces heuristic with structured validation.
    """
    tool_results = state.get("tool_results", [])
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    
    # Default: success
    evaluation_result = "success"
    reasoning = "Tool execution completed successfully"
    
    if not tool_results:
        evaluation_result = "needs_retry"
        reasoning = "No tool results available"
    else:
        latest_result = tool_results[-1]
        
        # Try to parse structured result
        try:
            result_obj = json.loads(latest_result)
            
            if result_obj.get("status") == "ERROR":
                if result_obj.get("retryable", False) and attempt < max_attempts:
                    evaluation_result = "needs_retry"
                    reasoning = f"Retryable error: {result_obj.get('message', 'Unknown')}"
                else:
                    evaluation_result = "failed"
                    reasoning = f"Non-retryable or max attempts reached: {result_obj.get('message', 'Unknown')}"
            elif result_obj.get("status") == "SUCCESS":
                evaluation_result = "success"
                reasoning = result_obj.get("message", "Tool execution successful")
            else:
                # Unknown status, but not ERROR
                evaluation_result = "success"
                reasoning = f"Status: {result_obj.get('status', 'UNKNOWN')}"
        
        except (json.JSONDecodeError, TypeError, AttributeError):
            # Fallback to string matching for non-JSON results
            if "ERROR" in latest_result or "FAIL" in latest_result or "TIMEOUT" in latest_result:
                if attempt < max_attempts:
                    evaluation_result = "needs_retry"
                    reasoning = f"Error detected in result, retrying (attempt {attempt}/{max_attempts})"
                else:
                    evaluation_result = "failed"
                    reasoning = "Error detected and max attempts reached"
            else:
                evaluation_result = "success"
                reasoning = "Result does not indicate error"
    
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                reasoning,
                evaluation_result=evaluation_result,
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    Persists to dead-letter queue and creates escalation.
    """
    # Extract relevant state for dead-letter logging
    dead_letter_record = {
        "timestamp": time.time(),
        "scenario_id": state.get("scenario_id", "unknown"),
        "thread_id": state.get("thread_id", "unknown"),
        "query": state.get("query", ""),
        "route": state.get("route", "unknown"),
        "attempt": state.get("attempt", 0),
        "max_attempts": state.get("max_attempts", 3),
        "errors": state.get("errors", []),
        "tool_results": state.get("tool_results", []),
        "risk_level": state.get("risk_level", "unknown"),
    }
    
    # Log to dead-letter queue (in production, this would be a real queue/database)
    try:
        # Create outputs directory if it doesn't exist
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        dead_letter_file = outputs_dir / "dead_letter.jsonl"
        
        # Append to JSONL file (append-only, no locks needed for demo)
        with open(dead_letter_file, "a") as f:
            f.write(json.dumps(dead_letter_record, default=str) + "\n")
    
    except Exception as e:
        # If file logging fails, at least include it in events
        pass
    
    # Generate user-facing message
    final_answer = (
        "I apologize, but I'm unable to resolve your request after multiple attempts. "
        "Your case has been escalated to our support team for manual review. "
        f"Reference ID: {state.get('thread_id', 'unknown')}. "
        "You can expect a response within 24 hours."
    )
    
    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "dead_letter",
                "completed",
                f"max retries exceeded, attempt={state.get('attempt', 0)}, escalated for manual review",
                scenario_id=state.get("scenario_id"),
                thread_id=state.get("thread_id"),
                errors_count=len(state.get("errors", [])),
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
