"""
Eidolon - Human-in-the-Loop Gate

ARCHITECTURE NOTE (v2 fix):
  LangGraph interrupt() pauses the graph mid-node and re-runs the ENTIRE node
  when Command(resume=...) is called. This caused the diagnosis panel to render
  twice (before the prompt AND during resume).

  Fix: hitl_gate() is now a PURE INTERRUPT NODE - no display logic here.
  All display (panel, confidence table, prompt) lives in graph.py:run_agent(),
  which runs exactly once. hitl_gate() only:
    1. Calls interrupt() to pause the graph (production path)
    2. Returns the human decision from the resume payload

  DEMO_MODE retains interactive input but also ONLY in graph.py.
"""
from __future__ import annotations

import ghost_config as cfg
from src.agent.state import AgentState


def hitl_gate(state: AgentState) -> AgentState:
    """
    Pure HITL interrupt node.

    Production path:
      - Calls LangGraph interrupt() → graph pauses here
      - When graph.py resumes with Command(resume={"approved": bool}),
        this function receives the decision and returns updated state
      - No display, no prompt - all UI is in graph.py

    Demo / non-interrupt path:
      - If pipeline_status is already "complete" (set before entering this node
        due to some edge case), pass through unchanged
    """
    # If already decided (shouldn't happen normally, but guard against it)
    if state.get("hitl_approved") is not None:
        return state

    from langgraph.types import interrupt
    decision = interrupt({
        "request": "human_review",
        "hypothesis": state.get("best_hypothesis"),
        "confidence": state.get("composite_confidence", 0.0),
    })

    approved = decision.get("approved", False) if isinstance(decision, dict) else bool(decision)
    feedback = decision.get("feedback", "") if isinstance(decision, dict) else ""

    return {
        **state,
        "hitl_approved": approved,
        "hitl_feedback": feedback,
        # Do NOT set pipeline_status here - patch_synthesis_node sets "complete",
        # and _after_hitl routes to END on rejection (pipeline_status stays "awaiting_hitl"
        # which is fine - graph.py checks hitl_approved, not pipeline_status, post-resume)
    }
