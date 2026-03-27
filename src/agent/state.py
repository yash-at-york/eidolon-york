"""
Eidolon - Agent State
TypedDict schema for the LangGraph stateful pipeline.
Every node reads from and writes to this shared state object.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    # Input 
    error_event: str             # raw error string from CloudWatch / logs
    service_namespace: str       # e.g. "payment-svc"

    # Triage 
    error_type: str              # "auth" | "db" | "type_mismatch" | "timeout" | "unknown"
    is_duplicate: bool           # True if this error was seen recently
    severity: str                # "low" | "medium" | "high" | "critical"
    skip_reason: str | None      # Why the pipeline was skipped (duplicate / noise)

    # Context 
    similar_nodes: list[dict]    # Qdrant search results
    call_graph: dict             # FalkorDB subgraph {"nodes": [...], "edges": [...]}
    delta_history: list[dict]    # Recent delta manifests for temporal reasoning

    # Hypothesis 
    hypotheses: list[dict]       # [{"hypothesis": str, "affected_nodes": [...], "confidence": float, "reasoning": str}]

    # Validation 
    validation_results: list[dict]  # Per-hypothesis validation scores
    best_hypothesis: dict | None
    composite_confidence: float

    # HITL 
    hitl_approved: bool | None   # None = pending, True = approved, False = rejected
    hitl_feedback: str | None    # Optional rejection reason

    # Patch 
    hashed_patch: dict | None    # {"node_id": ..., "change_sketch": ...}
    reconstructed_diff: str | None  # Human-readable unified diff

    # Pipeline control 
    pipeline_status: str         # "running" | "skipped" | "awaiting_hitl" | "complete" | "failed"
    error_message: str | None    # If pipeline_status == "failed"
