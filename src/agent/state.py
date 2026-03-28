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

    # Cached embedding (computed once in triage, reused in context_fetch) 
    # Avoids a full model forward pass on every node that needs vector similarity.
    error_vector: list[float] | None

    # Ghost Stack Trace (sanitized locally, safe for LLM transmission) 
    # Produced by StackTraceSanitizer — all identifiers hashed, line numbers preserved.
    ghost_stack_trace: dict | None

    # Exact CPG node identified by line number from stack trace 
    # When set, context_fetch uses this as primary anchor instead of Qdrant top-1.
    exact_node_id: str | None

    # Error fingerprint (structured, local-only ID) 
    error_fingerprint: str | None    # short SHA-256 hex, e.g. "a3f7c211b9d0"
    error_fingerprint_key: str | None  # human-readable key, e.g. "401::POST /verify-token::HTTPException"

    # Solution Library (Case-Based Reasoning) 
    # Set when triage finds a matching or similar solution.
    solution_library_hit: dict | None  # Solution.to_dict() if found
    solution_hit_type: str | None      # "exact" | "similar" | None

    # Context 
    similar_nodes: list[dict]    # Qdrant search results
    call_graph: dict             # FalkorDB subgraph {"nodes": [...], "edges": [...]}
    delta_history: list[dict]    # Recent delta manifests for temporal reasoning

    # Hypothesis 
    hypotheses: list[dict]       # [{"hypothesis": str, "affected_nodes": [...], "confidence": float, "reasoning": str}]

    # Refinement loop state 
    retry_count: int             # How many hypothesis refinement attempts have been made
    failed_hypotheses: list[dict]  # Previous low-confidence hypotheses (injected into retry prompts)

    # Rejection memory (loaded from ErrorFingerprinter at triage time) 
    rejection_memory: list[dict]   # Past human-rejected hypotheses for this error fingerprint

    # Validation 
    validation_results: list[dict]  # Per-hypothesis validation scores
    best_hypothesis: dict | None
    composite_confidence: float
    needs_refinement: bool       # Set by validation when confidence < REFINE threshold

    # HITL 
    hitl_approved: bool | None   # None = pending, True = approved, False = rejected
    hitl_feedback: str | None    # Optional rejection reason

    # Patch 
    hashed_patch: dict | None    # {"node_id": ..., "change_sketch": ...}
    reconstructed_diff: str | None  # Human-readable unified diff

    # Pipeline control 
    pipeline_status: str         # "running" | "skipped" | "awaiting_hitl" | "complete" | "failed"
    error_message: str | None    # If pipeline_status == "failed"
