"""
Eidolon - Agent Tools
LangGraph-compatible tool functions used by agent nodes.
Tools are pure functions - they query the vector/graph stores and return dicts.
"""
from __future__ import annotations

import json
from pathlib import Path

from ghost_config import (
    AGENT_CONTEXT_TOP_K,
    AGENT_GRAPH_DEPTH,
    DEMO_MODE,
)
from src.cloud.embedder import embed_error_text, embed_node
from src.cloud.graph_store import GraphStore
from src.cloud.vector_store import VectorStore


# Module-level store singletons
_vector_store: VectorStore | None = None
_graph_store: GraphStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_graph_store() -> GraphStore:
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store



# Tool 1: Semantic search on ghost nodes


def qdrant_search_tool(query: str, service: str | None = None) -> list[dict]:
    """
    Search the Qdrant vector store for ghost nodes semantically similar to `query`.
    query: a natural-language description of the failing behaviour or error trace.
    Returns: top-k similar ghost nodes with scores.
    """
    query_vec = embed_error_text(query)
    results = _get_vector_store().search(
        query_vector=query_vec,
        top_k=AGENT_CONTEXT_TOP_K,
        service_filter=service,
    )
    return results



# Tool 2: Graph traversal for causal neighborhood


def falkordb_traverse_tool(node_id: str, service: str | None = None) -> dict:
    """
    Return the N-hop call-graph neighbourhood of node_id.
    node_id: a hashed function identifier (h_XXXXXXXX).
    Returns: {"nodes": [...], "edges": [...]}
    """
    namespaced = f"{service}::{node_id}" if service else node_id
    return _get_graph_store().traverse(namespaced, depth=AGENT_GRAPH_DEPTH)



# Tool 3: Temporal delta history


def get_delta_history_tool(service: str, limit: int = 5) -> list[dict]:
    """
    Return the most recent delta manifests for a service.
    Used for temporal reasoning: "what changed in the last N commits?"
    In demo mode, reads from demo/last_manifest.json.
    """
    if DEMO_MODE:
        manifest_path = Path("demo") / "last_manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
                if data.get("service") == service or not service:
                    return [data]
            except Exception:
                pass
        return []

    # Production: query the NATS JetStream history or a manifest log DB
    # For V1, this returns an empty list - to be implemented in Phase 2
    return []
