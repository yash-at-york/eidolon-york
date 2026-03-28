"""
Eidolon - Agent Nodes (v4 - truly agentic)

Improvements over v3:
  1. Error embedding cached in state — computed once in triage, reused in context_fetch.
  2. Error fingerprinting (fast exact dedup before vector dedup).
  3. Rejection memory injected into hypothesis prompts.
  4. Error-type-aware retrieval strategy in context_fetch.
  5. ReAct inner loop in hypothesis_node — LLM can request additional context.
  6. Hypothesis refinement loop — validation can route back to hypothesis on low confidence.
  7. Structural score degenerate default fixed — empty graph → 0.5, not 1.0.
  8. Web dashboard events (no-op in CLI mode).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
import openai
from openai import RateLimitError, APIConnectionError, InternalServerError, APIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
from rich.console import Console
from rich.live import Live

import ghost_config as cfg
from src.agent.state import AgentState
from src.agent.tools import falkordb_traverse_tool, get_delta_history_tool, qdrant_search_tool
from src.cloud.embedder import embed_error_text
from src.cloud.vector_store import VectorStore
from src.core.error_fingerprint import ErrorFingerprinter
from src.core.stack_trace_sanitizer import StackTraceSanitizer
from src.core.solution_library import SolutionLibrary

console = Console()

# ── Event hub (lazy import - zero overhead in CLI mode) ───────────────────────

def _emit(event_type: str, **data) -> None:
    try:
        from src.web.events import event_hub
        event_hub.emit(event_type, **data)
    except Exception:
        pass

# ── LLM Client Management ─────────────────────────────────────────────────────

_hf_client: InferenceClient | None = None
_openai_client: openai.OpenAI | None = None

def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        if not cfg.HF_TOKEN:
            raise EnvironmentError(
                "HF_TOKEN not set. Add it to your .env file.\n"
                "Get your token from: https://huggingface.co/settings/tokens"
            )
        _hf_client = InferenceClient(token=cfg.HF_TOKEN)
    return _hf_client

def _ensure_openai_configured():
    global _openai_client
    if _openai_client is None:
        if not getattr(cfg, "OPENAI_API_KEY", ""):
            raise EnvironmentError("OPENAI_API_KEY not set in config.")
        _openai_client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)

def _is_retryable(e: Exception) -> bool:
    if isinstance(e, HfHubHTTPError) and getattr(e, "response", None) is not None:
        return e.response.status_code in (429, 503, 504)
    if isinstance(e, (RateLimitError, APIConnectionError, InternalServerError, APIError)):
        return True
    return False

@retry(wait=wait_exponential(multiplier=1, min=2, max=15), stop=stop_after_attempt(5), retry=retry_if_exception(_is_retryable))
def _chat(model: str, system: str, user: str, temperature: float = 0.2,
          max_tokens: int = 1024, top_p: float = 0.9) -> str:
    if getattr(cfg, "USE_OPENAI_MODELS", False):
        _ensure_openai_configured()
        response = _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            top_p=top_p,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    else:
        client = _get_hf_client()
        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content.strip()


@retry(wait=wait_exponential(multiplier=1, min=2, max=15), stop=stop_after_attempt(5), retry=retry_if_exception(_is_retryable))
def _chat_stream(model: str, system: str, user: str, temperature: float = 0.3,
                 max_tokens: int = 1500, label: str = "Generating", messages: list[dict] | None = None) -> str:
    full_text = ""
    token_count = 0

    if messages is None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    if getattr(cfg, "USE_OPENAI_MODELS", False):
        _ensure_openai_configured()
        try:
            stream = _openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=0.9,
                response_format={"type": "json_object"},
                stream=True
            )
            with Live(console=console, refresh_per_second=8) as live:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_text += delta
                    token_count += len(delta.split())
                    live.update(f"  [dim]{label}… {token_count} tokens[/dim]")
                    if token_count % 10 == 0:
                        _emit("stream_token", token_count=token_count)
        except Exception as e:
            if _is_retryable(e):
                raise e
            console.print(f"  [dim](stream unavailable: {e})[/dim]")
            full_text = _chat_messages(model, messages, temperature=temperature, max_tokens=max_tokens)
    else:
        client = _get_hf_client()
        try:
            stream = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stream=True,
            )
            with Live(console=console, refresh_per_second=8) as live:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_text += delta
                    token_count += len(delta.split())
                    live.update(f"  [dim]{label}… {token_count} tokens[/dim]")
                    if token_count % 10 == 0:
                        _emit("stream_token", token_count=token_count)
        except Exception as e:
            if _is_retryable(e):
                raise e
            console.print(f"  [dim](stream unavailable: {e})[/dim]")
            full_text = _chat(model, system, user, temperature=temperature, max_tokens=max_tokens)

    return full_text.strip()


def _chat_messages(model: str, messages: list[dict], temperature: float = 0.3,
                   max_tokens: int = 1024) -> str:
    """Multi-turn chat for the ReAct inner loop (no streaming, no response_format)."""
    if getattr(cfg, "USE_OPENAI_MODELS", False):
        _ensure_openai_configured()
        response = _openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    else:
        client = _get_hf_client()
        response = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()


def _extract_json(text: str) -> dict | list:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


# ── Node 1 — Triage ───────────────────────────────────────────────────────────

def triage_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan] Node 1: Triage[/]")
    _emit("node_start", node="triage", label="Triage")

    raw_event = state.get("error_event", "")
    service = state.get("service_namespace")

    # ── Accept structured event dict OR legacy string ─────────────────────────
    # Structured: {"error_message": "...", "traceback": [{...}]}
    # Legacy: "401 Unauthorized on POST /verify-token"
    is_structured = isinstance(raw_event, dict)
    error_string  = raw_event.get("error_message", str(raw_event)) if is_structured else raw_event

    # ── Step 1: Local stack trace sanitization (NEVER leaves machine in plaintext) ──
    ghost_stack_trace = None
    exact_node_id: str | None = None

    if is_structured and raw_event.get("traceback"):
        try:
            from src.core.mapper import GhostMapper
            mapper = GhostMapper(read_only=True)
            sanitizer = StackTraceSanitizer(mapper)
            gst = sanitizer.sanitize(raw_event)
            ghost_stack_trace = gst.to_dict()
            mapper.close()

            exc_class = gst.exception_class
            console.print(f"  [dim]Stack trace sanitized: {exc_class} in {len(gst.frames)} frame(s)[/]")

            # ── Emit sanitized trace to Ghost Mirror panel ────────────────────
            _emit("ghost_stack_trace", data=ghost_stack_trace)

            if gst.fault_frame and gst.fault_frame.ghost_function:
                exact_node_id = gst.fault_frame.ghost_function
                console.print(f"  [green]Exact node identified[/]: {exact_node_id} (line {gst.fault_frame.line_number})")
                # Emit fault node immediately so Ghost Mirror panel fills before context_fetch
                _emit("ghost_node", data={
                    "node_id": exact_node_id,
                    "node_type": "fault_frame",
                    "line_start": gst.fault_frame.line_number,
                    "structural_discriminators": {"kind": "fault_frame"},
                    "source": "stack_trace_exact",
                })
        except Exception as e:
            console.print(f"  [dim]Stack trace sanitization failed ({e}) — using string fallback[/dim]")
    elif isinstance(raw_event, str):
        try:
            from src.core.mapper import GhostMapper
            mapper = GhostMapper(read_only=True)
            sanitizer = StackTraceSanitizer(mapper)
            gst = sanitizer.sanitize_simple(raw_event)
            ghost_stack_trace = gst.to_dict()
            mapper.close()
            # Emit structural hints even for simple strings
            _emit("ghost_stack_trace", data=ghost_stack_trace)
        except Exception:
            pass

    # ── Step 2: Fingerprint the error ─────────────────────────────────────────
    fingerprinter = ErrorFingerprinter()
    fid, fkey = fingerprinter.fingerprint(error_string)
    console.print(f"  Fingerprint: [dim]{fkey}[/]  ID: [dim]{fid}[/]")

    exact_dup = fingerprinter.is_known_exact(fid, min_occurrences=2)
    if exact_dup:
        console.print(f"  [yellow]Fast dedup hit[/]: fingerprint {fid} seen before")

    rejection_memory = fingerprinter.get_rejections(fid, limit=3)
    if rejection_memory:
        console.print(f"  [dim]{len(rejection_memory)} past rejection(s) loaded[/]")
    fingerprinter.close()

    # ── Step 3: Solution Library lookup (exact + CBR similar) ─────────────────
    solution_hit = None
    solution_hit_type = None
    lib = SolutionLibrary()
    exact_solution = lib.retrieve_exact(fid)
    if exact_solution:
        solution_hit = exact_solution.to_dict()
        solution_hit_type = "exact"
        console.print(f"  [bold green]Solution library HIT (exact)[/]: {exact_solution.patch_type} (used {exact_solution.times_replayed}x, conf={exact_solution.confidence:.0%})")
    elif exact_node_id:
        # CBR: look for similar solutions by exception type + node
        exc_type = ghost_stack_trace.get("exception_type", "UnknownError") if ghost_stack_trace else "UnknownError"
        similar = lib.retrieve_similar(exc_type, exact_node_id, limit=1)
        if similar:
            solution_hit = similar[0].to_dict()
            solution_hit_type = "similar"
            console.print(f"  [cyan]Solution library HIT (similar)[/]: adapted from solution #{similar[0].solution_id}")
    lib.close()

    # ── Step 4: Embed the error ONCE and cache in state ───────────────────────
    is_duplicate = exact_dup
    dedup_score  = 1.0 if exact_dup else 0.0
    error_vector: list[float] | None = None

    system_prompt = """You are a senior SRE triage specialist.
Classify the error into exactly one type: auth | db | type_mismatch | timeout | unknown
Assess severity: low | medium | high | critical

Respond ONLY with valid JSON (no markdown wrappers or explanations outside the JSON):
{"error_type": "<type>", "severity": "<severity>", "reasoning": "<1 short punchy sentence>"}"""

    # Augment classification input with ghost stack trace hints if available
    classify_input = error_string
    if ghost_stack_trace:
        exc_class = ghost_stack_trace.get("exception_class", "")
        hints = ghost_stack_trace.get("structural_hints", {})
        classify_input = (
            f"{error_string}\n"
            f"Exception class: {exc_class}\n"
            f"Structural category: {hints.get('error_category', 'unknown')}"
        )

    def _run_embed_and_dedup():
        nonlocal is_duplicate, dedup_score
        if exact_dup:
            vec = embed_error_text(error_string)
            return vec, True, 1.0
        try:
            vec = embed_error_text(error_string)
            vs = VectorStore()
            similar = vs.search(vec, top_k=1, service_filter=service)
            if similar:
                top_score = similar[0].get("score", 0.0)
                if top_score >= cfg.DEDUP_SIMILARITY_THRESHOLD:
                    return vec, True, top_score
            return vec, False, 0.0
        except Exception as e:
            console.print(f"  [dim]Dedup/embed unavailable ({e})[/dim]")
            return None, False, 0.0

    def _run_classify():
        model = cfg.OPENAI_TRIAGE_MODEL if getattr(cfg, "USE_OPENAI_MODELS", False) else cfg.HF_TRIAGE_MODEL
        return _chat(
            model, system_prompt,
            f"Error event:\n{classify_input}",
            temperature=0.1, max_tokens=cfg.HF_TRIAGE_MAX_TOKENS, top_p=0.9,
        )

    console.print("  [dim] Classifying + embed+dedup (parallel)…[/dim]")
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_embed = pool.submit(_run_embed_and_dedup)
        fut_classify = pool.submit(_run_classify)
        error_vector, is_duplicate, dedup_score = fut_embed.result()
        raw_classify = fut_classify.result()

    if is_duplicate:
        console.print(f"  [yellow]Dedup hit[/]: similarity={dedup_score:.2f}")
        _emit("duplicate", score=dedup_score)

    # Override error_type from ghost stack trace structural hints if available
    error_type, severity, reasoning = "unknown", "medium", ""
    if ghost_stack_trace:
        hints = ghost_stack_trace.get("structural_hints", {})
        exc_to_type = {
            "auth": ["PermissionError", "HTTPException"],
            "db": ["ConnectionError"],
            "timeout": ["TimeoutError"],
            "type_mismatch": ["TypeError", "AttributeError"],
        }
        exc_class = ghost_stack_trace.get("exception_class", "")
        for etype, exc_list in exc_to_type.items():
            if exc_class in exc_list:
                error_type = etype
                break

    try:
        result = _extract_json(raw_classify)
        # LLM classification overrides only if stack trace didn't give us a definitive type
        if error_type == "unknown":
            error_type = result.get("error_type", "unknown")
        severity  = result.get("severity", "medium")
        reasoning = result.get("reasoning", "")
        console.print(f"  Type: [yellow]{error_type}[/]  Severity: [red]{severity}[/]  Duplicate: [dim]{is_duplicate}[/]")
        if ghost_stack_trace:
            console.print(f"  Exception class: [cyan]{ghost_stack_trace.get('exception_class', '')}[/]  Ghost message: [dim]{ghost_stack_trace.get('ghost_message', '')[:80]}[/]")
        console.print(f"  Reasoning: [dim]{reasoning}[/]")
    except Exception as e:
        console.print(f"[red]Triage parse failed: {e}[/]")

    _emit("node_complete", node="triage", error_type=error_type,
          severity=severity, reasoning=reasoning, is_duplicate=is_duplicate)

    return {
        **state,
        "error_event":          error_string,   # normalise to string in state
        "error_type":           error_type,
        "severity":             severity,
        "is_duplicate":         is_duplicate,
        "error_vector":         error_vector,
        "error_fingerprint":    fid,
        "error_fingerprint_key": fkey,
        "ghost_stack_trace":    ghost_stack_trace,
        "exact_node_id":        exact_node_id,
        "rejection_memory":     rejection_memory,
        "solution_library_hit": solution_hit,
        "solution_hit_type":    solution_hit_type,
        "retry_count":          state.get("retry_count", 0),
        "failed_hypotheses":    state.get("failed_hypotheses", []),
        "pipeline_status":      "skipped" if is_duplicate else "running",
        "skip_reason":          f"Duplicate (score={dedup_score:.2f})" if is_duplicate else None,
    }


# ── Context Retrieval Strategies (error-type-aware) ───────────────────────────

_TYPE_STRATEGIES: dict[str, dict] = {
    "auth": {
        "description": "authentication / authorisation failure",
        "qdrant_hint": "middleware guard auth check authorisation token JWT",
        "payload_filters": {},   # future: filter on role=auth_middleware
        "graph_direction": "callers",  # want to know what calls into the failing auth node
    },
    "db": {
        "description": "database connectivity or query failure",
        "qdrant_hint": "database query ORM connection transaction commit",
        "payload_filters": {},
        "graph_direction": "neighborhood",
    },
    "type_mismatch": {
        "description": "type contract violation",
        "qdrant_hint": "type cast conversion parameter return annotation",
        "payload_filters": {},
        "graph_direction": "neighborhood",
    },
    "timeout": {
        "description": "timeout or network latency",
        "qdrant_hint": "async await timeout network retry sleep wait",
        "payload_filters": {},
        "graph_direction": "neighborhood",
    },
    "unknown": {
        "description": "general runtime error",
        "qdrant_hint": "",
        "graph_direction": "neighborhood",
    },
}


# ── Node 2 — Context Fetch ────────────────────────────────────────────────────

def context_fetch_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan] Node 2: Context Fetch[/]")
    _emit("node_start", node="context_fetch", label="Context Fetch")

    error_event   = state.get("error_event", "")
    service       = state.get("service_namespace")
    error_type    = state.get("error_type", "unknown")
    exact_node_id = state.get("exact_node_id")       # set by triage from stack trace
    ghost_st      = state.get("ghost_stack_trace") or {}

    # ── IMPROVEMENT: Reuse cached vector; don't re-embed ──────────────────────
    cached_vector = state.get("error_vector")
    strategy = _TYPE_STRATEGIES.get(error_type, _TYPE_STRATEGIES["unknown"])

    # Override strategy from ghost stack trace structural hints if richer
    gst_hints = ghost_st.get("structural_hints", {})
    exc_class = ghost_st.get("exception_class", "")
    if exc_class and exc_class in _TYPE_STRATEGIES:
        strategy = _TYPE_STRATEGIES[exc_class]
        console.print(f"  [dim]Strategy overridden by exception class: {exc_class}[/]")
    else:
        console.print(f"  [dim]Strategy: {error_type} → {strategy['description']}[/]")

    similar_nodes: list[dict] = []
    delta_history: list[dict] = []
    call_graph: dict = {"nodes": [], "edges": []}

    def _fetch_qdrant():
        if cached_vector is not None:
            return VectorStore().search(
                query_vector=cached_vector,
                top_k=cfg.AGENT_CONTEXT_TOP_K,
                service_filter=service,
            )
        console.print("  [dim yellow]Vector cache miss — re-embedding (fallback)[/dim yellow]")
        return qdrant_search_tool(error_event, service=service)

    def _fetch_delta():
        return get_delta_history_tool(service or "default-svc")

    console.print("  [dim] Querying Qdrant (cached vector) + delta history in parallel...[/dim]")
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_qdrant = pool.submit(_fetch_qdrant)
        fut_delta  = pool.submit(_fetch_delta)
        similar_nodes = fut_qdrant.result()
        delta_history = fut_delta.result()

    console.print(f"  [green]✓[/] Found {len(similar_nodes)} similar nodes  [dim]({error_type} strategy)[/]")
    console.print(f"  [green]✓[/] Delta history: {len(delta_history)} recent manifests")

    for node in similar_nodes[:5]:
        _emit("ghost_node", data=node)

    # ── Graph traversal: use exact node if known, else Qdrant top-1 ───────────
    anchor_node_id = exact_node_id  # from stack trace line number (precise)
    anchor_source  = "stack_trace_exact"

    if not anchor_node_id and similar_nodes:
        anchor_node_id = similar_nodes[0].get("node_id", "")
        anchor_source  = "qdrant_top1"

    if anchor_node_id:
        console.print(f"  [dim]Graph traversal anchor: {anchor_node_id} (source: {anchor_source})[/dim]")
        call_graph = falkordb_traverse_tool(anchor_node_id, service=service)
        console.print(f"  [green]✓[/] Graph: {len(call_graph['nodes'])} nodes, {len(call_graph['edges'])} edges")

        # ── Ghost stack trace frame enrichment ────────────────────────────────
        # Add ghost frames as additional call graph context when exact node known
        if exact_node_id and ghost_st.get("frames"):
            ghost_frames = [
                {"node_id": f["ghost_function"], "line": f["line_number"],
                 "ghost_code": f["ghost_code"], "file": f["file_basename"]}
                for f in ghost_st["frames"]
                if f.get("ghost_function")
            ]
            call_graph["ghost_frames"] = ghost_frames
            call_graph["exception_type"] = ghost_st.get("exception_type", "")
            call_graph["ghost_message"]  = ghost_st.get("ghost_message", "")
            console.print(f"  [dim]+{len(ghost_frames)} ghost frame(s) from sanitized stack trace[/dim]")

    _emit("node_complete", node="context_fetch",
          similar_count=len(similar_nodes),
          graph_nodes=len(call_graph["nodes"]),
          graph_edges=len(call_graph["edges"]),
          anchor_source=anchor_source)

    return {
        **state,
        "similar_nodes": similar_nodes,
        "call_graph":    call_graph,
        "delta_history": delta_history,
    }


# ── Node 3 — Hypothesis Generation (with ReAct inner loop) ───────────────────

_REACT_SYSTEM = """\
You are a senior SRE diagnosing a runtime failure from structural ghost code.
Ghost identifiers (h_XXXXXXXX) are hashed. Reason on structural properties only
(parameter types, call order, control flow — NOT identifier names).

You operate in a TOOL-USE LOOP. Each turn you MUST respond with valid JSON in ONE of two forms:

FORM A — Request more context (use when initial context is insufficient):
{
  "action": "query",
  "tool": "qdrant_search" | "falkordb_traverse",
  "input": "<natural language search query OR node_id like h_XXXXXXXX>",
  "reason": "<why you need this additional context>"
}

FORM B — Finalize (use when you have enough context to form hypotheses):
{
  "action": "finalize",
  "hypotheses": [
    {
      "rank": 1,
      "hypothesis": "<concise one-line description>",
      "affected_nodes": ["h_XXXXXXXX"],
      "root_cause": "<structural analysis, 2-3 sentences>",
      "patch_sketch": "<what structural change is needed>",
      "confidence": 0.0
    }
  ]
}

Return EXACTLY 2 hypotheses ranked by likelihood when finalizing.
Respond ONLY with valid JSON — no markdown, no prose outside the JSON object.\
"""

def hypothesis_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan] Node 3: Hypothesis Generation[/]")
    _emit("node_start", node="hypothesis", label="Hypothesis")

    error_event      = state.get("error_event", "")
    error_type       = state.get("error_type", "unknown")
    similar_nodes    = state.get("similar_nodes", [])
    call_graph       = state.get("call_graph", {})
    delta_history    = state.get("delta_history", [])
    failed_hyps      = state.get("failed_hypotheses", [])
    rejection_mem    = state.get("rejection_memory", [])
    retry_count      = state.get("retry_count", 0)
    service          = state.get("service_namespace")

    # ── Build initial context payload ─────────────────────────────────────────
    context_payload = {
        "error_type": error_type,
        "similar_ghost_nodes": similar_nodes[:5],
        "call_graph_nodes": call_graph.get("nodes", [])[:8],
        "call_graph_edges": call_graph.get("edges", [])[:12],
        "recent_changes": [
            {"changed_nodes": d.get("changed_nodes", [])[:3], "git_sha": d.get("git_sha")}
            for d in delta_history[:2]
        ],
    }

    # ── Inject refinement context if this is a retry ──────────────────────────
    if retry_count > 0 and failed_hyps:
        context_payload["REFINEMENT_CONTEXT"] = {
            "note": f"This is retry #{retry_count}. Previous attempt(s) scored too low. Approach DIFFERENTLY.",
            "failed_attempts": [
                {
                    "hypothesis": h.get("hypothesis", ""),
                    "patch_sketch": h.get("patch_sketch", ""),
                    "why_rejected": "Low composite confidence score from LLM judge.",
                }
                for h in failed_hyps[-2:]
            ],
        }
        console.print(f"  [yellow]Retry #{retry_count}[/]: injecting {len(failed_hyps)} failed attempt(s) into prompt")

    # ── Inject rejection memory (human-rejected past attempts) ───────────────
    if rejection_mem:
        context_payload["AVOID_THESE_APPROACHES"] = [
            {
                "hypothesis": r["hypothesis"],
                "patch_sketch": r["patch_sketch"],
                "human_feedback": r["feedback"] or "Rejected without feedback",
            }
            for r in rejection_mem
        ]
        console.print(f"  [red]⚠[/] Injecting {len(rejection_mem)} past human rejection(s) as anti-patterns")

    initial_user_msg = (
        f"Error: {error_event}\n\n"
        f"Ghost context:\n{json.dumps(context_payload, indent=2)}"
    )

    # ── ReAct inner loop ──────────────────────────────────────────────────────
    model = cfg.OPENAI_DIAGNOSIS_MODEL if getattr(cfg, "USE_OPENAI_MODELS", False) else cfg.HF_DIAGNOSIS_MODEL
    messages: list[dict] = [
        {"role": "system", "content": _REACT_SYSTEM},
        {"role": "user",   "content": initial_user_msg},
    ]

    hypotheses: list[dict] = []
    max_iters = cfg.REACT_MAX_ITERATIONS

    for iteration in range(max_iters):
        console.print(f"  [dim]ReAct iteration {iteration + 1}/{max_iters}…[/dim]")
        _emit("stream_token", token_count=0)

        try:
            # Use streaming for final iteration (gives live feedback), non-streaming for tool calls
            is_last = (iteration == max_iters - 1)
            if is_last:
                raw = _chat_stream(
                    model,
                    _REACT_SYSTEM,
                    "",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=cfg.HF_DIAGNOSIS_MAX_TOKENS,
                    label=f"Reasoning (iter {iteration+1})",
                )
            else:
                raw = _chat_messages(model, messages, temperature=0.3, max_tokens=cfg.HF_DIAGNOSIS_MAX_TOKENS)

        except Exception as e:
            console.print(f"[red]LLM call failed at iteration {iteration+1}: {e}[/red]")
            break

        parsed = _extract_json(raw)
        action = parsed.get("action", "finalize") if isinstance(parsed, dict) else "finalize"

        if action == "finalize":
            hypotheses = parsed.get("hypotheses", []) if isinstance(parsed, dict) else []
            console.print(f"  [green]✓[/] Finalized after {iteration + 1} iteration(s)")
            break

        elif action == "query":
            tool_name = parsed.get("tool", "qdrant_search")
            tool_input = parsed.get("input", error_event)
            reason = parsed.get("reason", "")
            console.print(f"  [cyan]Tool call:[/] {tool_name}({tool_input!r:.60s}…)  [dim]{reason}[/]")

            # Execute the requested tool
            try:
                if tool_name == "qdrant_search":
                    from src.agent.tools import qdrant_search_tool
                    tool_result = qdrant_search_tool(tool_input, service=service)
                    tool_result_str = json.dumps(tool_result[:3], indent=2)  # limit to top-3
                elif tool_name == "falkordb_traverse":
                    from src.agent.tools import falkordb_traverse_tool
                    tool_result = falkordb_traverse_tool(tool_input, service=service)
                    tool_result_str = json.dumps(tool_result, indent=2)
                else:
                    tool_result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})
            except Exception as e:
                tool_result_str = json.dumps({"error": str(e)})
                console.print(f"  [dim red]Tool error: {e}[/dim red]")

            # Append exchange to message history for next iteration
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{tool_result_str}\n\nContinue reasoning. Use FORM B (finalize) when ready.",
            })

        else:
            # Unexpected action — treat as finalize attempt
            console.print(f"  [dim]Unknown action '{action}' — treating as finalize[/dim]")
            hypotheses = parsed.get("hypotheses", []) if isinstance(parsed, dict) else []
            break

    # ── Parse and display ─────────────────────────────────────────────────────
    if not hypotheses:
        console.print("[red]Hypothesis generation produced no output[/red]")
        hypotheses = [{
            "rank": 1, "hypothesis": "Unable to generate hypothesis — see logs",
            "affected_nodes": [], "root_cause": "Generation failed or output format invalid.",
            "patch_sketch": "N/A", "confidence": 0.0,
        }]

    for h in hypotheses:
        console.print(
            f"  [cyan]H{h.get('rank', '?')}[/] ({h.get('confidence', 0):.0%}): "
            f"[white]{h.get('hypothesis', '')[:120]}[/]"
        )

    _emit("hypotheses", data=hypotheses)
    _emit("node_complete", node="hypothesis", count=len(hypotheses))

    return {**state, "hypotheses": hypotheses}


# ── Node 4 — Validation ───────────────────────────────────────────────────────

def validation_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan] Node 4: Validation[/]")
    _emit("node_start", node="validation", label="Validation")

    hypotheses      = state.get("hypotheses", [])
    error_event     = state.get("error_event", "")
    call_graph      = state.get("call_graph", {})
    graph_node_ids  = {n.get("node_id", "") for n in call_graph.get("nodes", [])}
    retry_count     = state.get("retry_count", 0)

    judge_system = """\
You are a code review expert validating a structural bug diagnosis.
Evaluate: does the hypothesis correctly identify the root cause? Is the patch sketch sound?

Respond ONLY with valid JSON (no markdown wrappers or explanations outside the JSON):
{"verdict": "accept|reject|uncertain", "score": 0.0, "reasoning": "<1 short punchy sentence>"}"""

    def _judge_one(h: dict) -> dict:
        base_confidence = h.get("confidence", 0.3)
        affected_nodes  = h.get("affected_nodes", [])

        # ── FIX: Degenerate structural score ─────────────────────────────────
        # Old code: if not graph_node_ids → score stays 1.0 (silent inflation).
        # New: empty call graph is genuinely unknown → 0.5 (neutral, slight penalty).
        if not graph_node_ids:
            structural_score = 0.5   # no graph data = uncertain, not perfect
        elif not affected_nodes:
            structural_score = 0.5   # hypothesis has no affected nodes = suspicious
        else:
            overlap = len(set(affected_nodes) & graph_node_ids) / len(affected_nodes)
            structural_score = max(0.3, overlap)  # floor at 0.3, never 0

        judge_user = (
            f"Error: {error_event}\n"
            f"Hypothesis: {h.get('hypothesis', '')}\n"
            f"Root cause: {h.get('root_cause', '')}\n"
            f"Patch sketch: {h.get('patch_sketch', '')}"
        )
        judge_score = 0.5
        verdict = "uncertain"
        reasoning = ""
        try:
            model = cfg.OPENAI_JUDGE_MODEL if getattr(cfg, "USE_OPENAI_MODELS", False) else cfg.HF_JUDGE_MODEL
            judge_raw = _chat(model, judge_system, judge_user,
                              temperature=0.1, max_tokens=cfg.HF_JUDGE_MAX_TOKENS, top_p=0.9)
            jr = _extract_json(judge_raw)
            judge_score = float(jr.get("score", 0.5))
            verdict = jr.get("verdict", "uncertain")
            reasoning = jr.get("reasoning", "")
        except Exception as e:
            console.print(f"  [dim]Judge failed for H{h.get('rank','?')}: {e}[/dim]")

        composite = (0.4 * base_confidence) + (0.35 * judge_score) + (0.25 * structural_score)
        return {
            "hypothesis": h,
            "hypothesis_rank": h.get("rank", 0),
            "base_confidence": base_confidence,
            "judge_score": judge_score,
            "structural_score": structural_score,
            "composite_confidence": composite,
            "verdict": verdict,
            "reasoning": reasoning,
        }

    console.print(f"  [dim] Judging {len(hypotheses)} hypothesis/es in parallel…[/dim]")
    validation_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, len(hypotheses))) as pool:
        futures = [pool.submit(_judge_one, h) for h in hypotheses]
        for fut in as_completed(futures):
            vr = fut.result()
            validation_results.append(vr)
            console.print(
                f"  Judge H{vr['hypothesis_rank']}: [yellow]{vr['verdict']}[/] "
                f"({vr['judge_score']:.0%})  structural [{vr['structural_score']:.0%}]  "
                f"composite [bold]{vr['composite_confidence']:.0%}[/]"
                + (f" - {vr['reasoning'][:70]}" if vr["reasoning"] else "")
            )

    validation_results.sort(key=lambda x: x["composite_confidence"], reverse=True)
    best_vr         = validation_results[0]
    best_score      = best_vr["composite_confidence"]
    best_hypothesis = best_vr["hypothesis"]

    console.print(f"  Best: H{best_hypothesis.get('rank','?')}  [bold green]{best_score:.0%}[/]")

    if best_score >= cfg.CONFIDENCE_THRESHOLD_AUTO:
        gate_label, gate_color = "AUTO_PROPOSE", "green"
    elif best_score >= cfg.CONFIDENCE_THRESHOLD_WARN:
        gate_label, gate_color = "PROPOSE_WITH_WARNING", "yellow"
    elif best_score >= cfg.CONFIDENCE_THRESHOLD_REFINE:
        gate_label, gate_color = "FLAG_FOR_HUMAN_REVIEW", "red"
    else:
        gate_label, gate_color = "REFINE_HYPOTHESES", "magenta"

    console.print(f"  Confidence gate: [{gate_color}]{gate_label}[/{gate_color}]")

    # ── Decide if we need refinement ──────────────────────────────────────────
    needs_refinement = (
        best_score < cfg.CONFIDENCE_THRESHOLD_REFINE
        and retry_count < cfg.MAX_HYPOTHESIS_RETRIES
    )

    if needs_refinement:
        console.print(
            f"  [magenta]Score {best_score:.0%} < refine threshold {cfg.CONFIDENCE_THRESHOLD_REFINE:.0%}[/]\n"
            f"  [dim]Retry {retry_count + 1}/{cfg.MAX_HYPOTHESIS_RETRIES} — routing back to hypothesis node[/]"
        )

    clean_results = [{k: v for k, v in vr.items() if k != "hypothesis"} for vr in validation_results]

    _emit("node_complete", node="validation",
          best_rank=best_hypothesis.get("rank", 0),
          composite_confidence=best_score,
          gate_label=gate_label,
          needs_refinement=needs_refinement)

    # Accumulate failed hypotheses for the refinement prompt
    current_failed = state.get("failed_hypotheses", [])
    updated_failed = current_failed + (
        [best_hypothesis] if needs_refinement else []
    )

    return {
        **state,
        "validation_results": clean_results,
        "best_hypothesis": best_hypothesis,
        "composite_confidence": best_score,
        "needs_refinement": needs_refinement,
        "retry_count": retry_count + (1 if needs_refinement else 0),
        "failed_hypotheses": updated_failed,
        "pipeline_status": "awaiting_hitl",
    }


# ── Node 5 — Patch Synthesis ──────────────────────────────────────────────────

def patch_synthesis_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan] Node 5: Patch Synthesis[/]")
    _emit("node_start", node="patch_synthesis", label="Patch Synthesis")

    best_hypothesis = state.get("best_hypothesis")
    if not best_hypothesis:
        _emit("error", message="No hypothesis to patch")
        return {**state, "pipeline_status": "failed", "error_message": "No hypothesis to patch"}

    error_event = state.get("error_event", "")
    call_graph  = state.get("call_graph", {})
    ghost_st    = state.get("ghost_stack_trace") or {}

    system_prompt = """\
You are generating a structural bug fix specification.
All identifiers are hashed (h_XXXXXXXX format) - use them exactly as given.
Generate a single, precise patch action.

Change types: INSERT_CALL | REORDER_CALLS | ADD_GUARD | CHANGE_RETURN | MODIFY_PARAM | ADD_IMPORT

If the diagnostic shows a NameError (called but not imported), use ADD_IMPORT.
If the diagnostic shows an AttributeError (wrong method), use MODIFY_PARAM or REORDER_CALLS.
If the diagnostic shows a missing guard, use ADD_GUARD.

Respond ONLY with valid JSON (no markdown wrappers or explanations outside the JSON):
{
  "patch": {
    "target_node": "h_XXXXXXXX",
    "change_type": "<type>",
    "description": "<what changes structurally>",
    "insert_before": "h_XXXXXXXX or null",
    "new_logic": "<structural pseudocode with hashed IDs and type names>"
  },
  "rationale": "<why this resolves the error>"
}"""

    # Enrich user prompt with ghost stack trace for precise diagnosis
    stack_context = ""
    if ghost_st.get("fault_frame"):
        ff = ghost_st["fault_frame"]
        stack_context = (
            f"Exception: {ghost_st.get('exception_type', '')} — {ghost_st.get('ghost_message', '')}\n"
            f"Fault location: line {ff.get('line_number', '?')} in {ff.get('ghost_function', '?')}\n"
            f"Fault code: {ff.get('ghost_code', '')}\n"
        )

    user_prompt = (
        f"Error: {error_event}\n"
        f"{stack_context}"
        f"Hypothesis: {best_hypothesis.get('hypothesis', '')}\n"
        f"Root cause: {best_hypothesis.get('root_cause', '')}\n"
        f"Patch sketch: {best_hypothesis.get('patch_sketch', '')}\n"
        f"Affected nodes: {json.dumps(best_hypothesis.get('affected_nodes', []))}\n"
        f"Call graph edges: {json.dumps(call_graph.get('edges', [])[:8])}"
    )

    patch = {}
    rationale = ""
    try:
        model = cfg.OPENAI_DIAGNOSIS_MODEL if getattr(cfg, "USE_OPENAI_MODELS", False) else cfg.HF_DIAGNOSIS_MODEL
        raw = _chat(model, system_prompt, user_prompt,
                    temperature=0.1, max_tokens=512, top_p=0.9)
        result    = _extract_json(raw)
        patch     = result.get("patch", {})
        rationale = result.get("rationale", "")

        console.print(f"  Patch type: [cyan]{patch.get('change_type', '?')}[/]")
        console.print(f"  Target: [yellow]{patch.get('target_node', '?')}[/]")
        console.print(f"  Rationale: [dim]{rationale[:500]}[/]")

        _emit("patch_ready", patch=patch, rationale=rationale)

    except Exception as e:
        console.print(f"[red]Patch synthesis failed: {e}[/]")
        patch = {"error": str(e)}
        _emit("error", message=f"Patch synthesis failed: {e}")

    _emit("node_complete", node="patch_synthesis")
    return {**state, "hashed_patch": patch, "pipeline_status": "complete"}
