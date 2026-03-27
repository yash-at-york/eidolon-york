"""
Eidolon - Agent Nodes (v3 - web-aware)
Five LangGraph node functions implementing the diagnostic pipeline.

Optimisations (v2): parallel triage, streaming hypothesis, parallel judge, tight token budgets.
Web (v3): each node emits structured events via event_hub for real-time dashboard visibility.
Events are no-ops if nothing is subscribed (zero overhead in CLI mode).
"""
from __future__ import annotations

import json
import logging
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from huggingface_hub import InferenceClient
from openai import OpenAI
from rich.console import Console
from rich.live import Live

import ghost_config as cfg
from src.agent.state import AgentState
from src.agent.tools import falkordb_traverse_tool, get_delta_history_tool, qdrant_search_tool
from src.cloud.embedder import embed_error_text
from src.cloud.vector_store import VectorStore

# Suppress noisy HTTP error messages from huggingface_hub.
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub.inference._client").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

console = Console()

# Event hub (lazy import - not required in CLI mode)
def _emit(event_type: str, **data) -> None:
    try:
        from src.web.events import event_hub
        event_hub.emit(event_type, **data)
    except Exception:
        pass


# ── Client factory ─────────────────────────────────────────────────────────────
# USE_OLLAMA=true  → OpenAI client pointed at local Ollama (http://localhost:11434/v1)
# USE_OLLAMA=false → HuggingFace InferenceClient via configured provider

_ollama_client: OpenAI | None = None
_hf_client: InferenceClient | None = None


def _get_ollama_client() -> OpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(
            base_url=cfg.OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama ignores the key; the SDK requires a non-empty value
        )
    return _ollama_client


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        if not cfg.HF_TOKEN:
            raise EnvironmentError(
                "HF_TOKEN not set. Add it to your .env file.\n"
                "Get your token from: https://huggingface.co/settings/tokens"
            )
        # provider=None → huggingface_hub 1.x defaults to "auto", which selects
        # the first working provider for the model from the user's HF preferences.
        # Explicit provider only used when HF_PROVIDER is set in .env.
        _hf_client = InferenceClient(
            token=cfg.HF_TOKEN,
        )
    return _hf_client


def _resolve_model(hf_model: str) -> str:
    """Return the model name for the active backend.

    In Ollama mode all three roles share the single pulled model.
    """
    return cfg.OLLAMA_MODEL if cfg.USE_OLLAMA else hf_model


def _chat(model: str, system: str, user: str, temperature: float = 0.2,
          max_tokens: int = 1024, top_p: float = 0.9) -> str:
    """Call the active LLM backend and return the assistant message.

    Raises RuntimeError on any API failure so callers can handle it cleanly.
    """
    resolved = _resolve_model(model)
    try:
        if cfg.USE_OLLAMA:
            response = _get_ollama_client().chat.completions.create(
                model=resolved,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            response = _get_hf_client().chat_completion(
                model=resolved,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        backend = f"ollama ({cfg.OLLAMA_BASE_URL})" if cfg.USE_OLLAMA else f"HuggingFace"
        short = str(exc).split("\n")[0][:120]
        raise RuntimeError(f"Inference failed [{resolved} via {backend}]: {short}") from exc


def _chat_stream(model: str, system: str, user: str, temperature: float = 0.3,
                 max_tokens: int = 1500, label: str = "Generating") -> str:
    """Streaming variant with a live token counter.

    Falls back silently to non-streaming if streaming fails.
    Raises RuntimeError only when both paths fail.
    """
    resolved = _resolve_model(model)
    full_text = ""
    token_count = 0

    try:
        if cfg.USE_OLLAMA:
            stream = _get_ollama_client().chat.completions.create(
                model=resolved,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stream=True,
            )
        else:
            stream = _get_hf_client().chat_completion(
                model=resolved,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
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

    except Exception:
        # Streaming failed - fall back to non-streaming; errors propagate as RuntimeError
        full_text = _chat(model, system, user, temperature=temperature, max_tokens=max_tokens)

    return full_text.strip()


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

# Node 1 - Triage

def triage_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan]🔍 Node 1: Triage[/]")
    _emit("node_start", node="triage", label="Triage", emoji="🔍")

    error_event = state.get("error_event", "")
    service = state.get("service_namespace")

    system_prompt = """You are a senior SRE triage specialist.
Classify the error into exactly one type: auth | db | type_mismatch | timeout | unknown
Assess severity: low | medium | high | critical

Respond ONLY with valid JSON (no extra text):
{"error_type": "<type>", "severity": "<severity>", "reasoning": "<one sentence>"}"""

    is_duplicate = False
    dedup_score = 0.0

    def _run_dedup():
        try:
            error_vec = embed_error_text(error_event)
            vs = VectorStore()
            similar = vs.search(error_vec, top_k=1, service_filter=service)
            if similar:
                top_score = similar[0].get("score", 0.0)
                if top_score >= cfg.DEDUP_SIMILARITY_THRESHOLD:
                    return True, top_score
        except Exception as e:
            console.print(f"  [dim]Dedup unavailable ({e})[/dim]")
        return False, 0.0

    def _run_classify():
        try:
            return _chat(
                cfg.HF_TRIAGE_MODEL, system_prompt,
                f"Error event:\n{error_event}",
                temperature=0.1, max_tokens=cfg.HF_TRIAGE_MAX_TOKENS, top_p=0.9,
            )
        except RuntimeError as exc:
            console.print(f"  [red]✗ Triage model unavailable — {exc}[/red]")
            return "{}"

    console.print("  [dim]→ Classifying + dedup check (parallel)…[/dim]")
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_dedup = pool.submit(_run_dedup)
        fut_classify = pool.submit(_run_classify)
        is_duplicate, dedup_score = fut_dedup.result()
        raw_classify = fut_classify.result()

    if is_duplicate:
        console.print(f"  [yellow]Dedup hit[/]: similarity={dedup_score:.2f}")
        _emit("duplicate", score=dedup_score)

    error_type, severity, reasoning = "unknown", "medium", ""
    try:
        result = _extract_json(raw_classify)
        error_type = result.get("error_type", "unknown")
        severity = result.get("severity", "medium")
        reasoning = result.get("reasoning", "")
        console.print(f"  Type: [yellow]{error_type}[/]  Severity: [red]{severity}[/]  Duplicate: [dim]{is_duplicate}[/]")
        console.print(f"  Reasoning: [dim]{reasoning}[/]")
    except Exception:
        console.print("  [dim]Triage classification unavailable — using defaults[/dim]")

    _emit("node_complete", node="triage", error_type=error_type,
          severity=severity, reasoning=reasoning, is_duplicate=is_duplicate)

    return {
        **state,
        "error_type": error_type,
        "severity": severity,
        "is_duplicate": is_duplicate,
        "pipeline_status": "skipped" if is_duplicate else "running",
        "skip_reason": f"Duplicate (score={dedup_score:.2f})" if is_duplicate else None,
    }

# Node 2 - Context Fetch

def context_fetch_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan]📡 Node 2: Context Fetch[/]")
    _emit("node_start", node="context_fetch", label="Context Fetch", emoji="📡")

    error_event = state.get("error_event", "")
    service = state.get("service_namespace")

    similar_nodes: list[dict] = []
    delta_history: list[dict] = []
    call_graph: dict = {"nodes": [], "edges": []}

    def _fetch_qdrant():
        return qdrant_search_tool(error_event, service=service)

    def _fetch_delta():
        return get_delta_history_tool(service or "default-svc")

    console.print("  [dim]→ Querying Qdrant + delta history in parallel...[/dim]")
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_qdrant = pool.submit(_fetch_qdrant)
        fut_delta = pool.submit(_fetch_delta)
        similar_nodes = fut_qdrant.result()
        delta_history = fut_delta.result()

    console.print(f"  [green]✔[/] Found {len(similar_nodes)} similar nodes")
    console.print(f"  [green]✔[/] Delta history: {len(delta_history)} recent manifests")

    # Emit ghost nodes for dashboard visualization
    for node in similar_nodes[:5]:
        _emit("ghost_node", data=node)

    if similar_nodes:
        top_node_id = similar_nodes[0].get("node_id", "")
        if top_node_id:
            console.print(f"  [dim]→ Traversing call graph from {top_node_id}...[/dim]")
            call_graph = falkordb_traverse_tool(top_node_id, service=service)
            console.print(f"  [green]✔[/] Graph: {len(call_graph['nodes'])} nodes, {len(call_graph['edges'])} edges")

    _emit("node_complete", node="context_fetch",
          similar_count=len(similar_nodes),
          graph_nodes=len(call_graph["nodes"]),
          graph_edges=len(call_graph["edges"]))

    return {
        **state,
        "similar_nodes": similar_nodes,
        "call_graph": call_graph,
        "delta_history": delta_history,
    }

# Node 3 - Hypothesis Generation

def hypothesis_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan]💡 Node 3: Hypothesis Generation[/]")
    _emit("node_start", node="hypothesis", label="Hypothesis", emoji="💡")

    error_event = state.get("error_event", "")
    error_type = state.get("error_type", "unknown")
    similar_nodes = state.get("similar_nodes", [])
    call_graph = state.get("call_graph", {})
    delta_history = state.get("delta_history", [])

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

    system_prompt = """You are a senior SRE diagnosing a runtime error from structural ghost code.
Ghost identifiers (h_XXXXXXXX) are hashed real function/variable names. Reason about their
structural properties: parameter types, call order, control flow - not their names.

Return exactly 2 hypotheses ranked by likelihood. Be concise.

Respond ONLY with valid JSON:
{
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
}"""

    user_prompt = (
        f"Error: {error_event}\n\n"
        f"Ghost context:\n{json.dumps(context_payload, indent=2)}"
    )

    hypotheses = []
    try:
        raw = _chat_stream(
            cfg.HF_DIAGNOSIS_MODEL, system_prompt, user_prompt,
            temperature=0.3, max_tokens=cfg.HF_DIAGNOSIS_MAX_TOKENS,
            label="Reasoning about structural cause",
        )
        result = _extract_json(raw)
        hypotheses = result.get("hypotheses", [])

        for h in hypotheses:
            console.print(
                f"  [cyan]H{h.get('rank', '?')}[/] ({h.get('confidence', 0):.0%}): "
                f"[white]{h.get('hypothesis', '')[:100]}[/]"
            )

        _emit("hypotheses", data=hypotheses)

    except RuntimeError as exc:
        console.print(f"  [red]✗ Diagnosis model unavailable — {exc}[/red]")
        hypotheses = [{
            "rank": 1, "hypothesis": "Diagnosis model unavailable",
            "affected_nodes": [], "root_cause": str(exc),
            "patch_sketch": "N/A", "confidence": 0.0,
        }]
        _emit("hypotheses", data=hypotheses)
    except Exception as exc:
        console.print(f"  [red]✗ Hypothesis parse error — {exc}[/red]")
        hypotheses = [{
            "rank": 1, "hypothesis": "Unable to parse hypothesis",
            "affected_nodes": [], "root_cause": str(exc),
            "patch_sketch": "N/A", "confidence": 0.0,
        }]
        _emit("hypotheses", data=hypotheses)

    _emit("node_complete", node="hypothesis", count=len(hypotheses))
    return {**state, "hypotheses": hypotheses}


# Node 4 - Validation

def validation_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan]✅ Node 4: Validation[/]")
    _emit("node_start", node="validation", label="Validation", emoji="✅")

    hypotheses = state.get("hypotheses", [])
    error_event = state.get("error_event", "")
    call_graph = state.get("call_graph", {})
    graph_node_ids = {n.get("node_id", "") for n in call_graph.get("nodes", [])}

    judge_system = """You are a code review expert validating a structural bug diagnosis.
Evaluate: does the hypothesis correctly identify the root cause? Is the patch sketch sound?

Respond ONLY with valid JSON (no extra text):
{"verdict": "accept|reject|uncertain", "score": 0.0, "reasoning": "<one sentence>"}"""

    def _judge_one(h: dict) -> dict:
        base_confidence = h.get("confidence", 0.3)
        affected_nodes = h.get("affected_nodes", [])
        structural_score = 1.0
        if affected_nodes and graph_node_ids:
            overlap = len(set(affected_nodes) & graph_node_ids) / len(affected_nodes)
            structural_score = max(0.4, overlap)

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
            judge_raw = _chat(cfg.HF_JUDGE_MODEL, judge_system, judge_user,
                              temperature=0.1, max_tokens=cfg.HF_JUDGE_MAX_TOKENS, top_p=0.9)
            jr = _extract_json(judge_raw)
            judge_score = float(jr.get("score", 0.5))
            verdict = jr.get("verdict", "uncertain")
            reasoning = jr.get("reasoning", "")
        except RuntimeError as exc:
            console.print(f"  [red]✗ Judge model unavailable — {exc}[/red]")
        except Exception:
            console.print(f"  [dim]Judge parse failed for H{h.get('rank', '?')} — using defaults[/dim]")

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

    console.print(f"  [dim]→ Judging {len(hypotheses)} hypothesis/es in parallel…[/dim]")
    validation_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, len(hypotheses))) as pool:
        futures = [pool.submit(_judge_one, h) for h in hypotheses]
        for fut in as_completed(futures):
            vr = fut.result()
            validation_results.append(vr)
            console.print(
                f"  Judge H{vr['hypothesis_rank']}: [yellow]{vr['verdict']}[/] "
                f"({vr['judge_score']:.0%}) → composite [bold]{vr['composite_confidence']:.0%}[/]"
                + (f" - {vr['reasoning'][:70]}" if vr["reasoning"] else "")
            )

    validation_results.sort(key=lambda x: x["composite_confidence"], reverse=True)
    best_vr = validation_results[0]
    best_score = best_vr["composite_confidence"]
    best_hypothesis = best_vr["hypothesis"]

    console.print(f"  Best: H{best_hypothesis.get('rank','?')} → [bold green]{best_score:.0%}[/]")

    if best_score >= cfg.CONFIDENCE_THRESHOLD_AUTO:
        gate_label, gate_color = "AUTO_PROPOSE", "green"
    elif best_score >= cfg.CONFIDENCE_THRESHOLD_WARN:
        gate_label, gate_color = "PROPOSE_WITH_WARNING", "yellow"
    else:
        gate_label, gate_color = "FLAG_FOR_HUMAN_REVIEW", "red"

    console.print(f"  Confidence gate: [{gate_color}]{gate_label}[/{gate_color}]")

    clean_results = [{k: v for k, v in vr.items() if k != "hypothesis"} for vr in validation_results]

    _emit("node_complete", node="validation",
          best_rank=best_hypothesis.get("rank", 0),
          composite_confidence=best_score,
          gate_label=gate_label)

    return {
        **state,
        "validation_results": clean_results,
        "best_hypothesis": best_hypothesis,
        "composite_confidence": best_score,
        "pipeline_status": "awaiting_hitl",
    }

# Node 5 - Patch Synthesis

def patch_synthesis_node(state: AgentState) -> AgentState:
    console.rule("[bold cyan]🔧 Node 5: Patch Synthesis[/]")
    _emit("node_start", node="patch_synthesis", label="Patch Synthesis", emoji="🔧")

    best_hypothesis = state.get("best_hypothesis")
    if not best_hypothesis:
        _emit("error", message="No hypothesis to patch")
        return {**state, "pipeline_status": "failed", "error_message": "No hypothesis to patch"}

    error_event = state.get("error_event", "")
    call_graph = state.get("call_graph", {})

    system_prompt = """You are generating a structural bug fix specification.
All identifiers are hashed (h_XXXXXXXX format) - use them exactly as given.
Generate a single, precise patch action.

Change types: INSERT_CALL | REORDER_CALLS | ADD_GUARD | CHANGE_RETURN | MODIFY_PARAM

Respond ONLY with valid JSON:
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

    user_prompt = (
        f"Error: {error_event}\n"
        f"Hypothesis: {best_hypothesis.get('hypothesis', '')}\n"
        f"Root cause: {best_hypothesis.get('root_cause', '')}\n"
        f"Patch sketch: {best_hypothesis.get('patch_sketch', '')}\n"
        f"Affected nodes: {json.dumps(best_hypothesis.get('affected_nodes', []))}\n"
        f"Call graph edges: {json.dumps(call_graph.get('edges', [])[:8])}"
    )

    patch: dict = {}
    try:
        raw = _chat(cfg.HF_DIAGNOSIS_MODEL, system_prompt, user_prompt,
                    temperature=0.1, max_tokens=512, top_p=0.9)
        result = _extract_json(raw)
        patch = result.get("patch", {})
        rationale = result.get("rationale", "")

        console.print(f"  Patch type: [cyan]{patch.get('change_type', '?')}[/]")
        console.print(f"  Target: [yellow]{patch.get('target_node', '?')}[/]")
        console.print(f"  Rationale: [dim]{rationale[:100]}[/]")

        _emit("patch_ready", patch=patch, rationale=rationale)

    except RuntimeError as exc:
        console.print(f"  [red]✗ Patch model unavailable — {exc}[/red]")
        patch = {"error": str(exc)}
        _emit("error", message="Patch synthesis failed: model unavailable")
    except Exception as exc:
        console.print(f"  [red]✗ Patch parse error — {exc}[/red]")
        patch = {"error": str(exc)}
        _emit("error", message="Patch synthesis failed: parse error")

    _emit("node_complete", node="patch_synthesis")
    return {**state, "hashed_patch": patch, "pipeline_status": "complete"}
