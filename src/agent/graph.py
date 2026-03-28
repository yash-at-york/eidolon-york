"""
Eidolon -- LangGraph Agent Graph (v4 - truly agentic)

New in v4:
  - Hypothesis refinement loop: validation → hypothesis when confidence is too low.
  - Rejection memory: on HITL reject, records the hypothesis + feedback via ErrorFingerprinter.
  - Outcome tracking: records approved/rejected outcome for pattern statistics.

Modes:
  CLI mode  (WEB_MODE=false): terminal Confirm.ask() for HITL
  Web mode  (WEB_MODE=true):  HITLBridge.wait_for_decision() via browser button click

CLI Usage:
    python src/agent/graph.py --event "401 Unauthorized on POST /x" --service svc
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE = True
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
    _HAS_SQLITE = False

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

import ghost_config as cfg
from src.agent.hitl import hitl_gate
from src.agent.nodes import (
    context_fetch_node,
    hypothesis_node,
    patch_synthesis_node,
    triage_node,
    validation_node,
)
from src.agent.reconstructor import PatchReconstructor
from src.agent.state import AgentState
from src.core.solution_library import SolutionLibrary

console = Console()


def _emit(event_type: str, **data) -> None:
    try:
        from src.web.events import event_hub
        event_hub.emit(event_type, **data)
    except Exception:
        pass


# ── Conditional edge functions ────────────────────────────────────────────────

def _after_triage(state: AgentState) -> str:
    if state.get("is_duplicate") or state.get("pipeline_status") == "skipped":
        return "end"
    # Fast-path: exact solution from library — skip hypothesis chain
    if state.get("solution_hit_type") == "exact":
        return "solution_replay"
    return "context_fetch"


def _after_validation(state: AgentState) -> str:
    """
    Route after validation:
      - If confidence is too low AND we haven't exceeded MAX_HYPOTHESIS_RETRIES:
        loop back to hypothesis_node for a refined attempt.
      - Otherwise: proceed to HITL gate.
    """
    if state.get("needs_refinement", False):
        return "hypothesis"
    return "hitl"


def _after_hitl(state: AgentState) -> str:
    if state.get("hitl_approved"):
        return "patch_synthesis"
    return "end"


# ── Solution Replay Node ────────────────────────────────────────────────

def solution_replay_node(state: AgentState) -> AgentState:
    """
    Fast-path node for known errors.
    When triage finds an exact solution library hit, this node presents
    the stored solution directly at HITL, bypassing hypothesis+validation.
    In demo/CLI mode: prints a summary. In web mode: emits events.
    MTTD for known errors: ~5 seconds.
    """
    console.rule("[bold green] Solution Replay — Known Error Pattern[/]")
    sol = state.get("solution_library_hit", {})
    fkey = state.get("error_fingerprint_key", "")

    console.print(f"  [dim]Fingerprint:[/] {fkey}")
    console.print(f"  [green]Patch type:[/]  {sol.get('patch_type', '?')}")
    console.print(f"  [green]Confidence:[/]  {sol.get('confidence', 0):.0%}")
    console.print(f"  [dim]Used before:[/] {sol.get('times_replayed', 0)} time(s)")

    # Reconstruct the patch from stored sketch
    patch_sketch = {}
    try:
        import json
        patch_sketch = json.loads(sol.get("patch_sketch", "{}"))
    except Exception:
        pass

    _emit("solution_replay",
          fingerprint_key=fkey,
          patch_type=sol.get("patch_type"),
          confidence=sol.get("confidence"),
          times_replayed=sol.get("times_replayed"))

    return {
        **state,
        "best_hypothesis": {
            "hypothesis":    f"[REPLAY] Previously confirmed fix: {sol.get('patch_type', '?')}",
            "affected_nodes": [sol.get("target_node_id", "")],
            "root_cause":    "Same error fingerprint as a previously diagnosed and human-approved case.",
            "patch_sketch":  sol.get("patch_sketch", ""),
            "confidence":    sol.get("confidence", 0.0),
        },
        "hashed_patch":        patch_sketch,
        "composite_confidence": sol.get("confidence", 0.0),
        "pipeline_status":     "awaiting_hitl",
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("triage",           triage_node)
    graph.add_node("solution_replay",  solution_replay_node)
    graph.add_node("context_fetch",    context_fetch_node)
    graph.add_node("hypothesis",       hypothesis_node)
    graph.add_node("validation",       validation_node)
    graph.add_node("hitl",             hitl_gate)
    graph.add_node("patch_synthesis",  patch_synthesis_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", _after_triage,
                                {"end": END, "context_fetch": "context_fetch",
                                 "solution_replay": "solution_replay"})
    graph.add_edge("solution_replay",  "hitl")
    graph.add_edge("context_fetch",    "hypothesis")
    graph.add_edge("hypothesis",       "validation")

    # Refinement loop: validation can route back to hypothesis
    graph.add_conditional_edges("validation", _after_validation,
                                {"hypothesis": "hypothesis", "hitl": "hitl"})

    graph.add_conditional_edges("hitl", _after_hitl,
                                {"patch_synthesis": "patch_synthesis", "end": END})
    graph.add_edge("patch_synthesis", END)

    if _HAS_SQLITE:
        db_path = str(Path(cfg.MAPPER_DB_PATH).parent / ".ghost_checkpoints.db")
        checkpointer = SqliteSaver.from_conn_string(db_path)
    else:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer, interrupt_before=["hitl"])


# ── HITL display (CLI mode only) ──────────────────────────────────────────────

def _display_hitl_panel(state: AgentState) -> None:
    best_hypothesis    = state.get("best_hypothesis", {})
    confidence         = state.get("composite_confidence", 0.0)
    validation_results = state.get("validation_results", [])
    retry_count        = state.get("retry_count", 0)
    fkey               = state.get("error_fingerprint_key", "")

    console.rule("[bold magenta] HITL Gate - Human Review Required[/]")

    if fkey:
        console.print(f"  [dim]Error fingerprint: {fkey}[/]")
    if retry_count > 0:
        console.print(f"  [yellow]Note: This is a refined hypothesis (after {retry_count} retry attempt(s))[/]")

    if best_hypothesis:
        panel_text = (
            f"[bold]Hypothesis:[/] {best_hypothesis.get('hypothesis', 'N/A')}\n\n"
            f"[bold]Root Cause:[/] {best_hypothesis.get('root_cause', 'N/A')}\n\n"
            f"[bold]Patch Sketch:[/] {best_hypothesis.get('patch_sketch', 'N/A')}\n\n"
            f"[bold]Affected Nodes:[/] {', '.join(best_hypothesis.get('affected_nodes', []))}"
        )
        console.print(Panel(panel_text, title="AI Diagnosis", border_style="magenta"))
    else:
        console.print("[bold red]No hypothesis generated.[/]")

    if validation_results:
        table = Table(title="Confidence Breakdown")
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", justify="right")
        for vr in validation_results[:1]:
            table.add_row("Base confidence",  f"{vr.get('base_confidence', 0):.0%}")
            table.add_row("LLM-as-judge",     f"{vr.get('judge_score', 0):.0%}")
            table.add_row("Structural check", f"{vr.get('structural_score', 0):.0%}")
            table.add_row("[bold]Composite[/]", f"[bold green]{confidence:.0%}[/]")
        console.print(table)

    if confidence >= cfg.CONFIDENCE_THRESHOLD_AUTO:
        console.print("[bold green]HIGH CONFIDENCE - Auto-propose eligible[/]")
    elif confidence >= cfg.CONFIDENCE_THRESHOLD_WARN:
        console.print("[bold yellow]MEDIUM CONFIDENCE - Review recommended[/]")
    else:
        console.print("[bold red]LOW CONFIDENCE - Human review strongly recommended[/]")


# ── Main entry ────────────────────────────────────────────────────────────────

def run_agent(error_event: str, service: str = "default-svc") -> dict:
    """
    Run the Eidolon pipeline end-to-end.

    CLI mode (default): terminal HITL prompt.
    Web mode (cfg.WEB_MODE=True): HITLBridge waits for browser button click.
    """
    graph = build_graph()
    _event_key = json.dumps(error_event, sort_keys=True) if isinstance(error_event, dict) else str(error_event)
    thread_id = f"{service}-{hash(_event_key) & 0xFFFF}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "error_event":       error_event,
        "service_namespace": service,
        "pipeline_status":   "running",
        "retry_count":       0,
        "failed_hypotheses": [],
    }

    _emit("pipeline_start", event=error_event, service=service)
    t0 = time.perf_counter()
    hitl_start = 0.0
    hitl_elapsed = 0.0

    # Phase 1: run until HITL interrupt
    state = graph.invoke(initial_state, config=config)

    # Phase 2: HITL
    if state.get("pipeline_status") == "awaiting_hitl":
        hitl_start = time.perf_counter()
        best_hypothesis    = state.get("best_hypothesis", {})
        confidence         = state.get("composite_confidence", 0.0)
        validation_results = state.get("validation_results", [])
        fid                = state.get("error_fingerprint", "")
        error_type         = state.get("error_type", "unknown")

        if cfg.WEB_MODE:
            _emit("hitl_ready",
                  hypothesis=best_hypothesis,
                  confidence=confidence,
                  validation_results=validation_results)

            from src.web.hitl_bridge import hitl_bridge
            decision = hitl_bridge.wait_for_decision()
            approved = decision.get("approved", False)
            feedback = decision.get("feedback", "")

            _emit("hitl_decision", approved=approved, feedback=feedback)

        else:
            console.print()
            _display_hitl_panel(state)
            console.print()
            approved = Confirm.ask("  [bold]Approve patch synthesis?[/]", default=True)
            feedback = ""
            if not approved:
                feedback = console.input("[dim]  Optional rejection reason: [/]").strip()
                console.print("[yellow]  Patch rejected.[/]")

        # ── Record outcome to rejection memory + solution library ─────────────────
        if fid:
            try:
                from src.core.error_fingerprint import ErrorFingerprinter
                from src.core.solution_library import SolutionLibrary
                fp = ErrorFingerprinter()
                if not approved and best_hypothesis:
                    fp.record_rejection(fid, best_hypothesis, feedback)
                    console.print(f"  [dim]Rejection recorded to memory (fingerprint {fid})[/]")
                fp.record_outcome(fid, approved, confidence, error_type)
                fp.close()
            except Exception as e:
                console.print(f"  [dim]Memory record failed (non-critical): {e}[/]")

            # ── Store approved solution in solution library ───────────────────────
            if approved and state.get("hashed_patch") and state.get("hit_type") != "exact":
                try:
                    reconstructor = PatchReconstructor()
                    reconstructed = reconstructor.reconstruct_patch(state["hashed_patch"])
                    target_node = state["hashed_patch"].get("target_node", "")
                    if not target_node and best_hypothesis:
                        target_node = (best_hypothesis.get("affected_nodes") or [""])[0]

                    lib = SolutionLibrary()
                    sid = lib.store(
                        fingerprint_id    = fid,
                        exception_type    = error_type,
                        target_node_id    = target_node,
                        patch             = state["hashed_patch"],
                        reconstructed_diff = reconstructed,
                        confidence        = confidence,
                    )
                    lib.close()
                    console.print(f"  [green]Solution stored to library[/] (id={sid}, fingerprint={fid})")
                except Exception as e:
                    console.print(f"  [dim]Solution library store failed (non-critical): {e}[/]")

        # Phase 3: resume graph
        console.print()
        state = graph.invoke(
            Command(resume={"approved": approved, "feedback": feedback}),
            config=config,
        )
        hitl_elapsed = time.perf_counter() - hitl_start

    elapsed = (time.perf_counter() - t0) - hitl_elapsed

    # Patch reconstruction
    if state.get("hashed_patch") and state.get("hitl_approved"):
        reconstructor = PatchReconstructor()
        patch_text = reconstructor.reconstruct_patch(state["hashed_patch"])
        reconstructor.print_patch(state["hashed_patch"])
        _emit("patch_reconstructed", text=patch_text)

    # MTTD
    _emit("pipeline_complete",
          mttd_seconds=round(elapsed, 1),
          status=state.get("pipeline_status", "unknown"))

    if cfg.MTTD_MODE:
        retry_suffix = f" ({state.get('retry_count', 0)} refinement(s))" if state.get("retry_count", 0) > 0 else ""
        colour = "green" if elapsed < 60 else ("yellow" if elapsed < 120 else "red")
        console.print(Panel(
            f"[bold {colour}]MTTD: {elapsed:.1f}s[/]{retry_suffix}\n"
            f"From error event to proposed patch in [bold]{elapsed:.1f} seconds[/].",
            title="Mean Time To Diagnose", border_style=colour,
        ))

    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Eidolon - Agentic Debugger")
    parser.add_argument("--event",   required=True,         help="Error event string")
    parser.add_argument("--service", default="default-svc", help="Service namespace")
    args = parser.parse_args()
    result = run_agent(args.event, args.service)
    print(f"\nPipeline status: {result.get('pipeline_status', 'unknown')}")
