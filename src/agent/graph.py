"""
Eidolon -- LangGraph Agent Graph (v3 - web-aware)

Modes:
  CLI mode  (WEB_MODE=false): terminal Confirm.ask() for HITL
  Web mode  (WEB_MODE=true):  HITLBridge.wait_for_decision() via browser button click

CLI Usage:
    python src/agent/graph.py --event "401 Unauthorized on POST /x" --service svc

Web mode is activated by ghost_config.WEB_MODE=True, set automatically by the
web launcher (demo/launch_web.py).
"""
from __future__ import annotations

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

console = Console()


def _emit(event_type: str, **data) -> None:
    try:
        from src.web.events import event_hub
        event_hub.emit(event_type, **data)
    except Exception:
        pass


# Conditional edge functions 

def _after_triage(state: AgentState) -> str:
    if state.get("is_duplicate") or state.get("pipeline_status") == "skipped":
        return "end"
    return "context_fetch"


def _after_hitl(state: AgentState) -> str:
    if state.get("hitl_approved"):
        return "patch_synthesis"
    return "end"


# Graph builder 

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("triage",          triage_node)
    graph.add_node("context_fetch",   context_fetch_node)
    graph.add_node("hypothesis",      hypothesis_node)
    graph.add_node("validation",      validation_node)
    graph.add_node("hitl",            hitl_gate)
    graph.add_node("patch_synthesis", patch_synthesis_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", _after_triage,
                                {"end": END, "context_fetch": "context_fetch"})
    graph.add_edge("context_fetch",  "hypothesis")
    graph.add_edge("hypothesis",     "validation")
    graph.add_edge("validation",     "hitl")
    graph.add_conditional_edges("hitl", _after_hitl,
                                {"patch_synthesis": "patch_synthesis", "end": END})
    graph.add_edge("patch_synthesis", END)

    if _HAS_SQLITE:
        db_path = str(Path(cfg.MAPPER_DB_PATH).parent / ".ghost_checkpoints.db")
        checkpointer = SqliteSaver.from_conn_string(db_path)
    else:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# HITL display (CLI mode only) 

def _display_hitl_panel(state: AgentState) -> None:
    best_hypothesis = state.get("best_hypothesis", {})
    confidence = state.get("composite_confidence", 0.0)
    validation_results = state.get("validation_results", [])

    console.rule("[bold magenta] HITL Gate - Human Review Required[/]")
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


# Main entry 

def run_agent(error_event: str, service: str = "default-svc") -> dict:
    """
    Run the Eidolon pipeline end-to-end.

    CLI mode (default): terminal HITL prompt.
    Web mode (cfg.WEB_MODE=True): HITLBridge waits for browser button click.
    """
    graph = build_graph()
    thread_id = f"{service}-{hash(error_event) & 0xFFFF}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "error_event":       error_event,
        "service_namespace": service,
        "pipeline_status":   "running",
    }

    _emit("pipeline_start", event=error_event, service=service)
    t0 = time.perf_counter()

    # Phase 1: run until HITL interrupt 
    state = graph.invoke(initial_state, config=config)

    # Phase 2: HITL 
    if state.get("pipeline_status") == "awaiting_hitl":
        best_hypothesis = state.get("best_hypothesis", {})
        confidence = state.get("composite_confidence", 0.0)
        validation_results = state.get("validation_results", [])

        if cfg.WEB_MODE:
            # Web mode: emit HITL event and wait for browser decision
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
            # CLI mode: terminal prompt
            console.print()
            _display_hitl_panel(state)
            console.print()
            approved = Confirm.ask("  [bold]Approve patch synthesis?[/]", default=True)
            feedback = ""
            if not approved:
                feedback = console.input("[dim]  Optional rejection reason: [/]").strip()
                console.print("[yellow]  Patch rejected.[/]")

        # Phase 3: resume graph 
        console.print()
        state = graph.invoke(
            Command(resume={"approved": approved, "feedback": feedback}),
            config=config,
        )

    elapsed = time.perf_counter() - t0

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
        colour = "green" if elapsed < 60 else ("yellow" if elapsed < 120 else "red")
        console.print(Panel(
            f"[bold {colour}]MTTD: {elapsed:.1f}s[/]\n"
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
