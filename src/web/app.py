"""
Eidolon - Web Dashboard Server
FastAPI + WebSocket backend that drives the conference dashboard.

Endpoints:
  GET  /                     → serve the dashboard HTML
  WS   /ws                   → real-time pipeline event stream
  POST /api/run-pipeline     → start the diagnostic pipeline in a background thread
  POST /api/inject-bug       → inject demo bug into demo/app_test.py
  POST /api/restore          → restore demo/app_test.py from backup
  POST /api/hitl/decide      → {approved: bool, feedback: str} from browser button
  GET  /api/status           → system status

Run with:
  python demo/launch_web.py      ← preferred (handles Docker + seed)
  uvicorn src.web.app:app --reload --port 7433
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

# Ensure project root is on path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ghost_config as cfg
from src.web.events import event_hub
from src.web.hitl_bridge import hitl_bridge

app = FastAPI(title="Eidolon Dashboard", version="1.0")

# Serve CSS + JS from /static
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# State 
_pipeline_running = False
_pipeline_lock = threading.Lock()


# Startup 

@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    event_hub.set_loop(loop)
    # Signal web mode to the pipeline
    os.environ["GHOST_WEB_MODE"] = "true"


# Dashboard HTML 

_HTML_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    if _HTML_PATH.exists():
        return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Eidolon</h1><p>index.html not found</p>")


# WebSocket 

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    q = event_hub.subscribe()
    try:
        while True:
            event = await q.get()
            await ws.send_text(event)
    except (WebSocketDisconnect, Exception):
        event_hub.unsubscribe(q)


# Pipeline runner 

class RunRequest(BaseModel):
    event: Any = ""          # str OR structured dict {error_message, traceback}
    service: str = "demo-svc"


def _run_pipeline_thread(error_event: Any, service: str) -> None:
    """Runs in a background thread. Imports are local to avoid circular deps."""
    global _pipeline_running
    try:
        # Reload config so WEB_MODE=True is picked up
        import importlib
        import ghost_config
        importlib.reload(ghost_config)

        from src.agent.graph import run_agent
        hitl_bridge.reset()
        event_hub.clear_history()
        run_agent(error_event, service)
    except Exception as e:
        event_hub.emit("error", message=str(e))
    finally:
        with _pipeline_lock:
            _pipeline_running = False


@app.post("/api/run-pipeline")
async def run_pipeline(body: RunRequest):
    global _pipeline_running
    with _pipeline_lock:
        if _pipeline_running:
            return JSONResponse({"error": "Pipeline already running"}, status_code=409)
        _pipeline_running = True

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(body.event, body.service),
        daemon=True,
    )
    thread.start()
    return {"status": "started", "event": body.event, "service": body.service}


# HITL 

class HITLDecision(BaseModel):
    approved: bool
    feedback: str = ""


@app.post("/api/hitl/decide")
async def hitl_decide(body: HITLDecision):
    if not hitl_bridge.is_active:
        return JSONResponse({"error": "No active HITL gate"}, status_code=400)
    hitl_bridge.decide(approved=body.approved, feedback=body.feedback)
    return {"status": "decision_received", "approved": body.approved}


# Demo controls 

@app.post("/api/inject-bug")
async def inject_bug():
    result = subprocess.run(
        [sys.executable, str(ROOT / "demo" / "inject_bug.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    event_hub.emit("bug_injected", stdout=result.stdout.strip())
    return {"status": "injected", "output": result.stdout.strip()}


@app.post("/api/restore")
async def restore_bug():
    result = subprocess.run(
        [sys.executable, str(ROOT / "demo" / "inject_bug.py"), "--restore"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    event_hub.emit("bug_restored", stdout=result.stdout.strip())
    return {"status": "restored", "output": result.stdout.strip()}


@app.post("/api/shutdown")
async def shutdown_server():
    def kill_trigger():
        import time
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=kill_trigger, daemon=True).start()
    return {"status": "shutting down"}


@app.post("/api/wipe-memory")
async def wipe_memory():
    """
    Clears all stored solutions from the Solution Library.
    After wiping, the next occurrence of any previously known error will go
    through the full pipeline (triage → hypothesis → validation → HITL → patch)
    rather than instant replay.  The schema and rejection memory are preserved.
    """
    try:
        from src.core.solution_library import SolutionLibrary
        lib = SolutionLibrary()
        stats_before = lib.get_library_stats()
        lib.wipe_all()
        lib.close()
        event_hub.emit("memory_wiped", solutions_removed=stats_before["total_solutions"])
        return {
            "status": "wiped",
            "solutions_removed": stats_before["total_solutions"],
            "message": "Solution Library cleared. Next run will use full pipeline.",
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# Status 

@app.get("/api/status")
async def status():
    return {
        "pipeline_running": _pipeline_running,
        "hitl_active": hitl_bridge.is_active,
        "web_mode": True,
        "models": {
            "triage": cfg.HF_TRIAGE_MODEL,
            "diagnosis": cfg.HF_DIAGNOSIS_MODEL,
            "judge": cfg.HF_JUDGE_MODEL,
        },
    }
