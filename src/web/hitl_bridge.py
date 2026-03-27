"""
Eidolon - Web HITL Bridge
Replaces the terminal Confirm.ask() with a threading.Event so the
pipeline thread can block waiting for a browser button click.

Flow:
  1. Pipeline thread calls hitl_bridge.wait_for_decision()  → blocks
  2. Browser user clicks APPROVE or REJECT
  3. FastAPI endpoint calls hitl_bridge.decide(approved=True/False) → unblocks
  4. Pipeline thread receives the decision and continues
"""
from __future__ import annotations

import threading
from typing import Optional


class HITLBridge:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._decision: Optional[dict] = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def wait_for_decision(self, timeout: float = 600.0) -> dict:
        """
        Block the pipeline thread until a human decision arrives via the web UI.
        Returns {"approved": bool, "feedback": str}.
        Defaults to rejected on timeout.
        """
        self._active = True
        self._decision = None
        self._event.clear()
        timed_out = not self._event.wait(timeout=timeout)
        self._active = False
        if timed_out:
            return {"approved": False, "feedback": "Timed out"}
        return self._decision or {"approved": False, "feedback": "No decision"}

    def decide(self, approved: bool, feedback: str = "") -> None:
        """Called from the FastAPI thread when the browser submits a decision."""
        self._decision = {"approved": approved, "feedback": feedback}
        self._event.set()

    def reset(self) -> None:
        """Reset for a new pipeline run."""
        self._active = False
        self._decision = None
        self._event.clear()


# Singleton
hitl_bridge = HITLBridge()
