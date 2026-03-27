"""
Eidolon - Event Hub
Thread-safe event emitter that bridges the synchronous pipeline
to the async WebSocket broadcaster.

Usage (from any pipeline thread):
    from src.web.events import event_hub
    event_hub.emit("node_start", node="triage")

Usage (from FastAPI async context):
    q = event_hub.subscribe()
    async for event in q: ...
    event_hub.unsubscribe(q)
"""
from __future__ import annotations

import asyncio
import json
from typing import Set


class EventHub:
    def __init__(self) -> None:
        self._clients: Set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        # In-memory log of all events for late-joining clients
        self._history: list[str] = []

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Must be called once from the FastAPI startup context."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Register a new WebSocket client. Returns its event queue."""
        q: asyncio.Queue = asyncio.Queue()
        self._clients.add(q)
        # Replay recent history so a newly opened tab gets current state
        for ev in self._history[-50:]:
            q.put_nowait(ev)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    def emit(self, event_type: str, **data) -> None:
        """
        Thread-safe emit. Can be called from the pipeline thread.
        Puts the serialised event into every connected client's queue.
        """
        payload = json.dumps({"type": event_type, **data})
        self._history.append(payload)
        if len(self._history) > 200:
            self._history = self._history[-100:]

        if self._loop and not self._loop.is_closed():
            for q in list(self._clients):
                self._loop.call_soon_threadsafe(q.put_nowait, payload)

    def clear_history(self) -> None:
        self._history.clear()


# Singleton
event_hub = EventHub()
