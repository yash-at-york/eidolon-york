"""
Eidolon - Cloud Sync Worker
Subscribes to NATS JetStream, consumes delta manifests, and syncs changed
nodes to Qdrant (vector store) + FalkorDB (graph store).

Run: python -m src.cloud.sync_worker
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import nats
from rich.console import Console
from rich.logging import RichHandler

import ghost_config as cfg
from src.cloud.embedder import embed_nodes_batch
from src.cloud.graph_store import GraphStore
from src.cloud.vector_store import VectorStore
from src.core.delta_protocol import DeltaManifest
from src.core.mapper import _load_or_create_session_key

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("ghost.sync")
console = Console()


class SyncWorker:
    """
    NATS subscriber → Qdrant + FalkorDB sync pipeline.

    For each delta manifest:
    1. Verify HMAC signature
    2. Embed changed nodes (CodeT5+)
    3. Upsert to Qdrant (semantic search)
    4. Sync full CPG to FalkorDB (graph traversal)
    5. Log event with metrics
    """

    def __init__(self) -> None:
        self._vector_store = VectorStore()
        self._graph_store = GraphStore()
        self._session_key = _load_or_create_session_key(cfg.SESSION_KEY_PATH)
        self._nc = None
        self._sub = None
        self._running = False

    async def start(self) -> None:
        console.print("[bold cyan]Eidolon Sync Worker starting...[/]")

        try:
            self._nc = await nats.connect(
                cfg.NATS_URL,
                error_cb=self._error_cb,
                disconnected_cb=self._disconnected_cb,
            )
            js = self._nc.jetstream()
            log.info(f"Connected to NATS at {cfg.NATS_URL}")

            # Durable consumer - survives worker restarts
            self._sub = await js.subscribe(
                cfg.NATS_SUBJECT,
                durable="ghost_sync_worker",
                stream=cfg.NATS_STREAM,
            )
            log.info(f"Subscribed to [{cfg.NATS_SUBJECT}] on stream [{cfg.NATS_STREAM}]")

        except Exception as e:
            log.error(f"NATS connection failed: {e}")
            log.info("Running in FILE_MODE - reading from demo/last_manifest.json")
            self._nc = None

        self._running = True
        await self._consume_loop()

    async def _consume_loop(self) -> None:
        """Main consumer loop."""
        if self._nc is None:
            # File fallback for demo mode
            await self._process_file_fallback()
            return

        while self._running:
            try:
                msg = await self._sub.next_msg(timeout=1.0)
                await self._process_message(msg)
                await msg.ack()
            except nats.errors.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Error consuming message: {e}")

    async def _process_file_fallback(self) -> None:
        """Watch demo/last_manifest.json for demo mode (no NATS required)."""
        import asyncio
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        manifest_path = Path("demo") / "last_manifest.json"
        log.info(f"Watching {manifest_path} for new manifests...")

        class ManifestHandler(FileSystemEventHandler):
            def __init__(self, worker):
                self._worker = worker

            def on_modified(self, event):
                if Path(event.src_path).name == "last_manifest.json":
                    asyncio.run_coroutine_threadsafe(
                        self._worker._process_json_file(str(event.src_path)),
                        asyncio.get_event_loop(),
                    )

        observer = Observer()
        observer.schedule(ManifestHandler(self), str(manifest_path.parent), recursive=False)
        observer.start()

        while self._running:
            await asyncio.sleep(1)

        observer.stop()
        observer.join()

    async def _process_message(self, msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            await self._sync_manifest(data)
        except Exception as e:
            log.error(f"Failed to process message: {e}")

    async def _process_json_file(self, path: str) -> None:
        try:
            data = json.loads(Path(path).read_text())
            await self._sync_manifest(data)
        except Exception as e:
            log.error(f"Failed to process manifest file: {e}")

    async def _sync_manifest(self, data: dict) -> None:
        manifest = DeltaManifest.from_dict(data)

        # Verify integrity
        from src.core.delta_protocol import verify_manifest
        if not verify_manifest(manifest, self._session_key):
            log.warning(f"Manifest signature INVALID for {manifest.file_path} - skipping")
            return

        service = manifest.service
        changed_nodes = manifest.changed_nodes
        all_edges = manifest.all_edges

        if not changed_nodes:
            log.info(f"[{service}] No changed nodes in manifest - skipping embed")
        else:
            # Embed changed nodes
            vectors = embed_nodes_batch(changed_nodes)

            # Batch upsert to Qdrant
            records = [
                (
                    node["id"],
                    vec,
                    {
                        "service": service,
                        "type": node.get("type", "Function"),
                        "is_async": node.get("is_async", False),
                        "return_type": node.get("return_type", "Any"),
                        "git_sha": manifest.git_sha,
                        "synced_at": manifest.timestamp,
                    }
                )
                for node, vec in zip(changed_nodes, vectors)
            ]
            self._vector_store.upsert_batch(records)
            log.info(f"[{service}] Upserted {len(records)} nodes to Qdrant ")

        # Sync full CPG to FalkorDB (edges are always updated)
        cpg_for_graph = {
            "nodes": changed_nodes,
            "edges": all_edges,
        }
        self._graph_store.sync_from_payload(cpg_for_graph, service)
        log.info(f"[{service}] Synced {len(all_edges)} edges to FalkorDB ")
        log.info(f"[{service}] Merkle root: {manifest.merkle_root[:16]}… @ {manifest.git_sha}")

    async def _error_cb(self, e: Exception) -> None:
        log.error(f"NATS error: {e}")

    async def _disconnected_cb(self) -> None:
        log.warning("NATS disconnected")

    async def stop(self) -> None:
        self._running = False
        if self._nc:
            await self._nc.drain()
        log.info("Sync worker stopped")


async def main() -> None:
    worker = SyncWorker()

    loop = asyncio.get_event_loop()

    def _handle_signal(*_):
        loop.create_task(worker.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # Windows

    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
