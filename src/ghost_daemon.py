"""
Eidolon - Ghost Daemon
File watcher that triggers on save, runs CPG extraction, computes a Merkle
delta manifest, and publishes it to NATS JetStream.

In DEMO_MODE, also prints a rich colored payload summary to the terminal.

Usage:
    python src/ghost_daemon.py                        # uses .env config
    python src/ghost_daemon.py --demo-mode            # force demo mode
    python src/ghost_daemon.py --watch demo/app_test.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import nats
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

import ghost_config as cfg
from src.core.cpg_extractor import CPGExtractor
from src.core.delta_protocol import create_manifest, verify_manifest
from src.core.mapper import GhostMapper
from src.security.payload_scanner import PayloadScanner

console = Console()



# NATS publisher


class NATSPublisher:
    """Async NATS JetStream publisher with reconnect support."""

    def __init__(self, url: str, subject: str) -> None:
        self.url = url
        self.subject = subject
        self._nc = None
        self._js = None

    async def connect(self) -> None:
        self._nc = await nats.connect(self.url)
        self._js = self._nc.jetstream()
        # Ensure stream exists
        try:
            await self._js.add_stream(
                name=cfg.NATS_STREAM,
                subjects=[f"{cfg.NATS_SUBJECT}.>", cfg.NATS_SUBJECT],
            )
        except Exception:
            pass  # stream already exists

    async def publish(self, payload: dict) -> None:
        if self._js is None:
            await self.connect()
        data = json.dumps(payload).encode()
        await self._js.publish(self.subject, data)

    async def close(self) -> None:
        if self._nc:
            await self._nc.drain()



# File event handler


class GhostEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler. On each file save:
    1. Parse → CPG payload
    2. Compute Merkle delta manifest
    3. Scan payload for plaintext (security gate)
    4. Publish to NATS (or print in demo mode)
    """

    def __init__(
        self,
        watch_path: str,
        mapper: GhostMapper,
        extractor: CPGExtractor,
        scanner: PayloadScanner,
        publisher: NATSPublisher | None,
        demo_mode: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.watch_path = Path(watch_path).resolve()
        self.mapper = mapper
        self.extractor = extractor
        self.scanner = scanner
        self.publisher = publisher
        self.demo_mode = demo_mode
        self.loop = loop
        self._prev_nodes: list[dict] = []

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return
        changed_path = Path(str(event.src_path)).resolve()

        # Only trigger on the watched file(s)
        if self.watch_path.is_file() and changed_path != self.watch_path:
            return
        if self.watch_path.is_dir() and not str(changed_path).startswith(str(self.watch_path)):
            return
        if not str(changed_path).endswith(".py"):
            return

        console.rule(f"[bold cyan]Eidolon - File Save Detected[/]")
        console.print(f"  [dim]→ {changed_path}[/]")

        try:
            source_code = changed_path.read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/]")
            return

        # 1. Extract CPG
        try:
            cpg = self.extractor.extract(source_code, str(changed_path))
            cpg_dict = cpg.to_dict()
        except Exception as e:
            console.print(f"[red]CPG extraction failed: {e}[/]")
            return

        # 2. Security scan
        if cfg.PAYLOAD_SCAN_ENABLED:
            scan_result = self.scanner.scan(cpg_dict)
            if not scan_result["passed"]:
                console.print(f"[bold red]PAYLOAD REJECTED - plaintext identifiers detected:[/]")
                for v in scan_result["violations"]:
                    console.print(f"  [red]• {v}[/]")
                return

        # 3. Compute delta manifest
        session_key = self.mapper._key  # bytes
        manifest = create_manifest(cpg_dict, self._prev_nodes, session_key)
        self._prev_nodes = cpg_dict["nodes"]
        self.mapper.save_checksum(manifest.merkle_root)

        n_changed = len(manifest.changed_nodes)
        n_edges = len(manifest.all_edges)

        console.print(
            f"  [green][/] Merkle root: [yellow]{manifest.merkle_root[:16]}…[/]  "
            f"Changed nodes: [cyan]{n_changed}[/]  "
            f"Total edges: [cyan]{n_edges}[/]"
        )

        # 4. Demo mode: rich terminal display
        if self.demo_mode:
            self._print_demo_payload(cpg_dict, manifest)

        # 5. Publish to NATS (if not demo-only)
        if self.publisher and not self.demo_mode:
            asyncio.run_coroutine_threadsafe(
                self.publisher.publish(manifest.to_dict()),
                self.loop,
            )
            console.print(f"  [green][/] Delta manifest published to NATS [cyan]{cfg.NATS_SUBJECT}[/]")
        elif self.demo_mode:
            # In demo mode, still write to a local file for agent access
            out_path = Path("demo") / "last_manifest.json"
            out_path.write_text(manifest.to_json())
            console.print(f"  [green][/] Manifest saved to [cyan]{out_path}[/] (demo mode)")

    def _print_demo_payload(self, cpg_dict: dict, manifest) -> None:
        """Rich terminal display for conference demo mode."""
        console.print()

        # Nodes table
        table = Table(title="Ghost Payload - What the AI Sees", show_lines=True)
        table.add_column("Hash ID", style="yellow", no_wrap=True)
        table.add_column("Type", style="cyan")
        table.add_column("Async", justify="center")
        table.add_column("Params", style="green")
        table.add_column("Returns", style="magenta")

        for node in cpg_dict["nodes"]:
            params = ", ".join(p["name"] for p in node.get("parameters", []))
            table.add_row(
                node["id"],
                node["type"],
                "" if node.get("is_async") else "✗",
                params or "[dim]none[/]",
                node.get("return_type", "Any"),
            )
        console.print(table)

        # Edges table
        if cpg_dict["edges"]:
            edge_table = Table(title="CPG Edges - Causal Structure", show_lines=True)
            edge_table.add_column("From", style="yellow", no_wrap=True)
            edge_table.add_column("Edge Type", style="bold")
            edge_table.add_column("To", style="cyan", no_wrap=True)
            edge_table.add_column("Properties", style="dim")
            for edge in cpg_dict["edges"][:20]:  # cap at 20 for readability
                props = {k: v for k, v in edge.items() if k not in ("from", "to", "type")}
                edge_table.add_row(
                    edge["from"],
                    _edge_color(edge["type"]),
                    edge["to"],
                    str(props) if props else "",
                )
            console.print(edge_table)

        console.print(Panel(
            f"[bold green]Zero proprietary identifiers transmitted.[/]\n"
            f"[dim]The AI agent will receive only hashed structures like the above.[/]",
            title="Privacy Guarantee",
            border_style="green",
        ))
        console.print()


def _edge_color(edge_type: str) -> str:
    colors = {
        "CALLS": "[bold yellow]CALLS[/]",
        "DATA_FLOW": "[bold blue]DATA_FLOW[/]",
        "IMPORTS": "[bold dim]IMPORTS[/]",
    }
    return colors.get(edge_type, edge_type)



# Entry point


async def run(watch_path: str, demo_mode: bool) -> None:
    console.print(Panel(
        f"[bold cyan]Eidolon Daemon[/]\n"
        f"Mode: [yellow]{'DEMO' if demo_mode else 'PRODUCTION'}[/]\n"
        f"Watching: [green]{watch_path}[/]\n"
        f"Service: [magenta]{cfg.SERVICE_NAMESPACE}[/]",
        title="Eidolon",
        border_style="cyan",
    ))

    mapper = GhostMapper()
    extractor = CPGExtractor(mapper, service=cfg.SERVICE_NAMESPACE)
    scanner = PayloadScanner()

    publisher = None
    if not demo_mode:
        publisher = NATSPublisher(cfg.NATS_URL, cfg.NATS_SUBJECT)
        await publisher.connect()
        console.print(f"[green][/] Connected to NATS at [cyan]{cfg.NATS_URL}[/]")

    loop = asyncio.get_event_loop()

    handler = GhostEventHandler(
        watch_path=watch_path,
        mapper=mapper,
        extractor=extractor,
        scanner=scanner,
        publisher=publisher,
        demo_mode=demo_mode,
        loop=loop,
    )

    watch_target = str(Path(watch_path).parent if Path(watch_path).is_file() else watch_path)
    observer = Observer()
    observer.schedule(handler, path=watch_target, recursive=cfg.WATCH_RECURSIVE)
    observer.start()
    console.print(f"[green][/] Watchdog active - press [bold]Ctrl+C[/] to stop\n")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        mapper.close()
        if publisher:
            await publisher.close()
        console.print("\n[yellow]Eidolon daemon stopped.[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eidolon - Privacy-preserving file watcher")
    parser.add_argument("--demo-mode", action="store_true", help="Enable rich demo output")
    parser.add_argument("--watch", type=str, default=None, help="Override watch path")
    args = parser.parse_args()

    demo_mode = args.demo_mode or cfg.DEMO_MODE
    watch_path = args.watch or cfg.WATCH_PATH

    asyncio.run(run(watch_path, demo_mode))


if __name__ == "__main__":
    main()
