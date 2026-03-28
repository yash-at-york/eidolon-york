"""
Eidolon - Demo Data Seeder
Directly extracts CPG from demo/app_test.py, embeds all nodes, and upserts to
Qdrant + FalkorDB WITHOUT needing the NATS daemon flow.

Use this to pre-populate the vector/graph stores before running the agent
so the Context Fetch node finds real similar nodes instead of returning 0.

Usage:
    .venv\Scripts\python.exe demo/seed_demo_data.py
    .venv\Scripts\python.exe demo/seed_demo_data.py --service demo-svc --file demo/app_test.py
"""
from __future__ import annotations

import argparse
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import ghost_config as cfg
from src.core.mapper import GhostMapper
from src.core.cpg_extractor import CPGExtractor
from src.core.delta_protocol import create_manifest
from src.core.mapper import _load_or_create_session_key
from src.cloud.embedder import embed_nodes_batch, embed_error_text
from src.cloud.vector_store import VectorStore
from src.cloud.graph_store import GraphStore
from src.security.payload_scanner import PayloadScanner

console = Console()


def seed(target_file: str = "demo/app_test.py", service: str = "demo-svc") -> None:
    console.print()
    console.rule("[bold cyan]Eidolon - Demo Data Seeder[/]")
    console.print(f"  Target: [yellow]{target_file}[/]  Service: [cyan]{service}[/]")
    console.print()

    path = Path(target_file)
    if not path.exists():
        console.print(f"[red]File not found: {target_file}[/]")
        sys.exit(1)

    source_code = path.read_text(encoding="utf-8")
    t0 = time.perf_counter()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as prog:

        # 1. CPG Extraction 
        t_cpg = prog.add_task("Extracting CPG...", total=None)
        mapper = GhostMapper(db_path=cfg.MAPPER_DB_PATH, key_path=cfg.SESSION_KEY_PATH)
        extractor = CPGExtractor(mapper, service=service)
        payload = extractor.extract(source_code, target_file)
        d = payload.to_dict()
        prog.update(t_cpg, description=f"[green]CPG extracted[/] - {len(d['nodes'])} nodes, {len(d['edges'])} edges")
        prog.stop_task(t_cpg)

        # 2. Security scan 
        t_scan = prog.add_task("Scanning payload...", total=None)
        scan = PayloadScanner().scan(d)
        if not scan["passed"]:
            prog.stop()
            console.print(f"[red]PAYLOAD SCAN FAILED - {len(scan['violations'])} violations[/]")
            for v in scan["violations"]:
                console.print(f"  {v}")
            mapper.close()
            sys.exit(1)
        prog.update(t_scan, description=f"[green]Scan passed[/] - {scan['scanned_ids']} IDs clean")
        prog.stop_task(t_scan)

        # 3. Embedding 
        t_emb = prog.add_task("Embedding nodes...", total=None)
        vectors = embed_nodes_batch(d["nodes"])
        prog.update(t_emb, description=f"[green]Embedded[/] - {len(vectors)} vectors ({len(vectors[0]) if vectors else 0}-dim)")
        prog.stop_task(t_emb)

        # 4. Upsert to Qdrant 
        t_qdrant = prog.add_task("Upserting to Qdrant...", total=None)
        vector_store = VectorStore()
        records = [
            (
                node["id"],
                vec,
                {
                    "service": service,
                    "type": node.get("type", "Function"),
                    "is_async": node.get("is_async", False),
                    "return_type": node.get("return_type", "Any"),
                    "logic_steps": len(node.get("logic_sequence", [])),
                    "param_count": len(node.get("parameters", [])),
                    "seeded_at": time.time(),
                }
            )
            for node, vec in zip(d["nodes"], vectors)
        ]
        vector_store.upsert_batch(records)
        prog.update(t_qdrant, description=f"[green]Qdrant[/] - {len(records)} nodes upserted")
        prog.stop_task(t_qdrant)

        # 5. Sync to FalkorDB 
        t_graph = prog.add_task("Syncing to FalkorDB...", total=None)
        graph_store = GraphStore()
        graph_store.sync_from_payload(d, service)
        prog.update(t_graph, description=f"[green]FalkorDB[/] - {len(d['edges'])} edges synced")
        prog.stop_task(t_graph)

        # 6. Save manifest for delta history 
        t_manifest = prog.add_task("Saving manifest...", total=None)
        session_key = _load_or_create_session_key(cfg.SESSION_KEY_PATH)
        manifest = create_manifest(
            {"service": service, "file_path": target_file, "nodes": d["nodes"], "edges": d["edges"]},
            [],
            session_key,
        )
        manifest_path = Path("demo") / "last_manifest.json"
        manifest_path.parent.mkdir(exist_ok=True)
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
        prog.update(t_manifest, description=f"[green]Manifest[/] - saved to demo/last_manifest.json")
        prog.stop_task(t_manifest)

    mapper.close()
    elapsed = time.perf_counter() - t0

    # Summary table 
    console.print()
    table = Table(title=f"Seed Complete - {elapsed:.1f}s", show_header=True)
    table.add_column("Store", style="cyan")
    table.add_column("Items", justify="right", style="green")
    table.add_row("Qdrant (vectors)", str(len(records)))
    table.add_row("FalkorDB (edges)", str(len(d["edges"])))
    table.add_row("Mapper entries", str(len(d["nodes"]) * 3))
    console.print(table)
    console.print()
    console.print("[bold green]Data seeded successfully![/]")
    console.print("[dim]Now run: .venv\\Scripts\\python.exe src/agent/graph.py "
                  "--event \"401 Unauthorized on POST /verify-token\" --service demo-svc[/]")
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Qdrant + FalkorDB with demo CPG data")
    parser.add_argument("--file", default="demo/app_test.py")
    parser.add_argument("--service", default="demo-svc")
    args = parser.parse_args()
    seed(args.file, args.service)
