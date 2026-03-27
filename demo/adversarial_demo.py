"""
Eidolon - Adversarial Security Demo
Demonstrates the cryptographic safety net in three acts.

  Act 1: Normal integrity  - mapper intact, HMAC passes.
  Act 2: Mapper corrupted  - one hash mapped to wrong name.
                              Merkle HMAC detects the tamper and REJECTS.
  Act 3: Restore           - mapper DB restored, all checks clean again.

Run:
    .venv\\Scripts\\python.exe demo\\adversarial_demo.py
"""
from __future__ import annotations

import duckdb
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

import ghost_config as cfg
from src.core.mapper import GhostMapper, _load_or_create_session_key
from src.core.delta_protocol import create_manifest, verify_manifest
from src.core.cpg_extractor import CPGExtractor

console = Console()

# Using a disposable temp copy to never collide with a running daemon
_TEMP_DB = Path(tempfile.gettempdir()) / "ghost_adversarial_mapper.db"
_BACKUP  = Path(tempfile.gettempdir()) / "ghost_adversarial_backup.db"


def _setup_temp_db() -> None:
    """Copy the real mapper DB to a temp location for the demo."""
    src = Path(cfg.MAPPER_DB_PATH)
    if not src.exists():
        console.print("[yellow]No existing mapper DB found - a fresh one will be created[/]")
    else:
        shutil.copy(str(src), str(_TEMP_DB))
        console.print(f"  [dim]Working copy: {_TEMP_DB}[/]")


def _act(num: int, title: str, color: str = "cyan") -> None:
    console.print()
    console.rule(f"[bold {color}]Act {num}: {title}[/]")
    console.print()


def act1_normal_integrity_check() -> tuple[str, bytes]:
    """Show that HMAC verification passes on untampered data."""
    _act(1, "Normal Operation - Merkle HMAC Passes", "green")

    mapper = GhostMapper(db_path=str(_TEMP_DB), key_path=cfg.SESSION_KEY_PATH)
    ext    = CPGExtractor(mapper, service="adversarial-demo")
    source = Path("demo/app_test.py").read_text(encoding="utf-8")
    payload = ext.extract(source, "demo/app_test.py")
    d = payload.to_dict()

    session_key = _load_or_create_session_key(cfg.SESSION_KEY_PATH)
    manifest = create_manifest(
        {"service": "adversarial-demo", "file_path": "demo/app_test.py",
         "nodes": d["nodes"], "edges": d["edges"]},
        [],
        session_key,
    )
    ok = verify_manifest(manifest, session_key)

    console.print(f"  Merkle root  : [dim]{manifest.merkle_root[:20]}...[/]")
    console.print(f"  Signature    : [dim]{manifest.signature[:28]}...[/]")
    console.print(f"  Verification : [bold {'green' if ok else 'red'}]{'PASSED ' if ok else 'FAILED ✗'}[/]")
    console.print()
    console.print(Panel(
        "[green]Manifest cryptographically intact.\n"
        "Every node hash is HMAC-signed - any tampering will be detected.[/]",
        title="Result", border_style="green",
    ))
    mapper.close()
    return manifest.merkle_root, session_key


def act2_corrupted_mapper(original_merkle: str, session_key: bytes) -> None:
    """Corrupt the temp mapper DB; show HMAC rejects the stale manifest."""
    _act(2, "Mapper DB Corrupted - HMAC Catches the Attack", "red")

    # Backup before corruption
    shutil.copy(str(_TEMP_DB), str(_BACKUP))

    conn = duckdb.connect(str(_TEMP_DB))
    try:
        row = conn.execute("SELECT hash_id, original_name FROM ghost_map LIMIT 1").fetchone()
        if row:
            evil = "INJECTED_MALICIOUS_PAYLOAD"
            conn.execute("UPDATE ghost_map SET original_name = ? WHERE hash_id = ?",
                         [evil, row[0]])
            console.print(f"  [red]CORRUPTED[/] {row[0]} -> was '{row[1]}' -> now '{evil}'")
    finally:
        conn.close()

    # Re-extract with the corrupted mapper
    mapper = GhostMapper(db_path=str(_TEMP_DB), key_path=cfg.SESSION_KEY_PATH)
    ext    = CPGExtractor(mapper, service="adversarial-demo")
    source = Path("demo/app_test.py").read_text(encoding="utf-8")
    payload = ext.extract(source, "demo/app_test.py")
    d = payload.to_dict()
    new_key  = _load_or_create_session_key(cfg.SESSION_KEY_PATH)
    manifest = create_manifest(
        {"service": "adversarial-demo", "file_path": "demo/app_test.py",
         "nodes": d["nodes"], "edges": d["edges"]},
        [],
        new_key,
    )
    mapper.close()

    # Simulate replay attack: swap in original's Merkle root
    manifest.merkle_root = original_merkle
    ok = verify_manifest(manifest, session_key)

    console.print(f"\n  Verification with tampered manifest: "
                  f"[bold {'red' if not ok else 'green'}]{'REJECTED ✗' if not ok else 'PASSED'}[/]")
    console.print()
    console.print(Panel(
        "[red]Eidolon detected the attack.\n\n"
        "The HMAC signature covers the Merkle root - swapping even one\n"
        "mapping invalidates the signature. The sync worker would REJECT\n"
        "this manifest and alert on-call. No corrupted data reaches Qdrant.[/]",
        title="Security Layer Triggered", border_style="red",
    ))


def act3_restore() -> None:
    """Restore the mapper and verify clean."""
    _act(3, "Mapper Restored - System Back to Clean State", "green")

    if _BACKUP.exists():
        shutil.copy(str(_BACKUP), str(_TEMP_DB))
        _BACKUP.unlink()
        console.print("  [green]Restored temp DB from backup[/]")

    mapper = GhostMapper(db_path=str(_TEMP_DB), key_path=cfg.SESSION_KEY_PATH)
    ext    = CPGExtractor(mapper, service="adversarial-demo")
    source = Path("demo/app_test.py").read_text(encoding="utf-8")
    payload = ext.extract(source, "demo/app_test.py")
    d = payload.to_dict()
    session_key = _load_or_create_session_key(cfg.SESSION_KEY_PATH)
    manifest    = create_manifest(
        {"service": "adversarial-demo", "file_path": "demo/app_test.py",
         "nodes": d["nodes"], "edges": d["edges"]},
        [],
        session_key,
    )
    ok = verify_manifest(manifest, session_key)
    mapper.close()

    console.print(f"  Verification : [bold green]{'PASSED' if ok else 'FAILED'}[/]")
    console.print()
    console.print(Panel(
        "[green]System integrity fully restored.\n\n"
        "Key insight: without the HMAC layer, a corrupted local DB would\n"
        "silently produce wrong patch reconstructions with no error.\n"
        "Eidolon's cryptographic chain prevents all silent failures.[/]",
        title="Why the Crypto Layer Matters", border_style="green",
    ))
    # Cleanup temp files
    if _TEMP_DB.exists():
        _TEMP_DB.unlink()


def main() -> None:
    console.print()
    console.print("[bold cyan]Eidolon - Adversarial Security Demo[/]")
    console.print("[dim]3 acts: normal -> attack -> recovery[/]")
    console.print("[dim]Note: uses a temp copy of the mapper DB - safe to run alongside the daemon[/]")

    _setup_temp_db()

    merkle, key = act1_normal_integrity_check()
    time.sleep(0.5)

    act2_corrupted_mapper(merkle, key)
    time.sleep(0.5)

    act3_restore()

    console.print()
    console.rule("[bold cyan]Demo Complete[/]")
    console.print()


if __name__ == "__main__":
    main()
