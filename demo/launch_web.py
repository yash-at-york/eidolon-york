"""
Eidolon - Unified Web Launcher
One command to rule them all.

Usage:
    python demo/launch_web.py
    python demo/launch_web.py --port 7433
    python demo/launch_web.py --skip-seed  (if already seeded)

What it does:
  1. Checks Docker containers are running (starts if not)
  2. Seeds Qdrant with demo data (if collection is empty)
  3. Sets GHOST_WEB_MODE=true
  4. Starts Eidolon dashboard at http://localhost:7433
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def banner():
    print("\n" + "═" * 62)
    print("  Eidolon  ·  Agentic Debugger  ·  Web Dashboard")
    print("═" * 62 + "\n")


def check_docker():
    print("▸ Checking Docker services…")
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(ROOT / "infrastructure" / "docker-compose.yml"), "ps", "--services", "--filter", "status=running"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        running = result.stdout.strip().split()
        needed = {"ghost_qdrant", "ghost_falkordb", "ghost_nats"}
        # Check by container name
        ps_result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        containers = set(ps_result.stdout.strip().split())
        missing = needed - containers

        if missing:
            print(f"  Starting containers: {', '.join(missing)}")
            subprocess.run(
                ["docker", "compose", "-f", str(ROOT / "infrastructure" / "docker-compose.yml"), "up", "-d"],
                cwd=str(ROOT), check=True,
            )
            print("  Waiting for services to be ready…")
            time.sleep(4)
        else:
            print("  ✓ All containers running")
    except FileNotFoundError:
        print("  ⚠ Docker not found - assuming services are running manually")
    except Exception as e:
        print(f"  ⚠ Docker check failed ({e}) - continuing anyway")


def seed_data(skip: bool):
    if skip:
        print("▸ Skipping seed (--skip-seed flag)")
        return

    print("▸ Seeding demo data…")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "demo" / "seed_demo_data.py")],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            # Extract last meaningful line
            last = [l for l in result.stdout.strip().split("\n") if l.strip()]
            print(f"  ✓ {last[-1].strip() if last else 'Seed complete'}")
        else:
            print(f"  ⚠ Seed failed: {result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print("  ⚠ Seed timed out - continuing (may already be seeded)")
    except Exception as e:
        print(f"  ⚠ Seed error: {e}")


def start_server(port: int):
    os.environ["GHOST_WEB_MODE"] = "true"
    os.environ["GHOST_DEMO_MODE"] = "true"

    url = f"http://localhost:{port}"
    print(f"\n▸ Starting Eidolon dashboard at {url}")
    print("  Press Ctrl+C to stop\n")
    print("═" * 62)
    print(f"  Open your browser: {url}")
    print("═" * 62 + "\n")

    try:
        import uvicorn
        uvicorn.run(
            "src.web.app:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="warning",
        )
    except ImportError:
        print("Installing uvicorn…")
        subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn[standard]", "fastapi"], check=True)
        import uvicorn
        uvicorn.run("src.web.app:app", host="0.0.0.0", port=port, reload=False, log_level="warning")


def main():
    banner()
    parser = argparse.ArgumentParser(description="Eidolon - Web Dashboard Launcher")
    parser.add_argument("--port",       type=int, default=7433, help="Dashboard port (default 7433)")
    parser.add_argument("--skip-seed",  action="store_true",    help="Skip demo data seeding")
    parser.add_argument("--skip-docker",action="store_true",    help="Skip Docker check")
    args = parser.parse_args()

    if not args.skip_docker:
        check_docker()
    seed_data(args.skip_seed)
    start_server(args.port)


if __name__ == "__main__":
    main()
