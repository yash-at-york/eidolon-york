# Eidolon

A software agent that diagnoses bugs in code it is not allowed to read.

Before anything leaves the local machine, every identifier in the source - function names, variable names, class names - is replaced with an HMAC-SHA256 hash. What the AI receives is a structural graph: call relationships, control flow, type signatures, parameter counts. No names. The AI reasons about the *shape* of the code, not its content. When it proposes a fix, the fix is expressed in the same hashed form. A local lookup table, which never leaves the machine, translates it back.

This matters in practice: most code that breaks in production contains domain logic - payment flows, user records, business rules - that an organisation is not comfortable sending to an external model. Eidolon offers a way to use a capable AI for diagnosis without making that trade-off.

---

## How it works

A file watcher extracts a Code Property Graph (CPG) from the monitored source, hashing all identifiers against a per-session key. This structural payload is synced via NATS to a local-network hub running Qdrant (vector search) and FalkorDB (graph traversal). When an error event arrives, a five-node LangGraph pipeline runs:

1. **Triage** - classify error type and check for duplicates in the vector store
2. **Context Fetch** - retrieve structurally similar ghost nodes and the call graph
3. **Hypothesis** - generate ranked root-cause hypotheses from the structural evidence
4. **Validation** - a second model acts as judge; structural overlap is checked independently
5. **Patch Synthesis** - produce a hashed change specification

The pipeline pauses before the final step. A human reviews the diagnosis and decides whether to proceed. The fix, once approved, is de-hashed locally and presented as a readable description. Nothing automatically modifies any file.

---

## Quick start

**Prerequisites:** Docker, Python 3.11+, a HuggingFace Pro API token.

```bash
# 1. Clone and install
git clone https://github.com/yashcb/eidolon.git
cd eidolon
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add HF_TOKEN=hf_... to .env

# 3. Launch the web dashboard (handles Docker + seeding automatically)
python demo/launch_web.py
# Open http://localhost:7433
```

For CLI use without the dashboard:

```bash
docker compose -f infrastructure/docker-compose.yml up -d
python demo/seed_demo_data.py
python src/agent/graph.py --event "401 Unauthorized on POST /verify-token" --service demo-svc
```

---

## Privacy model

The session key (`~/.ghost_session_key`) is generated fresh per session and stored only on the local machine. Rotating it invalidates all previous hashes and forces a resync. The mapper database (`.ghost_mapper.db`) that holds the hash-to-name table never leaves the machine. The LangSmith tracing integration (optional) receives only hashed identifiers. The payload scanner enforces this before any data is transmitted.

---

## Models

Calls are routed through HuggingFace's Inference API. Defaults:

| Role | Model |
|---|---|
| Triage | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Diagnosis | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| Judge | `Qwen/Qwen2.5-7B-Instruct` |
| Embeddings | `Salesforce/codet5p-110m-embedding` (local) |

Override any model via environment variable (`HF_TRIAGE_MODEL`, `HF_DIAGNOSIS_MODEL`, etc.).

---

## Project layout

```
src/
  core/       # File watcher, CPG extraction, HMAC mapper, delta protocol
  cloud/      # Embedder, Qdrant and FalkorDB sync workers
  agent/      # LangGraph pipeline nodes, HITL gate, patch reconstructor
  web/        # FastAPI dashboard server, WebSocket event hub
demo/         # Seed data, bug injector, demo app, unified launcher
docs/         # Research notes and design documents
infrastructure/  # docker-compose for Qdrant, FalkorDB, NATS
tests/
```

---

## Status

This is a working research prototype. The core privacy guarantee is stable. The LangGraph pipeline is functional end-to-end including the HITL interrupt. Measured MTTD on the demo scenario is ~30 to 40 seconds. The GNN triage layer, FIXCHECK assertion generation, and other features are on the roadmap.
