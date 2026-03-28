"""
Eidolon - Central Configuration
All runtime behaviour is controlled from here and via environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root 
ROOT_DIR = Path(__file__).parent

# Operational Mode 
# Set GHOST_DEMO_MODE=true in .env for conference demo behaviour:
#   - Rich terminal output of ghost payload
#   - In-memory Qdrant (no Docker needed)
#   - HITL waits for Enter key from audience
DEMO_MODE: bool = os.getenv("GHOST_DEMO_MODE", "false").lower() == "true"

# Service Namespace 
# Used to namespace node IDs across multi-service deployments:  "svc::h_7b9x"
SERVICE_NAMESPACE: str = os.getenv("GHOST_SERVICE", "default-svc")

# Watch Target 
WATCH_PATH: str = os.getenv("GHOST_WATCH_PATH", str(ROOT_DIR / "demo" / "app_test.py"))
WATCH_RECURSIVE: bool = os.getenv("GHOST_WATCH_RECURSIVE", "false").lower() == "true"

# Mapper (Local DB) 
MAPPER_DB_PATH: str = os.getenv("GHOST_MAPPER_DB", str(ROOT_DIR / ".ghost_mapper.db"))
SESSION_KEY_PATH: str = os.getenv("GHOST_SESSION_KEY", str(ROOT_DIR / ".ghost_session_key"))

# HuggingFace 
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# LLM models
#   Triage  = speed-critical
#   Diagnosis = quality-critical 
#   Judge   = reliability-critical 
HF_TRIAGE_MODEL: str = os.getenv("HF_TRIAGE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
HF_DIAGNOSIS_MODEL: str = os.getenv("HF_DIAGNOSIS_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_JUDGE_MODEL: str = os.getenv("HF_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# OpenAI Models
USE_OPENAI_MODELS: bool = os.getenv("USE_OPENAI_MODELS", "true").lower() == "true"
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

OPENAI_TRIAGE_MODEL: str = os.getenv("OPENAI_TRIAGE_MODEL", "gpt-4o-mini")
OPENAI_DIAGNOSIS_MODEL: str = os.getenv("OPENAI_DIAGNOSIS_MODEL", "gpt-4o")
OPENAI_JUDGE_MODEL: str = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

# Per-role max_tokens limits
HF_TRIAGE_MAX_TOKENS: int = int(os.getenv("HF_TRIAGE_MAX_TOKENS", "512"))
HF_DIAGNOSIS_MAX_TOKENS: int = int(os.getenv("HF_DIAGNOSIS_MAX_TOKENS", "1500"))
HF_JUDGE_MAX_TOKENS: int = int(os.getenv("HF_JUDGE_MAX_TOKENS", "256"))

# Embedding model - loaded locally (no API call needed)
HF_EMBEDDING_MODEL: str = os.getenv("HF_EMBEDDING_MODEL", "Salesforce/codet5p-110m-embedding")

# Vector Store (Qdrant) 
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "ghost_nodes")
QDRANT_VECTOR_DIM: int = 256  

# In DEMO_MODE, use in-memory Qdrant (no Docker needed)
QDRANT_IN_MEMORY: bool = DEMO_MODE or os.getenv("QDRANT_IN_MEMORY", "false").lower() == "true"

# Graph Store (FalkorDB) 
FALKORDB_HOST: str = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT: int = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH: str = os.getenv("FALKORDB_GRAPH", "ghost_cpg")

# NATS JetStream 
NATS_URL: str = os.getenv("NATS_URL", "nats://localhost:4222")
NATS_SUBJECT: str = "ghost.delta"
NATS_STREAM: str = "GHOST"

# LangSmith Observability 
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ghost-twin")
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

# MTTD Measurement 
MTTD_MODE: bool = os.getenv("GHOST_MTTD_MODE", "true").lower() == "true"

# Deduplication 
# Cosine similarity threshold above which an incoming error is considered a duplicate.
DEDUP_SIMILARITY_THRESHOLD: float = float(os.getenv("GHOST_DEDUP_THRESHOLD", "0.92"))

# Agent 
AGENT_MAX_HYPOTHESES: int = int(os.getenv("AGENT_MAX_HYPOTHESES", "2"))
AGENT_CONTEXT_TOP_K: int = int(os.getenv("AGENT_CONTEXT_TOP_K", "5"))
AGENT_GRAPH_DEPTH: int = int(os.getenv("AGENT_GRAPH_DEPTH", "2"))
CONFIDENCE_THRESHOLD_AUTO: float = float(os.getenv("CONFIDENCE_THRESHOLD_AUTO", "0.85"))
CONFIDENCE_THRESHOLD_WARN: float = float(os.getenv("CONFIDENCE_THRESHOLD_WARN", "0.65"))

# Hypothesis Refinement Loop 
# If composite_confidence < REFINE threshold AND retry_count < MAX_RETRIES,
# route back to hypothesis_node for a second attempt with failure context injected.
CONFIDENCE_THRESHOLD_REFINE: float = float(os.getenv("CONFIDENCE_THRESHOLD_REFINE", "0.55"))
MAX_HYPOTHESIS_RETRIES: int = int(os.getenv("MAX_HYPOTHESIS_RETRIES", "2"))

# ReAct Inner Loop (hypothesis node) 
# Max iterations the LLM can request additional tool calls before finalizing.
REACT_MAX_ITERATIONS: int = int(os.getenv("REACT_MAX_ITERATIONS", "3"))

# Error Fingerprint Memory DB 
# Separate SQLite DB for rejection memory + outcome tracking.
MEMORY_DB_PATH: str = os.getenv("GHOST_MEMORY_DB", str(ROOT_DIR / ".ghost_memory.db"))

# Security 
PAYLOAD_SCAN_ENABLED: bool = os.getenv("GHOST_PAYLOAD_SCAN", "true").lower() == "true"

# Web Dashboard Mode 
WEB_MODE: bool = os.getenv("GHOST_WEB_MODE", "false").lower() == "true"
