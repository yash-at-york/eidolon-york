"""
Eidolon - Error Fingerprinter & Rejection Memory  [MY ADDITION]

WHY THIS EXISTS:
The pipeline's triage node does vector-similarity dedup, which is good for semantically
similar errors but (a) requires a full model forward pass every time, and (b) has no memory
of *what was tried and rejected* for a given error pattern.

This module adds two independent capabilities:

1. FAST EXACT FINGERPRINTING
   Extracts a structured 3-part key from raw error strings:
     {HTTP_code}::{endpoint_pattern}::{exception_type}
   e.g. "401::POST /verify-token::HTTPException"
   Result is hashed to a short ID (no PII). O(1) SQLite lookup before vector dedup.
   Common repeated errors skip the embedding entirely.

2. REJECTION MEMORY (LOCAL ONLY)
   When a human rejects a patch at HITL, we record:
     (fingerprint_id, hypothesis, feedback, timestamp)
   On future runs matching the same fingerprint, the top-3 rejections are injected
   into the hypothesis prompt as "structural approaches to AVOID".
   This gives the LLM genuine memory of what failed before — free learning loop,
   no ML training required.

   Also tracks accepted outcomes so we can build success pattern statistics over time.

STORAGE:
   Standard SQLite via stdlib `sqlite3` (not DuckDB) to avoid lock conflicts with the
   DuckDB-backed GhostMapper. File: .ghost_memory.db (local only, never transmitted).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from threading import Lock

import ghost_config as cfg

# ── Regex patterns for structured extraction ──────────────────────────────────
_HTTP_CODE_RE   = re.compile(r'\b([2-5]\d{2})\b')
_ENDPOINT_RE    = re.compile(r'\b(GET|POST|PUT|PATCH|DELETE|HEAD)\s+(/[\w/\-_.{}%?&=]*)', re.IGNORECASE)
_EXCEPTION_RE   = re.compile(r'\b(\w*(?:Error|Exception|Fault|Warning|Timeout|Unauthorized|Forbidden|NotFound|Conflict))\b')
_TRACEBACK_RE   = re.compile(r'File "([^"]+)",\s+line\s+(\d+)')


class ErrorFingerprinter:
    """
    Thread-safe error fingerprinter with SQLite-backed rejection memory.

    Designed to be instantiated once per pipeline run and kept alive.
    The underlying SQLite connection is thread-safe in WAL mode.
    """

    def __init__(self, db_path: str = cfg.MEMORY_DB_PATH) -> None:
        self._db_path = db_path
        self._lock = Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._setup_schema()

    # ── Schema ───────────────────────────────────────────────────────────────

    def _setup_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    fingerprint_id  TEXT PRIMARY KEY,
                    structured_key  TEXT NOT NULL,
                    first_seen_at   REAL NOT NULL,
                    last_seen_at    REAL NOT NULL,
                    occurrence_count INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS rejection_memory (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id  TEXT NOT NULL,
                    hypothesis_text TEXT NOT NULL,
                    patch_sketch    TEXT,
                    feedback        TEXT,
                    recorded_at     REAL NOT NULL,
                    FOREIGN KEY (fingerprint_id) REFERENCES fingerprints(fingerprint_id)
                );

                CREATE TABLE IF NOT EXISTS outcome_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id  TEXT NOT NULL,
                    approved        INTEGER NOT NULL,
                    composite_score REAL,
                    error_type      TEXT,
                    recorded_at     REAL NOT NULL,
                    FOREIGN KEY (fingerprint_id) REFERENCES fingerprints(fingerprint_id)
                );

                CREATE INDEX IF NOT EXISTS idx_rejection_fid ON rejection_memory(fingerprint_id);
                CREATE INDEX IF NOT EXISTS idx_outcome_fid   ON outcome_log(fingerprint_id);
            """)
            self._conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def fingerprint(self, error_text: str) -> tuple[str, str]:
        """
        Extract a structured fingerprint from a raw error string.

        Returns: (fingerprint_id: str, structured_key: str)
          fingerprint_id — short SHA-256 hex (12 chars), safe to store/log
          structured_key — human-readable composite: "{code}::{method} {endpoint}::{ExcType}"
        """
        code  = self._extract_http_code(error_text)
        ep    = self._extract_endpoint(error_text)
        exc   = self._extract_exception(error_text)
        key   = f"{code}::{ep}::{exc}"
        fid   = hashlib.sha256(key.encode()).hexdigest()[:12]

        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT occurrence_count FROM fingerprints WHERE fingerprint_id = ?",
                (fid,),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE fingerprints SET last_seen_at = ?, occurrence_count = occurrence_count + 1 WHERE fingerprint_id = ?",
                    (now, fid),
                )
            else:
                self._conn.execute(
                    "INSERT INTO fingerprints (fingerprint_id, structured_key, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?)",
                    (fid, key, now, now),
                )
            self._conn.commit()

        return fid, key

    def is_known_exact(self, fingerprint_id: str, min_occurrences: int = 2) -> bool:
        """
        Returns True if this fingerprint has been seen >= min_occurrences times.
        Fast O(1) SQLite lookup — no embedding needed.
        Used as a pre-check before the expensive vector dedup in triage.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT occurrence_count FROM fingerprints WHERE fingerprint_id = ?",
                (fingerprint_id,),
            ).fetchone()
        return bool(row and row[0] >= min_occurrences)

    def get_rejections(self, fingerprint_id: str, limit: int = 3) -> list[dict]:
        """
        Return the most recent rejection records for this fingerprint.
        Injected into the hypothesis prompt as anti-patterns to avoid.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT hypothesis_text, patch_sketch, feedback, recorded_at
                FROM rejection_memory
                WHERE fingerprint_id = ?
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (fingerprint_id, limit),
            ).fetchall()
        return [
            {
                "hypothesis": row[0],
                "patch_sketch": row[1] or "",
                "feedback": row[2] or "",
                "recorded_at": row[3],
            }
            for row in rows
        ]

    def record_rejection(
        self,
        fingerprint_id: str,
        hypothesis: dict,
        feedback: str = "",
    ) -> None:
        """
        Persist a rejected hypothesis so future runs can avoid the same mistake.
        Called from graph.py after the human rejects at HITL.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO rejection_memory
                    (fingerprint_id, hypothesis_text, patch_sketch, feedback, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    fingerprint_id,
                    hypothesis.get("hypothesis", ""),
                    hypothesis.get("patch_sketch", ""),
                    feedback,
                    time.time(),
                ),
            )
            self._conn.commit()

    def record_outcome(
        self,
        fingerprint_id: str,
        approved: bool,
        composite_score: float = 0.0,
        error_type: str = "unknown",
    ) -> None:
        """Track full outcome (approved or rejected) for pattern statistics."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO outcome_log
                    (fingerprint_id, approved, composite_score, error_type, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (fingerprint_id, int(approved), composite_score, error_type, time.time()),
            )
            self._conn.commit()

    def get_stats(self, fingerprint_id: str) -> dict:
        """Return approval rate + occurrence stats for a fingerprint."""
        with self._lock:
            fp_row = self._conn.execute(
                "SELECT structured_key, occurrence_count FROM fingerprints WHERE fingerprint_id = ?",
                (fingerprint_id,),
            ).fetchone()
            outcomes = self._conn.execute(
                "SELECT approved, COUNT(*) FROM outcome_log WHERE fingerprint_id = ? GROUP BY approved",
                (fingerprint_id,),
            ).fetchall()

        approved = next((c for a, c in outcomes if a == 1), 0)
        rejected = next((c for a, c in outcomes if a == 0), 0)
        total = approved + rejected
        return {
            "fingerprint_id": fingerprint_id,
            "structured_key": fp_row[0] if fp_row else "unknown",
            "occurrences": fp_row[1] if fp_row else 0,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / total if total else None,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ErrorFingerprinter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Private extraction helpers ────────────────────────────────────────────

    def _extract_http_code(self, text: str) -> str:
        m = _HTTP_CODE_RE.search(text)
        return m.group(1) if m else "000"

    def _extract_endpoint(self, text: str) -> str:
        m = _ENDPOINT_RE.search(text)
        if m:
            # Normalise path params: /users/123 → /users/{id}
            path = re.sub(r'/\d+', '/{id}', m.group(2))
            return f"{m.group(1).upper()} {path}"
        return "UNKNOWN"

    def _extract_exception(self, text: str) -> str:
        m = _EXCEPTION_RE.search(text)
        return m.group(1) if m else "UnknownError"
