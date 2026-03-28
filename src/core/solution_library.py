"""
Eidolon — Solution Library (Case-Based Reasoning Store)

WHAT THIS IS:
  A SQLite-backed library of approved patches. When a human approves a patch at
  HITL, it gets stored here — keyed by error fingerprint + CPG node ID.

  On future identical errors: EXACT REPLAY — skip hypothesis+validation entirely,
  present the stored solution at HITL in ~5 seconds instead of 60s+.

  On structurally similar (not identical) errors: ANALOGICAL ADAPTATION —
  inject the closest stored solution as the starting hypothesis with a note that
  it was previously confirmed for a similar structural pattern.

  This implements Case-Based Reasoning (CBR):
    1. RETRIEVE: find similar past cases by exact fingerprint or vector similarity
    2. REUSE: inject the solution as starting hypothesis or replay directly
    3. REVISE: if human modifies the adapted solution, record the revision
    4. RETAIN: store the final approved solution for future cases

STORAGE:
  solutions table     — core approved patch records
  solution_vectors    — 256-dim embeddings for similarity search (optional Qdrant fallback)
  adaptation_log      — track when a solution was adapted for a novel case

All local SQLite, same .ghost_memory.db as ErrorFingerprinter.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import ghost_config as cfg


@dataclass
class Solution:
    """A single stored approved solution."""
    solution_id:       int
    fingerprint_id:    str
    exception_type:    str
    target_node_id:    str          # h_XXXXXXXX of the CPG node the patch targets
    patch_type:        str          # INSERT_CALL | ADD_GUARD | REORDER_CALLS | etc.
    patch_sketch:      str          # hashed structural spec (safe for cloud)
    reconstructed_diff: str         # human-readable (de-hashed, local only)
    confidence:        float
    approved_at:       float
    times_replayed:    int = 0
    last_replayed_at:  float = 0.0
    adaptation_source: str | None = None  # if adapted from another solution_id

    def to_dict(self) -> dict:
        return {
            "solution_id":       self.solution_id,
            "fingerprint_id":    self.fingerprint_id,
            "exception_type":    self.exception_type,
            "target_node_id":    self.target_node_id,
            "patch_type":        self.patch_type,
            "patch_sketch":      self.patch_sketch,
            "confidence":        self.confidence,
            "times_replayed":    self.times_replayed,
        }

    def to_hypothesis_hint(self) -> dict:
        """Format as a hypothesis hint for injection into the LLM prompt."""
        return {
            "note": "Previously confirmed fix for structurally similar error — adapt as appropriate",
            "patch_type":    self.patch_type,
            "patch_sketch":  self.patch_sketch,
            "target_node":   self.target_node_id,
            "confidence":    self.confidence,
            "times_used":    self.times_replayed + 1,
        }


class SolutionLibrary:
    """
    Thread-safe Case-Based Reasoning solution store.

    Provides both exact O(1) lookup (by fingerprint) and approximate
    structural similarity search (by node_id + exception_type matching).

    Usage:
        lib = SolutionLibrary()

        # Store on HITL approval:
        lib.store(
            fingerprint_id="a3f7c211b9d0",
            exception_type="NameError",
            target_node_id="h_abf89e3f",
            patch={"change_type": "ADD_IMPORT", ...},
            reconstructed_diff="+ import decode_jwt from auth_utils",
            confidence=0.87,
        )

        # Retrieve for next run:
        exact = lib.retrieve_exact("a3f7c211b9d0")  # → Solution or None
        similar = lib.retrieve_similar("NameError", "h_abf89e3f", limit=3)  # CBR
    """

    def __init__(self, db_path: str = cfg.MEMORY_DB_PATH) -> None:
        self._lock = Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._setup_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _setup_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS solutions (
                    solution_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id      TEXT NOT NULL,
                    exception_type      TEXT NOT NULL,
                    target_node_id      TEXT NOT NULL,
                    patch_type          TEXT NOT NULL,
                    patch_sketch        TEXT NOT NULL,
                    reconstructed_diff  TEXT NOT NULL DEFAULT '',
                    confidence          REAL NOT NULL,
                    approved_at         REAL NOT NULL,
                    times_replayed      INTEGER NOT NULL DEFAULT 0,
                    last_replayed_at    REAL NOT NULL DEFAULT 0,
                    adaptation_source   INTEGER,   -- solution_id this was adapted from
                    FOREIGN KEY (adaptation_source) REFERENCES solutions(solution_id)
                );

                CREATE TABLE IF NOT EXISTS adaptation_log (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_solution_id  INTEGER NOT NULL,
                    adapted_for_fid     TEXT NOT NULL,
                    adapted_for_node    TEXT NOT NULL,
                    human_modified      INTEGER NOT NULL DEFAULT 0,
                    adapted_at          REAL NOT NULL,
                    FOREIGN KEY (source_solution_id) REFERENCES solutions(solution_id)
                );

                CREATE INDEX IF NOT EXISTS idx_sol_fingerprint ON solutions(fingerprint_id);
                CREATE INDEX IF NOT EXISTS idx_sol_exception   ON solutions(exception_type);
                CREATE INDEX IF NOT EXISTS idx_sol_node        ON solutions(target_node_id);
            """)
            self._conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def store(
        self,
        fingerprint_id:    str,
        exception_type:    str,
        target_node_id:    str,
        patch:             dict,
        reconstructed_diff: str = "",
        confidence:        float = 0.0,
        adaptation_source: int | None = None,
    ) -> int:
        """
        Persist an approved patch to the solution library.
        Returns the new solution_id.
        """
        patch_type   = patch.get("change_type", patch.get("patch_type", "UNKNOWN"))
        patch_sketch = json.dumps(patch)

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO solutions
                    (fingerprint_id, exception_type, target_node_id, patch_type,
                     patch_sketch, reconstructed_diff, confidence, approved_at,
                     adaptation_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (fingerprint_id, exception_type, target_node_id, patch_type,
                 patch_sketch, reconstructed_diff, confidence, time.time(),
                 adaptation_source),
            )
            self._conn.commit()
            return cur.lastrowid

    def retrieve_exact(self, fingerprint_id: str) -> Solution | None:
        """
        O(1) exact lookup by error fingerprint.
        Returns the most recently approved solution for this fingerprint, or None.
        Used for instant replay of known errors.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT solution_id, fingerprint_id, exception_type, target_node_id,
                       patch_type, patch_sketch, reconstructed_diff, confidence,
                       approved_at, times_replayed, last_replayed_at, adaptation_source
                FROM solutions
                WHERE fingerprint_id = ?
                ORDER BY approved_at DESC
                LIMIT 1
                """,
                (fingerprint_id,),
            ).fetchone()

        if not row:
            return None

        sol = self._row_to_solution(row)
        # Update replay stats
        self._mark_replayed(sol.solution_id)
        return sol

    def retrieve_similar(
        self,
        exception_type: str,
        target_node_id: str,
        limit: int = 3,
    ) -> list[Solution]:
        """
        Case-Based Reasoning: find solutions for structurally similar errors.

        Priority:
          1. Same exception_type + same target_node_id (most similar)
          2. Same exception_type + different node (same error class, different location)
          3. Different exception_type + same node (same code location, different error)

        This enables: "we fixed a NameError in h_abf89e3f before — can we adapt it
        for this AttributeError in h_abf89e3f?" or "we fixed NameError before in
        h_39ca9205 — can we adapt for NameError in h_abf89e3f (similar structure)?"
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT solution_id, fingerprint_id, exception_type, target_node_id,
                       patch_type, patch_sketch, reconstructed_diff, confidence,
                       approved_at, times_replayed, last_replayed_at, adaptation_source,
                       -- Similarity score: exact match scores higher
                       CASE
                           WHEN exception_type = ? AND target_node_id = ? THEN 3
                           WHEN exception_type = ? THEN 2
                           WHEN target_node_id = ? THEN 1
                           ELSE 0
                       END AS similarity_rank
                FROM solutions
                WHERE (exception_type = ? OR target_node_id = ?)
                ORDER BY similarity_rank DESC, confidence DESC, approved_at DESC
                LIMIT ?
                """,
                (exception_type, target_node_id,
                 exception_type, target_node_id,
                 exception_type, target_node_id,
                 limit),
            ).fetchall()

        solutions = []
        for row in rows:
            sol_row = row[:12]  # exclude similarity_rank
            solutions.append(self._row_to_solution(sol_row))
        return solutions

    def record_adaptation(
        self,
        source_solution_id: int,
        adapted_for_fid:    str,
        adapted_for_node:   str,
        human_modified:     bool = False,
    ) -> None:
        """Track when a stored solution was adapted for a novel case."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO adaptation_log
                    (source_solution_id, adapted_for_fid, adapted_for_node,
                     human_modified, adapted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_solution_id, adapted_for_fid, adapted_for_node,
                 int(human_modified), time.time()),
            )
            self._conn.commit()

    def get_library_stats(self) -> dict:
        """Summary statistics for the solution library."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM solutions").fetchone()[0]
            by_type = self._conn.execute(
                "SELECT exception_type, COUNT(*) FROM solutions GROUP BY exception_type"
            ).fetchall()
            total_replays = self._conn.execute(
                "SELECT SUM(times_replayed) FROM solutions"
            ).fetchone()[0] or 0
            adaptations = self._conn.execute(
                "SELECT COUNT(*) FROM adaptation_log"
            ).fetchone()[0]

        return {
            "total_solutions": total,
            "total_replays": total_replays,
            "total_adaptations": adaptations,
            "by_exception_type": {row[0]: row[1] for row in by_type},
        }

    def has_solution(self, fingerprint_id: str) -> bool:
        """Fast check: does any solution exist for this fingerprint?"""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM solutions WHERE fingerprint_id = ? LIMIT 1",
                (fingerprint_id,),
            ).fetchone()
        return row is not None

    def wipe_all(self) -> int:
        """
        Delete all stored solutions (and their adaptation log entries).
        Returns the number of rows deleted.
        Used for demo resets: wipe memory so the next run uses the full pipeline.
        Rejection memory (ghost_memory.db fingerprint table) is NOT touched.
        """
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM solutions").fetchone()[0]
            self._conn.execute("DELETE FROM adaptation_log")
            self._conn.execute("DELETE FROM solutions")
            self._conn.commit()
        return count

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SolutionLibrary":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Private ───────────────────────────────────────────────────────────────

    def _row_to_solution(self, row: tuple) -> Solution:
        return Solution(
            solution_id       = row[0],
            fingerprint_id    = row[1],
            exception_type    = row[2],
            target_node_id    = row[3],
            patch_type        = row[4],
            patch_sketch      = row[5],
            reconstructed_diff = row[6],
            confidence        = row[7],
            approved_at       = row[8],
            times_replayed    = row[9],
            last_replayed_at  = row[10],
            adaptation_source = row[11],
        )

    def _mark_replayed(self, solution_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE solutions SET times_replayed = times_replayed + 1, last_replayed_at = ? WHERE solution_id = ?",
                (time.time(), solution_id),
            )
            self._conn.commit()
