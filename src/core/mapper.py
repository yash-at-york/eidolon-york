"""
Eidolon - HMAC-SHA256 Mapper
Provides privacy-preserving identifier hashing with a rotating per-session key.
The mapping DB (hash → original name) NEVER leaves this machine.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path
from threading import Lock

import duckdb

from ghost_config import MAPPER_DB_PATH, SESSION_KEY_PATH



# Session Key Management


def _load_or_create_session_key(key_path: str) -> bytes:
    """
    Load a persisted session key or generate a new one.
    The key is a 32-byte random secret stored only on the local machine.
    It is used as the HMAC key - rotating it invalidates all previous hashes
    and forces a re-sync to the cloud (intentional security property).
    """
    p = Path(key_path)
    if p.exists():
        return bytes.fromhex(p.read_text().strip())
    key = secrets.token_bytes(32)
    p.write_text(key.hex())
    return key



# Mapper


class GhostMapper:
    """
    Thread-safe HMAC-SHA256 identifier mapper backed by DuckDB.

    Security properties:
    - HMAC(identifier, session_key) → deterministic 8-char hex, impossible to
      reverse without the session key.
    - Fixed-salt SHA-256 is NOT used (as in v3) - HMAC with a random session
      key is resistant to dictionary attacks.
    - The .ghost_session_key file is .gitignored and never uploaded anywhere.

    Usage:
        mapper = GhostMapper()
        hashed = mapper.hash("verify_jwt_token")   # → "h_a3f7c211"
        original = mapper.lookup("h_a3f7c211")     # → "verify_jwt_token"
        mapper.close()
    """

    def __init__(
        self,
        db_path: str = MAPPER_DB_PATH,
        key_path: str = SESSION_KEY_PATH,
        read_only: bool = False,
    ) -> None:
        self._key = _load_or_create_session_key(key_path)
        self._lock = Lock()
        self._read_only = read_only

        # DuckDB: read_only=True allows concurrent access alongside an existing
        # write connection (e.g. ghost_daemon or sync_worker holding the file).
        self._db = duckdb.connect(db_path, read_only=read_only)
        if not read_only:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS hash_map (
                    hash_id      TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    created_at   TIMESTAMP DEFAULT current_timestamp
                )
            """)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS sync_checksums (
                    merkle_root  TEXT NOT NULL,
                    synced_at    TIMESTAMP DEFAULT current_timestamp
                )
            """)

    # Public API 

    def hash(self, name: str | bytes) -> str:
        """
        Hash an identifier name and persist the mapping locally.
        Returns a stable string of the form 'h_XXXXXXXX'.
        Hashing the same name twice yields the same result within a session.
        """
        if not name:
            return "h_unknown"
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        name = name.strip()
        if not name:
            return "h_unknown"

        raw = hmac.new(self._key, name.encode("utf-8"), hashlib.sha256).hexdigest()[:8]
        hash_id = f"h_{raw}"

        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO hash_map (hash_id, original_name) VALUES (?, ?)",
                [hash_id, name],
            )
        return hash_id

    def lookup(self, hash_id: str) -> str | None:
        """Reverse a hash back to the original name (local only)."""
        with self._lock:
            result = self._db.execute(
                "SELECT original_name FROM hash_map WHERE hash_id = ?",
                [hash_id],
            ).fetchone()
        return result[0] if result else None

    def save_checksum(self, merkle_root: str) -> None:
        """Persist a Merkle root to track sync integrity."""
        with self._lock:
            self._db.execute(
                "INSERT INTO sync_checksums (merkle_root) VALUES (?)",
                [merkle_root],
            )

    def last_checksum(self) -> str | None:
        """Return the most recent stored Merkle root."""
        with self._lock:
            result = self._db.execute(
                "SELECT merkle_root FROM sync_checksums ORDER BY synced_at DESC LIMIT 1"
            ).fetchone()
        return result[0] if result else None

    def all_mappings(self) -> list[tuple[str, str]]:
        """Return all (hash_id, original_name) pairs - for debugging only."""
        with self._lock:
            return self._db.execute(
                "SELECT hash_id, original_name FROM hash_map ORDER BY hash_id"
            ).fetchall()

    def close(self) -> None:
        """Flush and close the DuckDB connection."""
        with self._lock:
            self._db.close()

    # Context manager 

    def __enter__(self) -> "GhostMapper":
        return self

    def __exit__(self, *_) -> None:
        self.close()
