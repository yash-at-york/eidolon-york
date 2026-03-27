"""
Eidolon - Outbound Payload Scanner
Guards the privacy boundary before any data leaves the machine.

Scans outbound delta manifests for plaintext Python identifiers.
If the node_id fields contain recognisable dictionary words instead of
h_XXXXXXXX hashes, SOMETHING in the hashing pipeline has failed silently.
This scanner is the last line of defence.
"""
from __future__ import annotations

import re

# Hash pattern 
_HASH_PATTERN = re.compile(r"^h_[0-9a-f]{8}$")

# Known risky plaintext - common Python identifiers that must never leak 
# This is a conservative list. Extend as needed for specific domain risk.
_RISKY_WORDS: set[str] = {
    # Python builtins
    "self", "cls", "args", "kwargs", "request", "response", "session",
    "config", "settings", "env", "password", "secret", "token", "key",
    "user", "username", "email", "api_key", "auth", "database", "db",
    "connection", "cursor", "query", "result", "data", "payload",
    "client", "server", "host", "port", "url", "path",
    # Common framework identifiers
    "app", "router", "handler", "middleware", "endpoint", "route",
    "model", "schema", "view", "controller", "service", "repository",
    "validate", "serialize", "deserialize", "encode", "decode",
    "encrypt", "decrypt", "hash", "verify", "authenticate", "authorize",
    "login", "logout", "register", "create", "read", "update", "delete",
    "get", "post", "put", "patch", "delete",
    "init", "setup", "teardown", "cleanup", "close", "open",
    "load", "save", "write", "read", "parse", "format",
    # FastAPI / SQLAlchemy specific (for app_test.py demo)
    "FastAPI", "HTTPException", "Session", "Query", "filter",
    "decode_jwt", "verify_user_token", "create_user_token",
    "get_user_token", "retrieve_user_token", "user_id", "jwt_token",
    "db_session", "decoded_payload", "user_record",
}


class PayloadScanner:
    """
    Scans CPG payloads and delta manifests for privacy violations.

    A violation is any node_id that:
    1. Does NOT match the expected h_XXXXXXXX pattern, AND
    2. Appears in the risky-word dictionary (or is a non-hash plaintext word)
    """

    def scan(self, payload: dict) -> dict:
        """
        Scan a CPGPayload dict for plaintext identifier leakage.

        Returns:
            {"passed": bool, "violations": [str], "scanned_ids": int}
        """
        violations: list[str] = []
        scanned = 0

        # Scan nodes
        for node in payload.get("nodes", []):
            node_id = node.get("id", "")
            scanned += 1
            v = self._check_id(node_id, context="node")
            if v:
                violations.append(v)

            # Scan parameter hashes
            for param in node.get("parameters", []):
                param_id = param.get("name", "")
                scanned += 1
                v = self._check_id(param_id, context="param")
                if v:
                    violations.append(v)

        # Scan edges
        for edge in payload.get("edges", []):
            for field in ("from", "to"):
                edge_id = edge.get(field, "")
                scanned += 1
                v = self._check_id(edge_id, context=f"edge.{field}")
                if v:
                    violations.append(v)

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "scanned_ids": scanned,
        }

    def _check_id(self, value: str, context: str) -> str | None:
        """Return a violation string if the value is suspect, else None."""
        if not value:
            return None

        # Anything that is already in hash format - safe
        if _HASH_PATTERN.match(value):
            return None

        # Special structural IDs - allowed
        if "::" in value or value.startswith("__"):
            return None

        # Check if it looks like a plaintext identifier
        # A hash will always be h_{8 hex chars}; anything else is suspicious
        if not value.startswith("h_"):
            # Only flag if it's a known risky word or looks like a Python identifier
            if value in _RISKY_WORDS or re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", value):
                return f"[{context}] Plaintext identifier leaked: '{value}'"

        return None
