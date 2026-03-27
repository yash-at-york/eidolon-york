"""
Eidolon - Merkle Delta Protocol
Generates cryptographically signed delta manifests for cloud sync.

Design:
  - Merkle tree over all node content hashes → single root hash
  - On each file save, only CHANGED nodes are pushed (O(log n) comparison)
  - Manifest is HMAC-signed so the cloud sync worker can verify integrity
  - Temporal anchoring: every manifest records git SHA + ISO timestamp
"""
from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



# Merkle Tree


def _node_hash(node_dict: dict) -> str:
    """Deterministic hash of a single CPG node (content-addressable)."""
    serialized = json.dumps(node_dict, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def build_merkle_root(nodes: list[dict]) -> str:
    """
    Build a Merkle tree from a list of CPG node dicts and return the root hash.
    Empty node list → well-defined empty root hash.
    """
    if not nodes:
        return hashlib.sha256(b"").hexdigest()

    # Leaf layer: hash each node
    layer = [_node_hash(n) for n in nodes]

    # Binary-tree reduction
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left  # odd-count duplication
            combined = hashlib.sha256((left + right).encode()).hexdigest()
            next_layer.append(combined)
        layer = next_layer

    return layer[0]



# Delta computation


def compute_changed_nodes(
    prev_nodes: list[dict],
    curr_nodes: list[dict],
) -> list[dict]:
    """
    Return only nodes whose content changed between prev and curr.
    Uses content-addressable hashing - if the structural JSON of a node
    is identical, it has not changed.
    """
    prev_hashes = {n["id"]: _node_hash(n) for n in prev_nodes}
    curr_hashes = {n["id"]: _node_hash(n) for n in curr_nodes}

    changed = []
    for node in curr_nodes:
        node_id = node["id"]
        if node_id not in prev_hashes or prev_hashes[node_id] != curr_hashes[node_id]:
            changed.append(node)

    return changed



# Manifest


@dataclass
class DeltaManifest:
    service: str
    file_path: str
    git_sha: str
    timestamp: str
    merkle_root: str
    changed_nodes: list[dict]
    all_edges: list[dict]          # full edge list (cheap to transmit, important for CPG)
    signature: str = ""            # HMAC-SHA256, filled in by sign()

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "file_path": self.file_path,
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
            "changed_nodes": self.changed_nodes,
            "all_edges": self.all_edges,
            "signature": self.signature,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DeltaManifest":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})



# Protocol operations


def _get_git_sha(file_path: str) -> str:
    """Return the current HEAD commit SHA for temporal anchoring."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(Path(file_path).parent),
            timeout=2,
        )
        return result.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def create_manifest(
    cpg_payload: dict,
    prev_nodes: list[dict],
    session_key: bytes,
) -> DeltaManifest:
    """
    Create a signed DeltaManifest from a CPGPayload dict.

    Args:
        cpg_payload:  output of CPGExtractor.extract().to_dict()
        prev_nodes:   nodes from the previous sync cycle (for delta computation)
        session_key:  HMAC key bytes from the local session key file
    """
    curr_nodes = cpg_payload["nodes"]
    all_edges = cpg_payload["edges"]

    changed = compute_changed_nodes(prev_nodes, curr_nodes)
    merkle_root = build_merkle_root(curr_nodes)
    git_sha = _get_git_sha(cpg_payload["file_path"])
    timestamp = datetime.now(timezone.utc).isoformat()

    manifest = DeltaManifest(
        service=cpg_payload["service"],
        file_path=cpg_payload["file_path"],
        git_sha=git_sha,
        timestamp=timestamp,
        merkle_root=merkle_root,
        changed_nodes=changed,
        all_edges=all_edges,
    )

    # Sign the manifest (excludes the signature field itself)
    body = json.dumps({k: v for k, v in manifest.to_dict().items() if k != "signature"}, sort_keys=True)
    manifest.signature = hmac.new(session_key, body.encode(), hashlib.sha256).hexdigest()

    return manifest


def verify_manifest(manifest: DeltaManifest, session_key: bytes) -> bool:
    """
    Verify the HMAC signature of a DeltaManifest.
    Returns True if the manifest is authentic and untampered.
    """
    body = json.dumps(
        {k: v for k, v in manifest.to_dict().items() if k != "signature"},
        sort_keys=True,
    )
    expected = hmac.new(session_key, body.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, manifest.signature)
