"""Tests for the Merkle Delta Protocol."""
import pytest
from pathlib import Path
import sys, secrets

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.delta_protocol import (
    build_merkle_root,
    compute_changed_nodes,
    create_manifest,
    verify_manifest,
)


SAMPLE_NODES = [
    {"id": "h_aabb1122", "type": "Function", "is_async": False, "parameters": [], "return_type": "None"},
    {"id": "h_ccdd3344", "type": "Function", "is_async": True, "parameters": [], "return_type": "dict"},
    {"id": "h_eeff5566", "type": "Class", "is_async": False, "parameters": [], "return_type": "Any"},
]


def test_merkle_root_is_deterministic():
    """Same nodes in same order → same Merkle root."""
    r1 = build_merkle_root(SAMPLE_NODES)
    r2 = build_merkle_root(SAMPLE_NODES)
    assert r1 == r2
    assert len(r1) == 64  # SHA-256 hex


def test_merkle_root_changes_on_node_modification():
    """Changing any node → different root."""
    r1 = build_merkle_root(SAMPLE_NODES)
    modified = [*SAMPLE_NODES]
    modified[0] = {**SAMPLE_NODES[0], "return_type": "str"}
    r2 = build_merkle_root(modified)
    assert r1 != r2


def test_empty_nodes_merkle_root():
    """Empty list → well-defined empty hash."""
    r = build_merkle_root([])
    assert len(r) == 64


def test_compute_changed_nodes_detects_change():
    """One node changed → only that node in delta."""
    prev = SAMPLE_NODES
    curr = [
        {**SAMPLE_NODES[0], "return_type": "str"},  # changed
        SAMPLE_NODES[1],
        SAMPLE_NODES[2],
    ]
    changed = compute_changed_nodes(prev, curr)
    assert len(changed) == 1
    assert changed[0]["id"] == "h_aabb1122"


def test_compute_changed_nodes_new_node():
    """Entirely new node appears in delta."""
    prev = SAMPLE_NODES[:2]
    curr = [*SAMPLE_NODES]  # added h_eeff5566
    changed = compute_changed_nodes(prev, curr)
    assert len(changed) == 1
    assert changed[0]["id"] == "h_eeff5566"


def test_compute_changed_nodes_no_change():
    """Identical prev and curr → empty delta."""
    changed = compute_changed_nodes(SAMPLE_NODES, SAMPLE_NODES)
    assert changed == []


def test_create_and_verify_manifest():
    """Create a manifest, verify its HMAC signature."""
    session_key = secrets.token_bytes(32)
    payload = {
        "service": "test-svc",
        "file_path": "/demo/app_test.py",
        "nodes": SAMPLE_NODES,
        "edges": [],
    }
    manifest = create_manifest(payload, [], session_key)
    assert verify_manifest(manifest, session_key)


def test_tampered_manifest_fails_verify():
    """Modifying any field after signing → signature verification fails."""
    session_key = secrets.token_bytes(32)
    payload = {
        "service": "test-svc",
        "file_path": "/demo/app_test.py",
        "nodes": SAMPLE_NODES,
        "edges": [],
    }
    manifest = create_manifest(payload, [], session_key)
    manifest.merkle_root = "tampered"
    assert not verify_manifest(manifest, session_key)


def test_wrong_key_fails_verify():
    """Verifying with a different key → fails."""
    key1 = secrets.token_bytes(32)
    key2 = secrets.token_bytes(32)
    payload = {
        "service": "test-svc",
        "file_path": "/demo/app_test.py",
        "nodes": SAMPLE_NODES,
        "edges": [],
    }
    manifest = create_manifest(payload, [], key1)
    assert not verify_manifest(manifest, key2)


def test_manifest_contains_only_changed_nodes():
    """Delta manifest contains only nodes that changed."""
    session_key = secrets.token_bytes(32)
    prev_nodes = SAMPLE_NODES
    new_node = {"id": "h_new00001", "type": "Function", "is_async": False, "parameters": [], "return_type": "str"}
    curr_nodes = [*SAMPLE_NODES, new_node]
    payload = {
        "service": "test-svc",
        "file_path": "/test.py",
        "nodes": curr_nodes,
        "edges": [],
    }
    manifest = create_manifest(payload, prev_nodes, session_key)
    assert len(manifest.changed_nodes) == 1
    assert manifest.changed_nodes[0]["id"] == "h_new00001"
