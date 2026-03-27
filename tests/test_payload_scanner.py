"""Tests for the Payload Scanner."""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.payload_scanner import PayloadScanner


@pytest.fixture
def scanner():
    return PayloadScanner()


CLEAN_PAYLOAD = {
    "nodes": [
        {"id": "h_aabb1122", "type": "Function", "parameters": [{"name": "h_ccdd3344", "type": "str"}]},
        {"id": "h_eeff5566", "type": "Class", "parameters": []},
    ],
    "edges": [
        {"from": "h_aabb1122", "to": "h_eeff5566", "type": "CALLS"},
    ],
}

TAINTED_PAYLOAD = {
    "nodes": [
        {"id": "verify_user_token", "type": "Function", "parameters": []},  # plaintext!
    ],
    "edges": [],
}


def test_clean_payload_passes(scanner):
    result = scanner.scan(CLEAN_PAYLOAD)
    assert result["passed"] is True
    assert result["violations"] == []


def test_tainted_payload_fails(scanner):
    result = scanner.scan(TAINTED_PAYLOAD)
    assert result["passed"] is False
    assert len(result["violations"]) >= 1


def test_risky_word_in_node_id_fails(scanner):
    payload = {
        "nodes": [{"id": "password", "type": "Function", "parameters": []}],
        "edges": [],
    }
    result = scanner.scan(payload)
    assert result["passed"] is False


def test_risky_word_in_edge_fails(scanner):
    payload = {
        "nodes": [],
        "edges": [{"from": "authenticate", "to": "h_aabb1122", "type": "CALLS"}],
    }
    result = scanner.scan(payload)
    assert result["passed"] is False


def test_namespaced_id_is_allowed(scanner):
    """Service-namespaced IDs (svc::h_xxx) should pass."""
    payload = {
        "nodes": [{"id": "my-svc::h_aabb1122", "type": "Function", "parameters": []}],
        "edges": [{"from": "my-svc::h_aabb1122", "to": "h_ccdd3344", "type": "CALLS"}],
    }
    result = scanner.scan(payload)
    assert result["passed"] is True


def test_scanned_id_count_is_correct(scanner):
    """scanned_ids count should match total node + param + edge endpoint IDs."""
    result = scanner.scan(CLEAN_PAYLOAD)
    # 2 nodes + 1 param + 2 edge endpoints = 5
    assert result["scanned_ids"] == 5


def test_empty_payload_passes(scanner):
    result = scanner.scan({"nodes": [], "edges": []})
    assert result["passed"] is True
