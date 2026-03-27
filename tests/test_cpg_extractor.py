"""Tests for the CPG Extractor."""
import pytest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_extractor(tmp_path):
    from src.core.mapper import GhostMapper
    from src.core.cpg_extractor import CPGExtractor
    mapper = GhostMapper(
        db_path=str(tmp_path / "test.db"),
        key_path=str(tmp_path / "test.key"),
    )
    extractor = CPGExtractor(mapper, service="test-svc")
    return extractor, mapper


SAMPLE_CODE = '''
def authenticate(jwt_token: str, user_id: int) -> bool:
    """Verify a JWT token for the given user."""
    decoded = decode_jwt(jwt_token)
    if not decoded:
        return False
    return decoded.user_id == user_id


def get_user(user_id: int) -> dict:
    result = db_query("SELECT * FROM users WHERE id = ?", user_id)
    return result


def handler(request, user_id: int):
    if not authenticate(request.token, user_id):
        raise PermissionError("Access denied")
    return get_user(user_id)
'''


def test_extracts_nodes(tmp_path):
    """Three functions → three nodes."""
    extractor, mapper = make_extractor(tmp_path)
    payload = extractor.extract(SAMPLE_CODE)
    func_nodes = [n for n in payload.nodes if n.type == "Function"]
    assert len(func_nodes) == 3
    mapper.close()


def test_parameters_are_hashed(tmp_path):
    """Parameters must be hashed (h_XXXXXXXX format)."""
    extractor, mapper = make_extractor(tmp_path)
    payload = extractor.extract(SAMPLE_CODE)
    for node in payload.nodes:
        if node.type == "Function":
            for param in node.parameters:
                assert param["name"].startswith("h_"), f"Unhashed param: {param}"
    mapper.close()


def test_call_edges_extracted(tmp_path):
    """handler() calls authenticate() and get_user() → 2 CALLS edges."""
    extractor, mapper = make_extractor(tmp_path)
    payload = extractor.extract(SAMPLE_CODE)
    call_edges = [e for e in payload.edges if e.type == "CALLS"]
    assert len(call_edges) >= 2, f"Expected ≥2 CALLS edges, got {len(call_edges)}"
    mapper.close()


def test_data_flow_edges_extracted(tmp_path):
    """user_id parameter flows into authenticate/get_user → DATA_FLOW edges."""
    extractor, mapper = make_extractor(tmp_path)
    payload = extractor.extract(SAMPLE_CODE)
    data_flow_edges = [e for e in payload.edges if e.type == "DATA_FLOW"]
    assert len(data_flow_edges) >= 1, "Expected at least 1 DATA_FLOW edge"
    mapper.close()


def test_docstring_extracted(tmp_path):
    """authenticate() has a docstring - should be captured."""
    extractor, mapper = make_extractor(tmp_path)
    payload = extractor.extract(SAMPLE_CODE)
    auth_nodes = [n for n in payload.nodes if n.docstring and "JWT" in n.docstring]
    assert len(auth_nodes) == 1
    mapper.close()


def test_return_type_extracted(tmp_path):
    """authenticate() has return type bool."""
    extractor, mapper = make_extractor(tmp_path)
    payload = extractor.extract(SAMPLE_CODE)
    for node in payload.nodes:
        if node.return_type == "bool":
            return  # found it
    pytest.fail("No Function node with return_type='bool' found")


def test_empty_code_produces_empty_payload(tmp_path):
    extractor, mapper = make_extractor(tmp_path)
    payload = extractor.extract("")
    assert payload.nodes == []
    assert payload.edges == []
    mapper.close()


def test_to_dict_serializable(tmp_path):
    """CPGPayload.to_dict() should be JSON-serializable."""
    import json
    extractor, mapper = make_extractor(tmp_path)
    payload = extractor.extract(SAMPLE_CODE)
    d = payload.to_dict()
    # Should not raise
    json.dumps(d)
    mapper.close()
