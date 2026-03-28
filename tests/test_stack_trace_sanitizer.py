"""
Tests for StackTraceSanitizer — privacy + correctness guarantees.
All tests use a fresh GhostMapper with in-memory DuckDB to avoid file I/O.
"""
from __future__ import annotations

import re
import pytest
import duckdb

from src.core.mapper import GhostMapper
from src.core.stack_trace_sanitizer import (
    StackTraceSanitizer,
    GhostFrame,
    GhostStackTrace,
    _EXCEPTION_HINTS,
    _hash_code_line,
    _hash_exception_message,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mapper(tmp_path):
    """GhostMapper with temp files for test isolation."""
    key_file = tmp_path / ".test_session_key"
    db_file  = str(tmp_path / "test_mapper.db")
    return GhostMapper(db_path=db_file, key_path=str(key_file))


@pytest.fixture
def sanitizer(mapper):
    return StackTraceSanitizer(mapper)


# ── Privacy Guarantees ────────────────────────────────────────────────────────

class TestPrivacyGuarantees:
    """Core invariant: no plaintext user-defined identifiers leave the sanitizer."""

    REAL_IDENTIFIERS = [
        "verify_user_token", "decode_jwt", "jwt_token", "db_session",
        "user_id", "decoded_payload", "user_record", "User",
    ]

    def test_no_plaintext_function_names_in_frame(self, sanitizer):
        event = {
            "error_message": "NameError: name 'decode_jwt' is not defined",
            "traceback": [
                {"file": "/home/user/app/app_test.py", "line": 9,
                 "function": "verify_user_token",
                 "code": "decoded_payload = decode_jwt(jwt_token)"}
            ]
        }
        gst = sanitizer.sanitize(event)
        # ghost_function must be a hash, not the real name
        frame = gst.fault_frame
        assert frame is not None
        assert frame.ghost_function.startswith("h_"), f"Expected hash, got: {frame.ghost_function}"
        assert "verify_user_token" not in frame.ghost_function

    def test_no_plaintext_in_ghost_code(self, sanitizer):
        event = {
            "error_message": "NameError: name 'decode_jwt' is not defined",
            "traceback": [
                {"file": "app_test.py", "line": 9,
                 "function": "verify_user_token",
                 "code": "decoded_payload = decode_jwt(jwt_token)"}
            ]
        }
        gst = sanitizer.sanitize(event)
        frame = gst.fault_frame
        # Check no real identifier appears in ghost code
        for ident in self.REAL_IDENTIFIERS:
            assert ident not in frame.ghost_code, \
                f"Plaintext identifier '{ident}' found in ghost_code: {frame.ghost_code}"

    def test_no_directory_path_in_output(self, sanitizer):
        """Full file path stripped to basename only."""
        event = {
            "error_message": "NameError: name 'x' is not defined",
            "traceback": [
                {"file": "/home/user/supersecret/company/project/app.py",
                 "line": 5, "function": "my_func", "code": "x = 1"}
            ]
        }
        gst = sanitizer.sanitize(event)
        frame = gst.fault_frame
        assert frame.file_basename == "app.py"
        assert "/home/user/supersecret" not in str(gst.to_dict())
        assert "company" not in str(gst.to_dict())

    def test_exception_message_identifiers_hashed(self, sanitizer):
        event = {
            "error_message": "NameError: name 'decode_jwt' is not defined",
            "traceback": []
        }
        gst = sanitizer.sanitize(event)
        assert "decode_jwt" not in gst.ghost_message
        assert "h_" in gst.ghost_message  # replaced with hash

    def test_keywords_preserved(self, sanitizer):
        """Python keywords must NOT be hashed — LLM needs them for syntax."""
        event = {
            "error_message": "SyntaxError: invalid syntax",
            "traceback": [
                {"file": "app.py", "line": 3, "function": "my_func",
                 "code": "if not decoded_payload: raise HTTPException"}
            ]
        }
        gst = sanitizer.sanitize(event)
        frame = gst.fault_frame
        # 'if', 'not', 'raise' are keywords — must survive
        assert "if" in frame.ghost_code or "not" in frame.ghost_code or "raise" in frame.ghost_code

    def test_line_numbers_preserved(self, sanitizer):
        """Line numbers are structural metadata, NOT PII — must be preserved."""
        event = {
            "error_message": "NameError: name 'x' is not defined",
            "traceback": [
                {"file": "app.py", "line": 42, "function": "process", "code": "result = x()"}
            ]
        }
        gst = sanitizer.sanitize(event)
        assert gst.fault_frame.line_number == 42

    def test_exception_type_preserved(self, sanitizer):
        """Exception types (NameError, AttributeError) are generic Python vocab, not PII."""
        event = {
            "error_message": "AttributeError: 'Query' object has no attribute 'last'",
            "traceback": []
        }
        gst = sanitizer.sanitize(event)
        assert gst.exception_type == "AttributeError"

    def test_to_dict_contains_no_plaintext(self, sanitizer):
        """The full serialized output must not contain any real identifier."""
        event = {
            "error_message": "NameError: name 'decode_jwt' is not defined",
            "traceback": [
                {"file": "/server/app_test.py", "line": 9,
                 "function": "verify_user_token",
                 "code": "decoded_payload = decode_jwt(jwt_token)"}
            ]
        }
        gst = sanitizer.sanitize(event)
        output = str(gst.to_dict())
        for ident in self.REAL_IDENTIFIERS:
            assert ident not in output, f"Plaintext '{ident}' leaked into serialized output"


# ── Structural Hints (Dynamic, Not Static) ────────────────────────────────────

class TestStructuralHints:
    def test_name_error_hints(self, sanitizer):
        gst = sanitizer.sanitize_simple("NameError: name 'x' is not defined")
        hints = gst.structural_hints
        assert hints["error_category"] == "undefined_reference"
        assert hints["retrieval_strategy"] == "find_callers_of_undefined"

    def test_attribute_error_hints(self, sanitizer):
        gst = sanitizer.sanitize_simple("AttributeError: object has no attribute 'last'")
        hints = gst.structural_hints
        assert hints["error_category"] == "wrong_type_or_missing_method"

    def test_timeout_error_hints(self, sanitizer):
        gst = sanitizer.sanitize_simple("TimeoutError: connection timed out")
        hints = gst.structural_hints
        assert hints["error_category"] == "timeout"

    def test_import_error_hints(self, sanitizer):
        gst = sanitizer.sanitize_simple("ImportError: no module named requests")
        hints = gst.structural_hints
        assert hints["error_category"] == "missing_dependency"
        assert hints["cpg_edge_focus"] == "IMPORTS"

    def test_all_known_exceptions_have_hints(self):
        """Every exception type in the hints dict must have all required keys."""
        required = {"error_category", "retrieval_strategy", "likely_cause",
                    "cpg_edge_focus", "search_hint"}
        for exc_type, hints in _EXCEPTION_HINTS.items():
            missing = required - set(hints.keys())
            assert not missing, f"{exc_type} missing keys: {missing}"


# ── Exception Type Parsing ────────────────────────────────────────────────────

class TestExceptionParsing:
    def test_parse_nameerror(self, sanitizer):
        gst = sanitizer.sanitize({
            "error_message": "NameError: name 'foo' is not defined",
            "traceback": []
        })
        assert gst.exception_type == "NameError"

    def test_parse_attributeerror(self, sanitizer):
        gst = sanitizer.sanitize({
            "error_message": "AttributeError: 'NoneType' has no attribute 'id'",
            "traceback": []
        })
        assert gst.exception_type == "AttributeError"

    def test_parse_dotted_exception(self, sanitizer):
        """Dotted module paths: 'fastapi.exceptions.HTTPException' → 'HTTPException'"""
        gst = sanitizer.sanitize({
            "error_message": "fastapi.exceptions.HTTPException: 401",
            "traceback": []
        })
        assert gst.exception_type == "HTTPException"

    def test_fault_frame_is_last_frame(self, sanitizer):
        """Innermost (fault) frame = last frame in the traceback."""
        event = {
            "error_message": "NameError: name 'x' is not defined",
            "traceback": [
                {"file": "a.py", "line": 1, "function": "outer", "code": "inner()"},
                {"file": "b.py", "line": 9, "function": "inner", "code": "x = foo()"},
            ]
        }
        gst = sanitizer.sanitize(event)
        assert gst.fault_frame.line_number == 9
        assert gst.root_frame.line_number == 1

    def test_raw_traceback_string(self, sanitizer):
        raw = """
Traceback (most recent call last):
  File "app.py", line 9, in verify_user_token
    decoded_payload = decode_jwt(jwt_token)
NameError: name 'decode_jwt' is not defined
"""
        gst = sanitizer.sanitize_raw(raw)
        assert gst.exception_type == "NameError"
        assert gst.fault_frame is not None
        assert gst.fault_frame.line_number == 9
        assert gst.fault_frame.file_basename == "app.py"
        assert "decode_jwt" not in gst.ghost_message  # hashed

    def test_simple_string_no_crash(self, sanitizer):
        """Legacy single-line error strings must not crash."""
        gst = sanitizer.sanitize_simple("401 Unauthorized on POST /verify-token")
        assert gst is not None
        assert gst.frames == []


# ── Hash Code Line ────────────────────────────────────────────────────────────

class TestHashCodeLine:
    def test_assignment_hashed(self, mapper):
        result = _hash_code_line("x = compute(y)", mapper)
        assert "x" not in result
        assert "compute" not in result
        assert "y" not in result
        assert "=" in result  # operator preserved

    def test_keywords_unchanged(self, mapper):
        result = _hash_code_line("if not decoded: raise Exception", mapper)
        assert "if" in result
        assert "not" in result
        assert "raise" in result
        assert "decoded" not in result  # user identifier hashed

    def test_empty_line(self, mapper):
        result = _hash_code_line("", mapper)
        assert result == ""
