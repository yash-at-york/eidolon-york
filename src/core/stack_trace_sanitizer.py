"""
Eidolon — Ghost Stack Trace Sanitizer

PHILOSOPHY:
  The project's core promise: "code never leaves the machine in plaintext."
  Stack traces are the richest debugging signal — but they contain real function
  names, variable names, file paths, and code snippets.

  This module is the LOCAL-ONLY sanitization pass that must run BEFORE anything
  touches the network. It produces a "Ghost Stack Trace" where:

    HASHED (via GhostMapper, same key as CPG):
      - Function names      → h_XXXXXXXX  (matches CPG node IDs)
      - Variable names      → h_XXXXXXXX  (matches parameter hashes)
      - Called names in msg → h_XXXXXXXX  (matches CALLS edge targets)
      - File path dirs      → stripped    (only basename preserved)

    PRESERVED AS-IS (not PII, needed for structural reasoning):
      - Line numbers        → 9, 14, 24  (essential for exact node lookup)
      - Exception types     → NameError, AttributeError (generic Python vocab)
      - Operators/keywords  → =, if, return, for (syntax, not identity)
      - Argument counts     → 1, 2, 3    (structural, not identity)

  The output ghost_stack_trace is safe to transmit to the LLM cloud API because
  all user-defined identifiers are hashed with the same HMAC key used for CPG
  extraction — meaning the hashes are consistent. h_eacad148 in the ghost stack
  trace IS the same node h_eacad148 in FalkorDB and Qdrant.

WHAT THIS ENABLES:
  - Exact CPG node lookup by line number (no vector search needed)
  - LLM can reason: "NameError on line 9 in h_abf89e3f calling h_eacad148,
    but no IMPORTS edge for h_eacad148 → it's called but never imported"
  - Precise structural_score in validation (exact node is known)
  - Exception-type-aware retrieval strategies (already typed per exception class)

SUPPORTED INPUT FORMATS:
  1. Structured dict (preferred):
     {
       "error_message": "NameError: name 'decode_jwt' is not defined",
       "traceback": [
         {"file": "/abs/path/app.py", "line": 9, "function": "verify_user_token",
          "code": "decoded_payload = decode_jwt(jwt_token)"}
       ]
     }

  2. Raw multiline traceback string (fallback):
     \"\"\"
     Traceback (most recent call last):
       File "app.py", line 9, in verify_user_token
         decoded_payload = decode_jwt(jwt_token)
     NameError: name 'decode_jwt' is not defined
     \"\"\"

  Both produce the same GhostStackTrace output.
"""
from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.mapper import GhostMapper


# ── Python keyword + builtins sets (preserved, not hashed) ───────────────────

_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})

_BUILTINS = frozenset({
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict",
    "set", "tuple", "type", "isinstance", "issubclass", "hasattr", "getattr",
    "setattr", "delattr", "open", "input", "super", "object", "property",
    "staticmethod", "classmethod", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "min", "max", "sum", "abs", "round", "repr",
    "id", "hash", "hex", "oct", "bin", "chr", "ord", "vars", "dir",
    "next", "iter", "callable", "any", "all", "Exception", "ValueError",
    "TypeError", "NameError", "AttributeError", "KeyError", "IndexError",
    "RuntimeError", "StopIteration", "NotImplementedError", "IOError",
    "OSError", "FileNotFoundError", "PermissionError", "TimeoutError",
    "ConnectionError", "ImportError", "ModuleNotFoundError",
    "HTTPException", "self", "cls", "args", "kwargs",
})

_PRESERVE = _KEYWORDS | _BUILTINS


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class GhostFrame:
    """A single sanitized stack frame."""
    file_basename: str      # "app_test.py" — directory stripped
    line_number:   int      # 9 — preserved (not PII)
    ghost_function: str     # "h_abf89e3f" — hashed function name
    ghost_code:    str      # "h_var1 = h_func(h_param)" — hashed identifiers
    original_function: str = ""  # populated temporarily for local resolution only

    def to_dict(self) -> dict:
        return {
            "file_basename":   self.file_basename,
            "line_number":     self.line_number,
            "ghost_function":  self.ghost_function,
            "ghost_code":      self.ghost_code,
        }


@dataclass
class GhostStackTrace:
    """Fully sanitized stack trace — safe for LLM transmission."""
    exception_type:  str                     # "NameError" — preserved
    ghost_message:   str                     # "name 'h_eacad148' is not defined"
    frames:          list[GhostFrame] = field(default_factory=list)
    fault_frame:     GhostFrame | None = None   # innermost frame (most specific)
    root_frame:      GhostFrame | None = None   # outermost frame (entry point)
    exception_class: str = ""                # full Python exception class, e.g. "NameError"
    # Structural hints extracted from exception type (for retrieval strategy)
    structural_hints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "exception_type":  self.exception_type,
            "ghost_message":   self.ghost_message,
            "frames":          [f.to_dict() for f in self.frames],
            "fault_frame":     self.fault_frame.to_dict() if self.fault_frame else None,
            "root_frame":      self.root_frame.to_dict() if self.root_frame else None,
            "exception_class": self.exception_class,
            "structural_hints": self.structural_hints,
        }


# ── Exception type → structural hints mapping ─────────────────────────────────
# This is what answers Q2: "are we dynamic about error solving?"
# Instead of hardcoding HTTP routes, we extract structural hints from ANY
# Python exception class. These hints drive the context fetch strategy.

_EXCEPTION_HINTS: dict[str, dict] = {
    "NameError": {
        "error_category":    "undefined_reference",
        "retrieval_strategy": "find_callers_of_undefined",
        "likely_cause":      "Called identifier is not imported or defined in scope",
        "cpg_edge_focus":    "CALLS,IMPORTS",
        "search_hint":       "import definition scope undefined",
    },
    "AttributeError": {
        "error_category":    "wrong_type_or_missing_method",
        "retrieval_strategy": "find_type_mismatch",
        "likely_cause":      "Method or attribute doesn't exist on the actual runtime type",
        "cpg_edge_focus":    "CALLS,DATA_FLOW",
        "search_hint":       "attribute method call chain type annotation",
    },
    "TypeError": {
        "error_category":    "type_contract_violation",
        "retrieval_strategy": "find_type_boundary",
        "likely_cause":      "Wrong argument type passed to a function",
        "cpg_edge_focus":    "DATA_FLOW",
        "search_hint":       "type annotation parameter return conversion cast",
    },
    "ImportError": {
        "error_category":    "missing_dependency",
        "retrieval_strategy": "find_import_chain",
        "likely_cause":      "Module not installed or circular import",
        "cpg_edge_focus":    "IMPORTS",
        "search_hint":       "import module dependency installation",
    },
    "ModuleNotFoundError": {
        "error_category":    "missing_dependency",
        "retrieval_strategy": "find_import_chain",
        "likely_cause":      "Module not installed",
        "cpg_edge_focus":    "IMPORTS",
        "search_hint":       "import module dependency installation",
    },
    "KeyError": {
        "error_category":    "missing_key",
        "retrieval_strategy": "find_data_access_pattern",
        "likely_cause":      "Dict/mapping access for key that doesn't exist",
        "cpg_edge_focus":    "DATA_FLOW",
        "search_hint":       "dictionary key access default get check",
    },
    "IndexError": {
        "error_category":    "bounds_violation",
        "retrieval_strategy": "find_data_access_pattern",
        "likely_cause":      "List/array access out of bounds",
        "cpg_edge_focus":    "DATA_FLOW",
        "search_hint":       "list index bounds length check",
    },
    "ValueError": {
        "error_category":    "invalid_value",
        "retrieval_strategy": "find_validation_gap",
        "likely_cause":      "Value doesn't meet expected constraints",
        "cpg_edge_focus":    "DATA_FLOW,CALLS",
        "search_hint":       "validation check constraint guard",
    },
    "RuntimeError": {
        "error_category":    "runtime_state",
        "retrieval_strategy": "find_state_corruption",
        "likely_cause":      "Unexpected runtime state or recursion",
        "cpg_edge_focus":    "CALLS,DATA_FLOW",
        "search_hint":       "state guard condition check",
    },
    "TimeoutError": {
        "error_category":    "timeout",
        "retrieval_strategy": "find_async_or_network",
        "likely_cause":      "Network call or blocking operation exceeded time limit",
        "cpg_edge_focus":    "CALLS",
        "search_hint":       "async await timeout retry network sleep",
    },
    "ConnectionError": {
        "error_category":    "network",
        "retrieval_strategy": "find_async_or_network",
        "likely_cause":      "Network or database connection failure",
        "cpg_edge_focus":    "CALLS,IMPORTS",
        "search_hint":       "network connection retry database pool",
    },
    "PermissionError": {
        "error_category":    "auth",
        "retrieval_strategy": "find_auth_guard",
        "likely_cause":      "Unauthorized access attempt",
        "cpg_edge_focus":    "CALLS",
        "search_hint":       "auth guard permission check middleware",
    },
    "HTTPException": {
        "error_category":    "http_error",
        "retrieval_strategy": "find_auth_guard",
        "likely_cause":      "HTTP endpoint raised structured exception",
        "cpg_edge_focus":    "CALLS",
        "search_hint":       "http endpoint guard validation auth",
    },
    "AssertionError": {
        "error_category":    "assertion_failure",
        "retrieval_strategy": "find_validation_gap",
        "likely_cause":      "Precondition or invariant assertion failed",
        "cpg_edge_focus":    "DATA_FLOW",
        "search_hint":       "assertion precondition invariant guard",
    },
    "RecursionError": {
        "error_category":    "infinite_recursion",
        "retrieval_strategy": "find_state_corruption",
        "likely_cause":      "Infinite recursion — missing base case or cycle",
        "cpg_edge_focus":    "CALLS",
        "search_hint":       "recursion base case termination",
    },
}

_DEFAULT_HINTS = {
    "error_category":    "unknown",
    "retrieval_strategy": "general_similarity",
    "likely_cause":      "Unknown exception type",
    "cpg_edge_focus":    "CALLS,DATA_FLOW,IMPORTS",
    "search_hint":       "",
}


# ── Raw traceback parser ───────────────────────────────────────────────────────

_FRAME_RE    = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
_EXCEPT_RE   = re.compile(r'^(\w+(?:\.\w+)*Error|\w+(?:\.\w+)*Exception|HTTPException|TimeoutError'
                           r'|ConnectionError|PermissionError|ModuleNotFoundError): ?(.*)$',
                           re.MULTILINE)
_CODE_LINE_RE = re.compile(r'^\s{4}(.+)$')


def _parse_raw_traceback(raw: str) -> tuple[list[dict], str, str]:
    """
    Parse a raw Python traceback string.
    Returns: (frames_list, exception_type, exception_message)
    """
    frames: list[dict] = []
    lines  = raw.strip().splitlines()

    i = 0
    while i < len(lines):
        m = _FRAME_RE.search(lines[i])
        if m:
            file_path = m.group(1)
            line_no   = int(m.group(2))
            func_name = m.group(3)
            code_line = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                cm = _CODE_LINE_RE.match(next_line)
                if cm and not _FRAME_RE.search(next_line):
                    code_line = cm.group(1).strip()
                    i += 1
            frames.append({
                "file": file_path,
                "line": line_no,
                "function": func_name,
                "code": code_line,
            })
        i += 1

    # Extract exception line (last meaningful line)
    exc_type, exc_msg = "UnknownError", ""
    for line in reversed(lines):
        m = _EXCEPT_RE.match(line.strip())
        if m:
            exc_type = m.group(1).split(".")[-1]   # take last component of dotted name
            exc_msg  = m.group(2).strip()
            break

    return frames, exc_type, exc_msg


# ── Identifier tokenizer + hasher ─────────────────────────────────────────────

def _hash_code_line(code: str, mapper: GhostMapper) -> str:
    """
    Tokenize a single line of Python code and hash all NAME tokens
    that are not keywords or known builtins.
    Preserves operators, punctuation, literals, and whitespace structure.
    Falls back to returning the original line if tokenization fails.
    """
    if not code.strip():
        return code
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except tokenize.TokenError:
        # Tokenization can fail on partial lines — just hash word-tokens naively
        return _hash_words_naive(code, mapper)

    parts: list[str] = []
    for tok_type, tok_str, _, _, _ in tokens:
        if tok_type == tokenize.NAME and tok_str not in _PRESERVE:
            parts.append(mapper.hash(tok_str))
        elif tok_type == tokenize.ENDMARKER:
            break
        else:
            parts.append(tok_str)

    return " ".join(p for p in parts if p.strip())


def _hash_words_naive(text: str, mapper: GhostMapper) -> str:
    """Naive fallback: hash any word that looks like a Python identifier."""
    _IDENT_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    def _replace(m: re.Match) -> str:
        word = m.group(1)
        return word if word in _PRESERVE else mapper.hash(word)
    return _IDENT_RE.sub(_replace, text)


def _hash_exception_message(msg: str, mapper: GhostMapper) -> str:
    """
    Hash any Python identifiers embedded inside an exception message.
    Examples:
      "name 'decode_jwt' is not defined"
       → "name 'h_eacad148' is not defined"
      "object of type 'NoneType' has no attribute 'last'"
       → "object of type 'NoneType' has no attribute 'h_XXXX'"
    Preserves the English sentence structure.
    """
    # Hash single-quoted identifiers: 'identifier'
    _QUOTED_RE = re.compile(r"'([a-zA-Z_][a-zA-Z0-9_.]*)'")
    def _replace_quoted(m: re.Match) -> str:
        name = m.group(1)
        # Split on dots to handle "module.attr" — hash each component
        parts = [mapper.hash(p) if p not in _PRESERVE else p for p in name.split(".")]
        return f"'{'.'.join(parts)}'"
    return _QUOTED_RE.sub(_replace_quoted, msg)


# ── Main sanitizer ─────────────────────────────────────────────────────────────

class StackTraceSanitizer:
    """
    Converts raw error events (structured dict or raw string) into
    a GhostStackTrace where all user-defined identifiers are hashed.

    The GhostStackTrace is safe for cloud LLM transmission.
    The hashes are consistent with CPG node IDs (same GhostMapper, same session key).

    Usage:
        sanitizer = StackTraceSanitizer(mapper)

        # From structured event (preferred):
        ghost = sanitizer.sanitize({
            "error_message": "NameError: name 'decode_jwt' is not defined",
            "traceback": [
                {"file": "app_test.py", "line": 9, "function": "verify_user_token",
                 "code": "decoded_payload = decode_jwt(jwt_token)"}
            ]
        })

        # From raw string (fallback for legacy log lines):
        ghost = sanitizer.sanitize_raw(raw_traceback_string)

        # Always safe to transmit:
        state["ghost_stack_trace"] = ghost.to_dict()
    """

    def __init__(self, mapper: GhostMapper) -> None:
        self._mapper = mapper

    def sanitize(self, event: dict | str) -> GhostStackTrace:
        """
        Main entry point. Accepts either a structured event dict or a raw string.
        Returns a fully sanitized GhostStackTrace.
        """
        if isinstance(event, str):
            return self.sanitize_raw(event)

        # Structured path
        error_message = event.get("error_message", "")
        raw_frames    = event.get("traceback", [])

        # Parse exception type and message from error_message field
        exc_type, exc_msg = self._parse_error_message(error_message)

        frames = [self._sanitize_frame(f) for f in raw_frames]
        ghost_msg = _hash_exception_message(exc_msg, self._mapper)

        return self._build(exc_type, ghost_msg, frames)

    def sanitize_raw(self, raw: str) -> GhostStackTrace:
        """Parse and sanitize a raw multiline Python traceback string."""
        raw_frames, exc_type, exc_msg = _parse_raw_traceback(raw)
        frames = [self._sanitize_frame(f) for f in raw_frames]
        ghost_msg = _hash_exception_message(exc_msg, self._mapper)
        return self._build(exc_type, ghost_msg, frames)

    def sanitize_simple(self, error_string: str) -> GhostStackTrace:
        """
        Handle a simple one-line error string (no traceback).
        Used when the system receives legacy log-style errors.
        Returns a minimal GhostStackTrace with no frames but with
        exception type and structural hints extracted.
        """
        # Try to detect exception type from the string first
        exc_type = "UnknownError"
        exc_msg  = error_string

        # Check for known exception types in the string
        for known in _EXCEPTION_HINTS:
            if known in error_string:
                exc_type = known
                break

        ghost_msg = _hash_exception_message(exc_msg, self._mapper)
        return self._build(exc_type, ghost_msg, [])

    # ── Private ───────────────────────────────────────────────────────────────

    def _sanitize_frame(self, raw: dict) -> GhostFrame:
        file_path     = raw.get("file", "")
        line_number   = int(raw.get("line", 0))
        function_name = raw.get("function", "")
        code          = raw.get("code", "")

        # File: strip directory, keep only basename (no server topology leakage)
        file_basename  = Path(file_path).name if file_path else "unknown"

        # Function name: hash via mapper
        ghost_function = (
            self._mapper.hash(function_name)
            if function_name and function_name not in {"<module>", "<lambda>"}
            else function_name
        )

        # Code line: tokenize + hash all non-keyword identifiers
        ghost_code = _hash_code_line(code, self._mapper)

        return GhostFrame(
            file_basename   = file_basename,
            line_number     = line_number,
            ghost_function  = ghost_function,
            ghost_code      = ghost_code,
            original_function = function_name,  # local only, never transmitted
        )

    def _build(self, exc_type: str, ghost_msg: str, frames: list[GhostFrame]) -> GhostStackTrace:
        fault_frame = frames[-1] if frames else None
        root_frame  = frames[0]  if frames else None
        hints       = _EXCEPTION_HINTS.get(exc_type, _DEFAULT_HINTS)

        return GhostStackTrace(
            exception_type   = exc_type,
            ghost_message    = ghost_msg,
            frames           = frames,
            fault_frame      = fault_frame,
            root_frame       = root_frame,
            exception_class  = exc_type,
            structural_hints = hints,
        )

    def _parse_error_message(self, msg: str) -> tuple[str, str]:
        """Parse 'ExcType: message text' into (exc_type, message)."""
        if not msg:
            return "UnknownError", ""
        m = re.match(r'^(\w+(?:\.\w+)*):\s*(.*)', msg, re.DOTALL)
        if m:
            full_type = m.group(1)
            exc_type  = full_type.split(".")[-1]
            return exc_type, m.group(2).strip()
        return "UnknownError", msg
