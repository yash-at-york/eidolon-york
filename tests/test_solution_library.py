"""
Tests for SolutionLibrary — Case-Based Reasoning store.
Uses in-memory SQLite (:memory:) to avoid disk I/O.
"""
from __future__ import annotations

import json
import time
import pytest
from unittest.mock import patch

from src.core.solution_library import SolutionLibrary, Solution


@pytest.fixture
def lib(tmp_path):
    with patch("ghost_config.MEMORY_DB_PATH", str(tmp_path / "test_solutions.db")):
        l = SolutionLibrary(str(tmp_path / "test_solutions.db"))
        yield l
        l.close()


def _make_patch(change_type: str = "ADD_IMPORT", target: str = "h_abc123") -> dict:
    return {
        "target_node":   target,
        "change_type":   change_type,
        "description":   "Add missing import",
        "insert_before": None,
        "new_logic":     "import h_func from h_module",
    }


class TestStore:
    def test_store_returns_id(self, lib):
        sid = lib.store(
            fingerprint_id="fid001",
            exception_type="NameError",
            target_node_id="h_abf89e3f",
            patch=_make_patch(),
            confidence=0.88,
        )
        assert isinstance(sid, int)
        assert sid > 0

    def test_store_multiple(self, lib):
        id1 = lib.store("fid001", "NameError", "h_abc", _make_patch(), confidence=0.9)
        id2 = lib.store("fid002", "AttributeError", "h_def", _make_patch("ADD_GUARD"), confidence=0.7)
        assert id1 != id2

    def test_has_solution_true(self, lib):
        lib.store("fid001", "NameError", "h_abf89e3f", _make_patch(), confidence=0.88)
        assert lib.has_solution("fid001") is True

    def test_has_solution_false(self, lib):
        assert lib.has_solution("nonexistent") is False


class TestExactRetrieval:
    def test_retrieve_exact_hit(self, lib):
        lib.store("fid001", "NameError", "h_abf89e3f", _make_patch(target="h_abf89e3f"), confidence=0.88)
        sol = lib.retrieve_exact("fid001")
        assert sol is not None
        assert sol.fingerprint_id == "fid001"
        assert sol.exception_type == "NameError"
        assert sol.target_node_id == "h_abf89e3f"
        assert abs(sol.confidence - 0.88) < 1e-6

    def test_retrieve_exact_miss(self, lib):
        sol = lib.retrieve_exact("nonexistent")
        assert sol is None

    def test_retrieve_exact_increments_replays(self, lib):
        lib.store("fid001", "NameError", "h_abc", _make_patch(), confidence=0.9)
        sol1 = lib.retrieve_exact("fid001")
        sol2 = lib.retrieve_exact("fid001")
        assert sol2.times_replayed == sol1.times_replayed + 1

    def test_retrieve_returns_most_recent(self, lib):
        lib.store("fid001", "NameError", "h_abc", _make_patch("ADD_IMPORT"), confidence=0.7)
        time.sleep(0.01)
        lib.store("fid001", "NameError", "h_abc", _make_patch("ADD_GUARD"), confidence=0.9)
        sol = lib.retrieve_exact("fid001")
        assert sol.patch_type == "ADD_GUARD"  # most recent
        assert abs(sol.confidence - 0.9) < 1e-6


class TestSimilarRetrieval:
    def test_same_type_same_node_ranked_highest(self, lib):
        lib.store("fid001", "NameError",     "h_node1", _make_patch(), confidence=0.9)
        lib.store("fid002", "NameError",     "h_node2", _make_patch(), confidence=0.8)  # same type, diff node
        lib.store("fid003", "AttributeError","h_node1", _make_patch(), confidence=0.85)  # diff type, same node

        results = lib.retrieve_similar("NameError", "h_node1", limit=3)
        assert len(results) > 0
        # Exact type+node match should come first
        assert results[0].fingerprint_id == "fid001"

    def test_empty_when_no_match(self, lib):
        results = lib.retrieve_similar("TimeoutError", "h_unknown_node", limit=3)
        assert results == []

    def test_returns_at_most_limit(self, lib):
        for i in range(5):
            lib.store(f"fid{i:03d}", "NameError", f"h_node{i}", _make_patch(), confidence=0.8)
        results = lib.retrieve_similar("NameError", "h_node0", limit=2)
        assert len(results) <= 2

    def test_cross_exception_cbr(self, lib):
        """CBR: NameError in h_node1 solved before → find for AttributeError in h_node1."""
        lib.store("fid001", "NameError", "h_node1", _make_patch(), confidence=0.88)
        # Search for different exception type but same node
        results = lib.retrieve_similar("AttributeError", "h_node1", limit=3)
        assert len(results) == 1
        assert results[0].exception_type == "NameError"  # found cross-type by node


class TestSolutionToDict:
    def test_to_dict_keys(self, lib):
        lib.store("fid001", "NameError", "h_abc", _make_patch(), confidence=0.9)
        sol = lib.retrieve_exact("fid001")
        d = sol.to_dict()
        required_keys = {"solution_id", "fingerprint_id", "exception_type",
                         "target_node_id", "patch_type", "patch_sketch", "confidence"}
        assert required_keys.issubset(d.keys())

    def test_to_hypothesis_hint_keys(self, lib):
        lib.store("fid001", "NameError", "h_abc", _make_patch(), confidence=0.85)
        sol = lib.retrieve_exact("fid001")
        hint = sol.to_hypothesis_hint()
        assert "patch_type" in hint
        assert "patch_sketch" in hint
        assert "note" in hint
        assert "confidence" in hint

    def test_patch_sketch_is_json(self, lib):
        patch = _make_patch("ADD_IMPORT")
        lib.store("fid001", "NameError", "h_abc", patch, confidence=0.9)
        sol = lib.retrieve_exact("fid001")
        # Should be valid JSON
        parsed = json.loads(sol.patch_sketch)
        assert parsed.get("change_type") == "ADD_IMPORT"


class TestStats:
    def test_stats_empty(self, lib):
        stats = lib.get_library_stats()
        assert stats["total_solutions"] == 0
        assert stats["total_replays"] == 0

    def test_stats_after_store(self, lib):
        lib.store("fid001", "NameError", "h_abc", _make_patch(), confidence=0.9)
        lib.store("fid002", "AttributeError", "h_def", _make_patch("ADD_GUARD"), confidence=0.8)
        stats = lib.get_library_stats()
        assert stats["total_solutions"] == 2
        assert "NameError" in stats["by_exception_type"]
        assert "AttributeError" in stats["by_exception_type"]


class TestAdaptationLog:
    def test_record_adaptation(self, lib):
        sid = lib.store("fid001", "NameError", "h_abc", _make_patch(), confidence=0.9)
        lib.record_adaptation(sid, "fid002", "h_xyz", human_modified=True)
        # Verify via stats
        stats = lib.get_library_stats()
        assert stats["total_adaptations"] == 1
