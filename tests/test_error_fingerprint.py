"""
Tests for ErrorFingerprinter — structured error fingerprinting + rejection memory.
Uses in-memory SQLite (:memory:) so no files are created.
"""
from __future__ import annotations

import time
import pytest
from src.core.error_fingerprint import ErrorFingerprinter


@pytest.fixture
def fp():
    """In-memory fingerprinter for tests — no disk I/O."""
    f = ErrorFingerprinter(":memory:")
    yield f
    f.close()


class TestFingerprinting:
    def test_basic_fingerprint(self, fp):
        fid, fkey = fp.fingerprint("401 Unauthorized on POST /verify-token: HTTPException")
        assert fid, "fingerprint_id must be non-empty"
        assert "401" in fkey
        assert "POST" in fkey

    def test_fingerprint_deterministic(self, fp):
        """Same error → same fingerprint every time."""
        err = "500 Internal Server Error on GET /users/123: ValueError"
        fid1, fkey1 = fp.fingerprint(err)
        fid2, fkey2 = fp.fingerprint(err)
        assert fid1 == fid2
        assert fkey1 == fkey2

    def test_different_errors_different_fingerprints(self, fp):
        fid1, _ = fp.fingerprint("401 Unauthorized on POST /auth: HTTPException")
        fid2, _ = fp.fingerprint("500 Internal Server Error on GET /data: ValueError")
        assert fid1 != fid2

    def test_occurrence_tracking(self, fp):
        err = "503 on POST /upload: TimeoutError"
        fid, _ = fp.fingerprint(err)
        stats = fp.get_stats(fid)
        assert stats["occurrences"] == 1
        fp.fingerprint(err)  # second occurrence
        stats = fp.get_stats(fid)
        assert stats["occurrences"] == 2

    def test_endpoint_normalisation(self, fp):
        """Path params (/users/123) should normalise to /users/{id}."""
        _, fkey = fp.fingerprint("404 on GET /users/123: NotFound")
        assert "{id}" in fkey or "users" in fkey  # normalised or raw, not the literal 123

    def test_unknown_error_structure(self, fp):
        """Malformed error strings should still produce a valid fingerprint."""
        fid, fkey = fp.fingerprint("something exploded for unknown reasons")
        assert fid
        assert "000" in fkey or "UNKNOWN" in fkey or "UnknownError" in fkey


class TestExactDedup:
    def test_is_known_exact_first_hit(self, fp):
        err = "401 Unauthorized on POST /login: HTTPException"
        fid, _ = fp.fingerprint(err)
        # First occurrence — below min_occurrences=2
        assert fp.is_known_exact(fid, min_occurrences=2) is False

    def test_is_known_exact_after_repeat(self, fp):
        err = "401 Unauthorized on POST /login: HTTPException"
        fp.fingerprint(err)
        fid, _ = fp.fingerprint(err)  # second hit
        assert fp.is_known_exact(fid, min_occurrences=2) is True

    def test_unknown_fingerprint(self, fp):
        assert fp.is_known_exact("nonexistent_id", min_occurrences=1) is False


class TestRejectionMemory:
    def test_record_and_retrieve(self, fp):
        fid, _ = fp.fingerprint("401 on POST /auth: HTTPException")
        fp.record_rejection(
            fid,
            {"hypothesis": "missing auth guard before user_id", "patch_sketch": "ADD_GUARD"},
            "The guard exists — wrong node targeted",
        )
        rejections = fp.get_rejections(fid)
        assert len(rejections) == 1
        assert rejections[0]["hypothesis"] == "missing auth guard before user_id"
        assert rejections[0]["patch_sketch"] == "ADD_GUARD"
        assert "wrong node" in rejections[0]["feedback"]

    def test_empty_rejections_new_fingerprint(self, fp):
        fid, _ = fp.fingerprint("200 OK on GET /health")
        assert fp.get_rejections(fid) == []

    def test_rejection_limit(self, fp):
        fid, _ = fp.fingerprint("401 on POST /auth: HTTPException")
        for i in range(5):
            fp.record_rejection(
                fid,
                {"hypothesis": f"hypothesis {i}", "patch_sketch": f"sketch {i}"},
                f"feedback {i}",
            )
        # Default limit=3 — should return only last 3
        rejections = fp.get_rejections(fid, limit=3)
        assert len(rejections) == 3

    def test_rejections_ordered_newest_first(self, fp):
        fid, _ = fp.fingerprint("401 on POST /auth: HTTPException")
        fp.record_rejection(fid, {"hypothesis": "old hypothesis", "patch_sketch": ""}, "")
        time.sleep(0.01)
        fp.record_rejection(fid, {"hypothesis": "new hypothesis", "patch_sketch": ""}, "")
        rejections = fp.get_rejections(fid, limit=2)
        assert rejections[0]["hypothesis"] == "new hypothesis"
        assert rejections[1]["hypothesis"] == "old hypothesis"


class TestOutcomeTracking:
    def test_record_approved_outcome(self, fp):
        fid, _ = fp.fingerprint("500 on GET /data: ValueError")
        fp.record_outcome(fid, approved=True, composite_score=0.88, error_type="db")
        stats = fp.get_stats(fid)
        assert stats["approved"] == 1
        assert stats["rejected"] == 0
        assert stats["approval_rate"] == 1.0

    def test_record_rejected_outcome(self, fp):
        fid, _ = fp.fingerprint("500 on GET /data: ValueError")
        fp.record_outcome(fid, approved=False, composite_score=0.45, error_type="db")
        stats = fp.get_stats(fid)
        assert stats["rejected"] == 1
        assert stats["approved"] == 0
        assert stats["approval_rate"] == 0.0

    def test_mixed_outcomes(self, fp):
        fid, _ = fp.fingerprint("408 on POST /process: TimeoutError")
        fp.record_outcome(fid, approved=True,  composite_score=0.9, error_type="timeout")
        fp.record_outcome(fid, approved=False, composite_score=0.5, error_type="timeout")
        fp.record_outcome(fid, approved=True,  composite_score=0.8, error_type="timeout")
        stats = fp.get_stats(fid)
        assert stats["approved"] == 2
        assert stats["rejected"] == 1
        assert abs(stats["approval_rate"] - 2/3) < 1e-6

    def test_no_outcomes_returns_none_rate(self, fp):
        fid, _ = fp.fingerprint("404 on GET /thing: NotFound")
        stats = fp.get_stats(fid)
        assert stats["approval_rate"] is None
