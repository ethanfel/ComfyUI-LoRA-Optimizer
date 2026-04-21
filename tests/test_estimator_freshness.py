"""Tests for the estimator freshness check (Task C2)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _write_meta(dir_path: Path, sha: str = "abc", version: str | None = None):
    """Write a meta.json with a valid version by default (so version-check passes)."""
    from lora_estimator import ESTIMATOR_INDEX_VERSION
    meta = {"hf_commit_sha": sha,
            "estimator_index_version": version or ESTIMATOR_INDEX_VERSION}
    (dir_path / "meta.json").write_text(json.dumps(meta))


class TestFreshness(unittest.TestCase):
    def test_stale_sha_triggers_rebuild(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            _write_meta(Path(tmp), sha="abc")
            with mock.patch("lora_estimator._fetch_hf_head_sha", return_value="xyz"):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_called_once()

    def test_fresh_sha_no_rebuild(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            _write_meta(Path(tmp), sha="abc")
            with mock.patch("lora_estimator._fetch_hf_head_sha", return_value="abc"):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_not_called()

    def test_force_mode_always_rebuilds(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            _write_meta(Path(tmp))
            rebuild_fn = mock.Mock()
            ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="force")
            rebuild_fn.assert_called_once()

    def test_skip_mode_never_rebuilds(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            _write_meta(Path(tmp))
            with mock.patch("lora_estimator._fetch_hf_head_sha", return_value="xyz"):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="skip")
                rebuild_fn.assert_not_called()

    def test_no_internet_falls_back_to_cache(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            _write_meta(Path(tmp))
            with mock.patch("lora_estimator._fetch_hf_head_sha",
                            side_effect=ConnectionError("no network")):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_not_called()

    def test_missing_meta_triggers_rebuild(self):
        """No meta.json on disk → must rebuild regardless of network state."""
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            rebuild_fn = mock.Mock()
            ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
            rebuild_fn.assert_called_once()

    def test_stale_version_triggers_rebuild_without_network(self):
        """Index-version mismatch rebuilds unconditionally (schema change must invalidate pickle)."""
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            _write_meta(Path(tmp), sha="abc", version="0.0.1-old")
            # Network should not even be consulted when the schema is stale.
            with mock.patch("lora_estimator._fetch_hf_head_sha") as m_sha:
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_called_once()
                m_sha.assert_not_called()

    def test_corrupt_meta_triggers_rebuild(self):
        """Malformed meta.json should rebuild rather than crash."""
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "meta.json").write_text("{not valid json")
            rebuild_fn = mock.Mock()
            ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
            rebuild_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
