"""Tests for the estimator freshness check (Task C2)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestFreshness(unittest.TestCase):
    def test_stale_sha_triggers_rebuild(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "meta.json"
            meta_path.write_text(json.dumps({"hf_commit_sha": "abc"}))
            with mock.patch("lora_estimator._fetch_hf_head_sha", return_value="xyz"):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_called_once()

    def test_fresh_sha_no_rebuild(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "meta.json").write_text(json.dumps({"hf_commit_sha": "abc"}))
            with mock.patch("lora_estimator._fetch_hf_head_sha", return_value="abc"):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_not_called()

    def test_force_mode_always_rebuilds(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "meta.json").write_text(json.dumps({"hf_commit_sha": "abc"}))
            rebuild_fn = mock.Mock()
            ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="force")
            rebuild_fn.assert_called_once()

    def test_skip_mode_never_rebuilds(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "meta.json").write_text(json.dumps({"hf_commit_sha": "abc"}))
            with mock.patch("lora_estimator._fetch_hf_head_sha", return_value="xyz"):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="skip")
                rebuild_fn.assert_not_called()

    def test_no_internet_falls_back_to_cache(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "meta.json").write_text(json.dumps({"hf_commit_sha": "abc"}))
            with mock.patch("lora_estimator._fetch_hf_head_sha",
                            side_effect=ConnectionError("no network")):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_not_called()

    def test_missing_meta_triggers_rebuild(self):
        """No meta.json on disk → must rebuild regardless of network state."""
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            # No meta.json written.
            rebuild_fn = mock.Mock()
            ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
            rebuild_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
