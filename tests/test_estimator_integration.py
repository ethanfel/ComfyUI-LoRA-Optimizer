"""End-to-end integration test for the estimator pipeline (Task F2).

Exercises the full cross-module flow:
  scripts/build_estimator_index → lora_estimator.LoRAEstimator →
  lora_optimizer.LoRAEstimatorNode.estimate

Phase 1 analysis is mocked out (requires real model + LoRA weights), but
every other stage runs against real JSON files on disk and the real
pickle/meta sidecar the builder produces.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Same import pattern as test_estimator_node.py — reuse the already-loaded
# lora_optimizer_under_test module so mock.patch("lora_optimizer…") in the
# main suite still targets the right instance.
from tests.test_lora_optimizer import lora_optimizer


def _write_cache_json(path: Path, per_prefix: dict):
    path.write_text(json.dumps({
        "algo_version": "test",
        "per_prefix": per_prefix,
    }))


def _fake_pair_per_prefix(values: float, prefixes=("layer.0.attn", "layer.1.attn")):
    """Build a realistic per_prefix dict for a pair cache."""
    from lora_estimator import EstimatorFeatureExtractor
    return {
        pfx: {s: values for s in EstimatorFeatureExtractor.PAIR_STATS}
        for pfx in prefixes
    }


def _fake_lora_per_prefix(prefixes=("layer.0.attn", "layer.1.attn")):
    from lora_estimator import EstimatorFeatureExtractor
    return {
        pfx: {s: 0.1 for s in EstimatorFeatureExtractor.LORA_STATS}
        for pfx in prefixes
    }


def _write_config_json(path: Path, sorted_hashes, candidates, score=0.8,
                       arch_preset="dit", base_model_family="zimage"):
    path.write_text(json.dumps({
        "algo_version": "test",
        "arch_preset": arch_preset,
        "base_model_family": base_model_family,
        "lora_content_hashes": sorted_hashes,
        "score": score,
        "config": candidates[0]["config"],
        "candidates": candidates,
    }))


class TestEstimatorEndToEnd(unittest.TestCase):
    def test_pipeline_build_then_predict(self):
        """Build a real index from in-memory JSONs, have the node load and retrieve from it."""
        from scripts.build_estimator_index import build_index
        from lora_estimator import EstimatorFeatureExtractor

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds = tmp_path / "hf"
            (ds / "config").mkdir(parents=True)
            (ds / "pair").mkdir()
            (ds / "lora").mkdir()

            # Two zimage/pair configs with distinct pair values so they land at
            # different points in feature space.
            _write_pair_cache_ab = _write_cache_json
            _write_pair_cache_ab(ds / "pair" / "aaa_bbb.pair.json",
                                 _fake_pair_per_prefix(0.3))
            _write_pair_cache_ab(ds / "pair" / "aaa_ccc.pair.json",
                                 _fake_pair_per_prefix(0.8))
            for h in ("aaa", "bbb", "ccc"):
                _write_cache_json(ds / "lora" / f"{h}.lora.json",
                                  _fake_lora_per_prefix())

            cand_ab = [{
                "rank": 1,
                "config": {"merge_mode": "ties", "sparsification": "disabled",
                           "sparsification_density": 0.7},
                "score_heuristic": 0.8, "score_measured": 0.82, "score_final": 0.81,
                "per_prefix_decisions": {},
            }]
            cand_ac = [{
                "rank": 1,
                "config": {"merge_mode": "weighted_average", "sparsification": "disabled",
                           "sparsification_density": 0.5},
                "score_heuristic": 0.7, "score_measured": 0.72, "score_final": 0.71,
                "per_prefix_decisions": {},
            }]
            _write_config_json(ds / "config" / "aaa_bbb_dit.config.json",
                               ["aaa", "bbb"], cand_ab, score=0.81)
            _write_config_json(ds / "config" / "aaa_ccc_dit.config.json",
                               ["aaa", "ccc"], cand_ac, score=0.71)

            index_dir = tmp_path / "index"
            meta = build_index(local_dataset_dir=ds, out_dir=index_dir,
                               hf_commit_sha="testsha")
            self.assertEqual(meta["n_samples"], 2)

            # Drive the node — Phase 1 is mocked, everything else is real.
            node_cls = lora_optimizer.NODE_CLASS_MAPPINGS["LoRAEstimator"]
            node = node_cls()
            extractor = EstimatorFeatureExtractor()
            # Mock Phase 1 output skewed toward the (aaa, bbb) training point:
            # pair-stat magnitude ~0.3 matches that combo's signature.
            phase1 = {
                "pair_stats": {(0, 1): {"per_prefix": _fake_pair_per_prefix(0.3)}},
                "lora_stats": {
                    0: {"per_prefix": _fake_lora_per_prefix()},
                    1: {"per_prefix": _fake_lora_per_prefix()},
                },
                "combo_size": 2,
                "base_model_family": "zimage",
                "active_loras": [{"name": "a", "strength": 1.0},
                                 {"name": "b", "strength": 1.0}],
                "n_prefixes": 2,
            }

            with mock.patch.object(
                lora_optimizer.LoRAAutoTuner, "_run_phase1_for_estimator",
                return_value=phase1,
            ), mock.patch.object(node, "_resolve_index_dir", return_value=index_dir):
                tuner_data, report = node.estimate(
                    model=mock.Mock(),
                    lora_stack=[{"name": "a", "strength": 1.0},
                                {"name": "b", "strength": 1.0}],
                    k=2, rebuild_index="skip", top_n_output=2,
                )

        # tuner_data shape is valid and consumable by the optimizer's
        # settings_source=from_tuner_data path.
        self.assertIsNotNone(tuner_data)
        self.assertEqual(tuner_data["version"], 1)
        self.assertTrue(tuner_data["_estimator"])
        self.assertIn("top_n", tuner_data)
        self.assertGreaterEqual(len(tuner_data["top_n"]), 1)

        # Each emitted candidate carries the fields the optimizer indexes into.
        for cand in tuner_data["top_n"]:
            self.assertIn("config", cand)
            self.assertIn("merge_mode", cand["config"])
            self.assertIn("score_final", cand)
            self.assertIn("rank", cand)

        # Analysis summary carries the combo shape.
        summary = tuner_data["analysis_summary"]
        self.assertEqual(summary["combo_size"], 2)
        self.assertEqual(summary["base_model_family"], "zimage")
        self.assertEqual(summary["n_neighbors"], 2)

        # Report is human-readable.
        self.assertIn("[Estimator]", report)
        self.assertIn("Predicted top", report)

        # With phase1 values matching the (aaa, bbb) training point, that
        # neighbor should be closer than (aaa, ccc), so 'ties' wins over
        # 'weighted_average' in the aggregate.
        top_mode = tuner_data["top_n"][0]["config"]["merge_mode"]
        self.assertEqual(top_mode, "ties",
                         "closer neighbor's config should lead the aggregated ranking")

    def test_pipeline_gracefully_handles_missing_family(self):
        """Index has only zimage rows; query is flux — returns (None, explanation)."""
        from scripts.build_estimator_index import build_index

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds = tmp_path / "hf"
            (ds / "config").mkdir(parents=True)
            (ds / "pair").mkdir()
            (ds / "lora").mkdir()

            _write_cache_json(ds / "pair" / "aaa_bbb.pair.json",
                              _fake_pair_per_prefix(0.3))
            for h in ("aaa", "bbb"):
                _write_cache_json(ds / "lora" / f"{h}.lora.json",
                                  _fake_lora_per_prefix())
            cand = [{
                "rank": 1, "config": {"merge_mode": "ties", "sparsification": "disabled"},
                "score_heuristic": 0.8, "score_measured": 0.8, "score_final": 0.8,
            }]
            _write_config_json(ds / "config" / "aaa_bbb_dit.config.json",
                               ["aaa", "bbb"], cand)

            index_dir = tmp_path / "index"
            build_index(local_dataset_dir=ds, out_dir=index_dir, hf_commit_sha="sha")

            node_cls = lora_optimizer.NODE_CLASS_MAPPINGS["LoRAEstimator"]
            node = node_cls()
            phase1 = {
                "pair_stats": {(0, 1): {"per_prefix": _fake_pair_per_prefix(0.3)}},
                "lora_stats": {
                    0: {"per_prefix": _fake_lora_per_prefix()},
                    1: {"per_prefix": _fake_lora_per_prefix()},
                },
                "combo_size": 2,
                "base_model_family": "flux",   # mismatched
                "active_loras": [{"name": "a", "strength": 1.0},
                                 {"name": "b", "strength": 1.0}],
                "n_prefixes": 2,
            }
            with mock.patch.object(
                lora_optimizer.LoRAAutoTuner, "_run_phase1_for_estimator",
                return_value=phase1,
            ), mock.patch.object(node, "_resolve_index_dir", return_value=index_dir):
                tuner_data, report = node.estimate(
                    model=mock.Mock(),
                    lora_stack=[{"name": "a", "strength": 1.0},
                                {"name": "b", "strength": 1.0}],
                    k=3, rebuild_index="skip", top_n_output=1,
                )

        self.assertIsNone(tuner_data)
        self.assertIn("No neighbors", report)
        self.assertIn("flux", report)


if __name__ == "__main__":
    unittest.main()
