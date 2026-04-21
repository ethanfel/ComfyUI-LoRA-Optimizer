"""Tests for the LoRAEstimator ComfyUI node (Task F1)."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# Reuse the already-loaded optimizer module. Importing our own copy would
# create a second lora_optimizer entry in sys.modules and cause mock.patch("lora_optimizer…")
# in other test files to target the wrong instance.
from tests.test_lora_optimizer import lora_optimizer


def _build_tiny_index(out_dir: Path):
    """Write a minimal index.pkl + meta.json that LoRAEstimator.from_disk can load."""
    from lora_estimator import EstimatorFeatureExtractor, ESTIMATOR_INDEX_VERSION
    from sklearn.neighbors import NearestNeighbors

    extractor = EstimatorFeatureExtractor()
    # Two fake zimage/combo_size=2 rows with distinct feature vectors.
    X_raw = np.zeros((2, extractor.DIM), dtype=np.float32)
    X_raw[0, 0] = 1.0
    X_raw[1, 1] = 1.0
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std < 1e-8] = 1.0
    X_norm = (X_raw - mean) / std

    nn = NearestNeighbors(metric="cosine", n_neighbors=2)
    nn.fit(X_norm)

    cand = {
        "rank": 1,
        "config": {"merge_mode": "ties", "sparsification": "disabled"},
        "score_final": 0.82,
        "score_heuristic": 0.80,
        "score_measured": 0.84,
    }
    labels = [
        {"base_model_family": "zimage", "combo_size": 2, "candidates": [cand]},
        {"base_model_family": "zimage", "combo_size": 2, "candidates": [cand]},
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "index.pkl", "wb") as f:
        pickle.dump({
            "index": nn, "features": X_norm, "raw_features": X_raw,
            "labels": labels, "zscore": (mean.tolist(), std.tolist()),
        }, f)
    (out_dir / "meta.json").write_text(json.dumps({
        "hf_commit_sha": "testsha",
        "estimator_index_version": ESTIMATOR_INDEX_VERSION,
        "n_samples": 2, "feature_dim": extractor.DIM,
        "build_timestamp": "test",
    }))


class TestEstimatorNodeRegistration(unittest.TestCase):
    def test_node_registered(self):
        self.assertIn("LoRAEstimator", lora_optimizer.NODE_CLASS_MAPPINGS)

    def test_display_name_present(self):
        self.assertIn("LoRAEstimator", lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS)

    def test_input_types(self):
        node_cls = lora_optimizer.NODE_CLASS_MAPPINGS["LoRAEstimator"]
        inputs = node_cls.INPUT_TYPES()
        required = inputs["required"]
        for field in ("model", "lora_stack", "k", "rebuild_index", "top_n_output"):
            self.assertIn(field, required)

    def test_return_types(self):
        node_cls = lora_optimizer.NODE_CLASS_MAPPINGS["LoRAEstimator"]
        self.assertEqual(node_cls.RETURN_TYPES, ("TUNER_DATA", "STRING"))
        self.assertEqual(node_cls.RETURN_NAMES, ("tuner_data", "estimator_report"))


class TestEstimatorNodeEstimate(unittest.TestCase):
    def test_estimate_produces_tuner_data(self):
        """End-to-end: mock Phase 1, load a tiny index, confirm tuner_data shape."""
        node_cls = lora_optimizer.NODE_CLASS_MAPPINGS["LoRAEstimator"]
        node = node_cls()

        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()

        phase1 = {
            "pair_stats": {(0, 1): {"per_prefix": {
                "layer.0.attn": {s: 0.3 for s in extractor.PAIR_STATS}
            }}},
            "lora_stats": {
                0: {"per_prefix": {"layer.0.attn": {s: 0.1 for s in extractor.LORA_STATS}}},
                1: {"per_prefix": {"layer.0.attn": {s: 0.1 for s in extractor.LORA_STATS}}},
            },
            "combo_size": 2,
            "base_model_family": "zimage",
            "active_loras": [{"name": "a", "strength": 1.0}, {"name": "b", "strength": 1.0}],
            "n_prefixes": 1,
        }

        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "estimator"
            _build_tiny_index(index_dir)

            with mock.patch.object(
                lora_optimizer.LoRAAutoTuner, "_run_phase1_for_estimator",
                return_value=phase1,
            ), mock.patch.object(node, "_resolve_index_dir", return_value=index_dir):
                tuner_data, report = node.estimate(
                    model=mock.Mock(),
                    lora_stack=[{"name": "a", "strength": 1.0},
                                {"name": "b", "strength": 1.0}],
                    k=2, rebuild_index="skip", top_n_output=1,
                )

        self.assertIsNotNone(tuner_data)
        self.assertIn("top_n", tuner_data)
        self.assertGreaterEqual(len(tuner_data["top_n"]), 1)
        self.assertTrue(tuner_data.get("_estimator"))
        self.assertIn("analysis_summary", tuner_data)
        self.assertIn("source_loras", tuner_data)
        self.assertIn("[Estimator]", report)

    def test_estimate_empty_stack_returns_none(self):
        node_cls = lora_optimizer.NODE_CLASS_MAPPINGS["LoRAEstimator"]
        node = node_cls()
        with mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_run_phase1_for_estimator",
            return_value=None,
        ):
            tuner_data, report = node.estimate(
                model=mock.Mock(), lora_stack=[],
                k=3, rebuild_index="skip", top_n_output=1,
            )
        self.assertIsNone(tuner_data)
        self.assertIn("No active LoRAs", report)

    def test_estimate_no_matching_neighbors(self):
        """If the index has no rows matching family+size, return (None, explanatory report)."""
        node_cls = lora_optimizer.NODE_CLASS_MAPPINGS["LoRAEstimator"]
        node = node_cls()
        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()
        # Phase1 reports 'flux' but the tiny index only has 'zimage' rows.
        phase1 = {
            "pair_stats": {(0, 1): {"per_prefix": {}}},
            "lora_stats": {0: {"per_prefix": {}}, 1: {"per_prefix": {}}},
            "combo_size": 2, "base_model_family": "flux",
            "active_loras": [{"name": "a", "strength": 1.0}, {"name": "b", "strength": 1.0}],
            "n_prefixes": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "estimator"
            _build_tiny_index(index_dir)
            with mock.patch.object(
                lora_optimizer.LoRAAutoTuner, "_run_phase1_for_estimator",
                return_value=phase1,
            ), mock.patch.object(node, "_resolve_index_dir", return_value=index_dir):
                tuner_data, report = node.estimate(
                    model=mock.Mock(),
                    lora_stack=[{"name": "a", "strength": 1.0},
                                {"name": "b", "strength": 1.0}],
                    k=2, rebuild_index="skip", top_n_output=1,
                )
        self.assertIsNone(tuner_data)
        self.assertIn("No neighbors", report)


class TestRunPhase1ForEstimator(unittest.TestCase):
    """Guard the internal Phase-1 wrapper against silently losing fresh per-prefix stats."""

    def test_tracks_new_entries(self):
        """_run_phase1_for_estimator must ask _run_group_analysis to surface new entries.

        Without track_new_entries=True, prefixes not already in the community cache
        would drop out of pair_stats / lora_stats, giving the feature extractor an
        all-zero vector for first-time combos.
        """
        tuner = lora_optimizer.LoRAAutoTuner()
        # Minimal normalized stack so _normalize_stack returns something active.
        with mock.patch.object(tuner, "_normalize_stack",
                               return_value=[{"name": "a", "strength": 1.0, "lora": {}},
                                             {"name": "b", "strength": 1.0, "lora": {}}]), \
             mock.patch.object(tuner, "_get_model_keys", return_value={"k": None}), \
             mock.patch.object(tuner, "_collect_lora_prefixes", return_value=[]), \
             mock.patch.object(tuner, "_build_target_groups", return_value={"prefix.0": object()}), \
             mock.patch.object(tuner, "_lora_identity_hash", side_effect=lambda l: "h"), \
             mock.patch.object(tuner, "_lora_cache_load", return_value=None), \
             mock.patch.object(tuner, "_pair_cache_load", return_value=None), \
             mock.patch.object(tuner, "_get_compute_device", return_value=mock.Mock(type="cpu")), \
             mock.patch.object(tuner, "_run_group_analysis",
                               return_value={"new_lora_entries": {}, "new_pair_entries": {},
                                             "prefix_count": 1}) as run_analysis:
            tuner._run_phase1_for_estimator(model=mock.Mock(), clip=None,
                                             lora_stack=[{"name": "a", "strength": 1.0},
                                                         {"name": "b", "strength": 1.0}])
        self.assertTrue(run_analysis.call_args.kwargs.get("track_new_entries"),
                        "must pass track_new_entries=True so fresh prefixes surface")


if __name__ == "__main__":
    unittest.main()
