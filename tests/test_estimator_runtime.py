"""Tests for the estimator runtime (Task D1)."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _mk_labels(families, sizes, candidates_per_row):
    return [
        {"base_model_family": f, "combo_size": s, "candidates": c}
        for f, s, c in zip(families, sizes, candidates_per_row)
    ]


class TestLoRAEstimatorRetrieval(unittest.TestCase):
    def test_retrieve_filters_by_family_and_size(self):
        from lora_estimator import LoRAEstimator
        rng = np.random.default_rng(0)
        features = rng.random((10, 80)).astype(np.float32)
        # First 5 zimage/3, next 3 wan/3, last 2 zimage/4.
        families = ["zimage"] * 5 + ["wan"] * 3 + ["zimage"] * 2
        sizes = [3] * 5 + [3] * 3 + [4] * 2
        labels = _mk_labels(families, sizes, [[] for _ in range(10)])
        est = LoRAEstimator.from_memory(
            features=features, labels=labels,
            zscore=(np.zeros(80), np.ones(80)),
        )
        query = rng.random(80).astype(np.float32)
        neighbors = est.retrieve(query, base_model_family="zimage", combo_size=3, k=3)
        self.assertEqual(len(neighbors), 3)
        for n in neighbors:
            self.assertEqual(n["label"]["base_model_family"], "zimage")
            self.assertEqual(n["label"]["combo_size"], 3)

    def test_retrieve_returns_empty_when_no_match(self):
        from lora_estimator import LoRAEstimator
        est = LoRAEstimator.from_memory(
            features=np.zeros((3, 80), dtype=np.float32),
            labels=_mk_labels(["zimage"] * 3, [2] * 3, [[]] * 3),
            zscore=(np.zeros(80), np.ones(80)),
        )
        neighbors = est.retrieve(np.zeros(80, dtype=np.float32),
                                 base_model_family="flux", combo_size=2, k=5)
        self.assertEqual(neighbors, [])

    def test_retrieve_k_larger_than_matches(self):
        """k bigger than the filtered subset returns all matches, not an error."""
        from lora_estimator import LoRAEstimator
        est = LoRAEstimator.from_memory(
            features=np.zeros((4, 80), dtype=np.float32),
            labels=_mk_labels(["zimage"] * 2 + ["wan"] * 2, [2] * 4, [[]] * 4),
            zscore=(np.zeros(80), np.ones(80)),
        )
        neighbors = est.retrieve(np.zeros(80, dtype=np.float32),
                                 base_model_family="zimage", combo_size=2, k=10)
        self.assertEqual(len(neighbors), 2)


class TestLoRAEstimatorAggregate(unittest.TestCase):
    def test_aggregate_weights_by_inverse_distance(self):
        """Higher-scoring config across weighted neighbors wins; score is an average not a sum."""
        from lora_estimator import LoRAEstimator
        neighbors = [
            {"distance": 0.1, "label": {"candidates": [
                {"rank": 1, "config": {"merge_mode": "ties", "sparsification": "disabled"},
                 "score_final": 0.80, "score_heuristic": 0.78, "score_measured": 0.82},
                {"rank": 2, "config": {"merge_mode": "weighted_average", "sparsification": "disabled"},
                 "score_final": 0.70, "score_heuristic": 0.70, "score_measured": 0.70},
            ]}},
            {"distance": 0.2, "label": {"candidates": [
                {"rank": 1, "config": {"merge_mode": "ties", "sparsification": "disabled"},
                 "score_final": 0.85, "score_heuristic": 0.84, "score_measured": 0.86},
            ]}},
        ]
        top_n = LoRAEstimator.aggregate_candidates(neighbors, top_n=2)
        self.assertEqual(len(top_n), 2)
        self.assertEqual(top_n[0]["config"]["merge_mode"], "ties")
        # Weighted average: w1=1/1.1≈0.909, w2=1/1.2≈0.833.
        # ties: (0.909*0.80 + 0.833*0.85) / (0.909+0.833) ≈ 0.824
        self.assertAlmostEqual(top_n[0]["score_final"], 0.8239, places=2)
        self.assertEqual(top_n[0]["rank"], 1)
        self.assertEqual(top_n[1]["rank"], 2)
        # Score between 0 and 1 (weighted avg, not sum).
        self.assertLessEqual(top_n[0]["score_final"], 1.0)

    def test_aggregate_empty_neighbors_returns_empty(self):
        from lora_estimator import LoRAEstimator
        self.assertEqual(LoRAEstimator.aggregate_candidates([], top_n=3), [])

    def test_aggregate_skips_candidates_without_config(self):
        from lora_estimator import LoRAEstimator
        neighbors = [
            {"distance": 0.0, "label": {"candidates": [
                {"rank": 1, "score_final": 0.5},  # no config — ignored
                {"rank": 2, "config": {"merge_mode": "ties"}, "score_final": 0.6},
            ]}},
        ]
        top_n = LoRAEstimator.aggregate_candidates(neighbors, top_n=5)
        self.assertEqual(len(top_n), 1)
        self.assertEqual(top_n[0]["config"]["merge_mode"], "ties")


class TestLoRAEstimatorEmit(unittest.TestCase):
    def test_emit_tuner_data_shape(self):
        from lora_estimator import LoRAEstimator
        top_n = [{
            "rank": 1,
            "config": {"merge_mode": "ties", "sparsification": "disabled"},
            "score_final": 0.8, "score_heuristic": 0.75, "score_measured": 0.82,
            "score_external": None, "metrics": {}, "external_details": None,
        }]
        td = LoRAEstimator.emit_tuner_data(
            top_n=top_n,
            source_loras=[{"name": "a.safetensors", "strength": 1.0}],
            analysis_summary={"n_prefixes": 240, "confidence": 0.91},
        )
        self.assertIn("top_n", td)
        self.assertEqual(len(td["top_n"]), 1)
        self.assertIn("config", td["top_n"][0])
        self.assertIn("analysis_summary", td)
        self.assertIn("source_loras", td)
        self.assertEqual(td["version"], 1)
        self.assertTrue(td["_estimator"])


class TestLoRAEstimatorFromDisk(unittest.TestCase):
    def test_from_disk_roundtrip(self):
        """Build a pickle matching the builder's schema, load via from_disk, retrieve."""
        from lora_estimator import LoRAEstimator
        from sklearn.neighbors import NearestNeighbors
        features = np.zeros((3, 80), dtype=np.float32)
        features[0, 0] = 1.0
        features[1, 1] = 1.0
        features[2, 2] = 1.0
        mean = np.zeros(80, dtype=np.float32)
        std = np.ones(80, dtype=np.float32)
        nn = NearestNeighbors(metric="cosine", n_neighbors=3)
        nn.fit(features)
        labels = _mk_labels(["zimage"] * 3, [2] * 3, [[]] * 3)

        with tempfile.TemporaryDirectory() as tmp:
            idx_path = Path(tmp) / "index.pkl"
            with open(idx_path, "wb") as f:
                pickle.dump({
                    "index": nn, "features": features, "raw_features": features,
                    "labels": labels, "zscore": (mean.tolist(), std.tolist()),
                }, f)
            est = LoRAEstimator.from_disk(idx_path)
            q = np.zeros(80, dtype=np.float32)
            q[0] = 1.0
            neighbors = est.retrieve(q, base_model_family="zimage", combo_size=2, k=1)
            self.assertEqual(len(neighbors), 1)
            # Exact match — distance should be ~0.
            self.assertLess(neighbors[0]["distance"], 1e-4)


if __name__ == "__main__":
    unittest.main()
