"""Tests for EstimatorFeatureExtractor (Task B2)."""
import unittest

import numpy as np


class TestFeatureExtractor(unittest.TestCase):
    def test_extracts_fixed_dim_vector(self):
        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()
        phase1 = {
            "pair_stats": {
                (0, 1): {"per_prefix": {
                    "layer.0.attn": {s: 0.5 for s in extractor.PAIR_STATS},
                    "layer.1.attn": {s: 0.6 for s in extractor.PAIR_STATS},
                }},
            },
            "lora_stats": {
                0: {"per_prefix": {"layer.0.attn": {s: 0.1 for s in extractor.LORA_STATS}}},
                1: {"per_prefix": {"layer.0.attn": {s: 0.2 for s in extractor.LORA_STATS}}},
            },
            "combo_size": 2,
            "base_model_family": "zimage",
        }
        vec = extractor.featurize(phase1)
        self.assertEqual(vec.shape, (extractor.DIM,))
        self.assertEqual(vec.dtype, np.float32)

    def test_worst_pair_captured(self):
        """Worst-pair feature = stats of the pair with max mean excess_conflict."""
        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()
        phase1 = {
            "pair_stats": {
                (0, 1): {"per_prefix": {
                    "k": {**{s: 0.0 for s in extractor.PAIR_STATS}, "excess_conflict": 0.1}
                }},
                (0, 2): {"per_prefix": {
                    "k": {**{s: 0.0 for s in extractor.PAIR_STATS}, "excess_conflict": 0.9}
                }},
            },
            "lora_stats": {i: {"per_prefix": {}} for i in range(3)},
            "combo_size": 3,
            "base_model_family": "zimage",
        }
        vec = extractor.featurize(phase1)
        worst_slice = extractor.worst_pair_slice(vec)
        ec_idx = extractor.PAIR_STATS.index("excess_conflict")
        self.assertAlmostEqual(float(worst_slice[ec_idx]), 0.9, places=3)

    def test_family_one_hot(self):
        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()
        phase1 = {"pair_stats": {}, "lora_stats": {}, "combo_size": 2,
                  "base_model_family": "wan"}
        vec = extractor.featurize(phase1)
        fam_slice = extractor.family_slice(vec)
        expected = np.zeros(len(extractor.FAMILY_LABELS), dtype=np.float32)
        expected[extractor.FAMILY_LABELS.index("wan")] = 1.0
        np.testing.assert_array_equal(fam_slice, expected)

    def test_unknown_family_maps_to_unknown_bucket(self):
        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()
        phase1 = {"pair_stats": {}, "lora_stats": {}, "combo_size": 2,
                  "base_model_family": "nonexistent_model_family"}
        vec = extractor.featurize(phase1)
        fam_slice = extractor.family_slice(vec)
        self.assertEqual(float(fam_slice[extractor.FAMILY_LABELS.index("unknown")]), 1.0)

    def test_size_onehot(self):
        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()
        # combo_size=4 maps to index 2 (0:pair, 1:triple, 2:quad, 3:5+)
        phase1 = {"pair_stats": {}, "lora_stats": {}, "combo_size": 4,
                  "base_model_family": "zimage"}
        vec = extractor.featurize(phase1)
        size_slice = extractor.size_slice(vec)
        self.assertEqual(float(size_slice[2]), 1.0)
        self.assertEqual(float(size_slice.sum()), 1.0)

    def test_deterministic(self):
        """Same input → identical vector."""
        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()
        phase1 = {
            "pair_stats": {
                (0, 1): {"per_prefix": {
                    "k": {s: 0.3 for s in extractor.PAIR_STATS}
                }}
            },
            "lora_stats": {0: {"per_prefix": {"k": {s: 0.1 for s in extractor.LORA_STATS}}}},
            "combo_size": 2,
            "base_model_family": "zimage",
        }
        v1 = extractor.featurize(phase1)
        v2 = extractor.featurize(phase1)
        np.testing.assert_array_equal(v1, v2)


if __name__ == "__main__":
    unittest.main()
