"""Tests for the estimator index builder script (Task C1)."""
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

# Make scripts/ importable for the test.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _write_lora_cache(path: Path, per_prefix: dict):
    path.write_text(json.dumps({
        "algo_version": "test",
        "per_prefix": per_prefix,
    }))


def _write_pair_cache(path: Path, per_prefix: dict):
    path.write_text(json.dumps({
        "algo_version": "test",
        "per_prefix": per_prefix,
    }))


def _write_config(path: Path, sorted_hashes, candidates, score=0.8,
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


class TestIndexBuilder(unittest.TestCase):
    def test_build_from_fake_dataset(self):
        """Build index from an in-memory mini dataset, verify output files and content."""
        from scripts.build_estimator_index import build_index
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds = tmp_path / "hf"
            (ds / "config").mkdir(parents=True)
            (ds / "pair").mkdir()
            (ds / "lora").mkdir()

            # Two combos: (aaa, bbb) and (aaa, ccc). Both zimage / pair.
            pair_stats_ab = {"layer.0.attn": {s: 0.3 for s in [
                "overlap", "conflict", "dot", "norm_a_sq", "norm_b_sq",
                "weighted_total", "weighted_conflict", "expected_conflict",
                "excess_conflict", "subspace_overlap", "subspace_weight",
            ]}}
            pair_stats_ac = {"layer.0.attn": {s: 0.7 for s in [
                "overlap", "conflict", "dot", "norm_a_sq", "norm_b_sq",
                "weighted_total", "weighted_conflict", "expected_conflict",
                "excess_conflict", "subspace_overlap", "subspace_weight",
            ]}}
            lora_stats = {"layer.0.attn": {"norm_sq": 1.0, "rank": 16, "strength_sign": 1.0}}

            _write_pair_cache(ds / "pair" / "aaa_bbb.pair.json", pair_stats_ab)
            _write_pair_cache(ds / "pair" / "aaa_ccc.pair.json", pair_stats_ac)
            _write_lora_cache(ds / "lora" / "aaa.lora.json", lora_stats)
            _write_lora_cache(ds / "lora" / "bbb.lora.json", lora_stats)
            _write_lora_cache(ds / "lora" / "ccc.lora.json", lora_stats)

            cfg_cands = [{
                "rank": 1, "config": {"merge_mode": "ties", "sparsification": "disabled"},
                "score_heuristic": 0.8, "score_measured": 0.82, "score_final": 0.81,
                "per_prefix_decisions": {"layer.0.attn": "ties"},
            }]
            _write_config(ds / "config" / "aaa_bbb_dit.config.json",
                          ["aaa", "bbb"], cfg_cands, score=0.81)
            _write_config(ds / "config" / "aaa_ccc_dit.config.json",
                          ["aaa", "ccc"], cfg_cands, score=0.77)

            out = tmp_path / "out"
            meta = build_index(local_dataset_dir=ds, out_dir=out, hf_commit_sha="testsha")

            self.assertTrue((out / "index.pkl").exists())
            self.assertTrue((out / "meta.json").exists())

            on_disk_meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(on_disk_meta["hf_commit_sha"], "testsha")
            self.assertEqual(on_disk_meta["n_samples"], 2)
            self.assertIn("estimator_index_version", on_disk_meta)
            self.assertIn("feature_dim", on_disk_meta)
            self.assertIn("build_timestamp", on_disk_meta)

            data = pickle.loads((out / "index.pkl").read_bytes())
            self.assertIn("index", data)
            self.assertIn("features", data)
            self.assertIn("raw_features", data)
            self.assertIn("zscore", data)
            self.assertEqual(len(data["labels"]), 2)
            for lbl in data["labels"]:
                self.assertEqual(lbl["base_model_family"], "zimage")
                self.assertEqual(lbl["combo_size"], 2)
                self.assertIn("candidates", lbl)

            # z-score: mean/std lists of length DIM
            mean, std = data["zscore"]
            self.assertEqual(len(mean), on_disk_meta["feature_dim"])
            self.assertEqual(len(std), on_disk_meta["feature_dim"])

    def test_parse_config_name(self):
        """parse_config_name extracts sorted hashes + arch from the HF filename convention."""
        from scripts.build_estimator_index import parse_config_name

        hashes, arch = parse_config_name("aaa_bbb_dit.config.json")
        self.assertEqual(hashes, ["aaa", "bbb"])
        self.assertEqual(arch, "dit")

        hashes, arch = parse_config_name("aaa_bbb_ccc_dit.config.json")
        self.assertEqual(hashes, ["aaa", "bbb", "ccc"])
        self.assertEqual(arch, "dit")

        hashes, arch = parse_config_name("aaa_bbb_sd_unet.config.json")
        # sd_unet is a compound arch name — two trailing tokens.
        self.assertEqual(hashes, ["aaa", "bbb"])
        self.assertEqual(arch, "sd_unet")


if __name__ == "__main__":
    unittest.main()
