# LoRA Merge Estimator Pre-Node Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a ComfyUI pre-node that predicts optimal merge configs from the HF community dataset via k-NN retrieval over Phase 1 stats, emitting a `tuner_data` struct the existing optimizer replays — skipping Phase 2 grid sweep.

**Architecture:** Case-based retrieval on ~120-dim combo feature vectors (aggregated per-prefix stats + worst-pair summary + size/family one-hots). Cosine k-NN with hard filter on `base_model_family` and `combo_size`. Top-k candidates aggregated by inverse-distance weighting into a `tuner_data.top_n` struct.

**Tech Stack:** Python, scikit-learn (`NearestNeighbors`), numpy, huggingface_hub, existing ComfyUI plugin patterns (`NODE_CLASS_MAPPINGS`, `INPUT_TYPES`, `RETURN_TYPES`).

**Design doc:** [`docs/plans/2026-04-21-estimator-prenode-design.md`](./2026-04-21-estimator-prenode-design.md)

**Testing stance:** TDD throughout. Each task starts with a failing test. Commit after each task passes. Use the stub pattern already in `tests/test_lora_optimizer.py:_install_stubs()` for any ComfyUI-dependent tests.

---

## Phase A — Per-prefix instrumentation (Track 1, ships first)

### Task A1: Capture per-prefix merge decisions in AutoTuner

**Rationale:** The HF payload currently stores only the global config. We need to record which merge strategy was actually applied per prefix so the estimator can predict per-prefix policies. This instrumentation must land **before** the user reruns the 400 combos.

**Files:**
- Modify: `lora_optimizer.py:6727` (inside `_merge_one_group` — the final `pf_mode` value is the decision)
- Modify: `lora_optimizer.py:~6145` (the `optimize_merge` method — returns the collected decisions)
- Test: `tests/test_per_prefix_capture.py` (new)

**Step 1: Write failing test**

```python
# tests/test_per_prefix_capture.py
import unittest
from unittest import mock
# Reuse stub setup from test_lora_optimizer.py
from tests.test_lora_optimizer import _install_stubs, _load_optimizer_module

class TestPerPrefixCapture(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.mod = _load_optimizer_module()

    def test_merge_returns_per_prefix_decisions(self):
        """optimize_merge must return a dict mapping prefix -> chosen strategy"""
        opt = self.mod.LoRAOptimizer()
        # Run a mock merge that goes through _merge_one_group twice
        # ... (mock setup) ...
        _, _, _, _, lora_data = opt.optimize_merge(
            model=mock.Mock(), lora_stack=[...], output_strength=1.0,
            optimization_mode="per_prefix",
        )
        self.assertIn("per_prefix_decisions", lora_data)
        decisions = lora_data["per_prefix_decisions"]
        self.assertIsInstance(decisions, dict)
        # Each value is a strategy name (ties / weighted_average / slerp / etc.)
        for prefix, strat in decisions.items():
            self.assertIsInstance(prefix, str)
            self.assertIn(strat, {"ties", "weighted_average", "weighted_sum",
                                   "consensus", "slerp"})
```

**Step 2: Run test, verify it fails**

Run: `pytest tests/test_per_prefix_capture.py::TestPerPrefixCapture::test_merge_returns_per_prefix_decisions -v`
Expected: FAIL with `KeyError: 'per_prefix_decisions'`

**Step 3: Implement capture**

In `_merge_one_group` at `lora_optimizer.py:6727`, after the `pf_mode` is finalized (~line 6795), append the decision to a list that's passed through the closure:

```python
# Inside optimize_merge, before _merge_one_group definition:
per_prefix_decisions = {}

def _merge_one_group(label_prefix, target_group):
    nonlocal gpu_patch_bytes
    # ... existing logic ...
    # AFTER all overrides (line ~6795 where pf_mode is finalized):
    per_prefix_decisions[label_prefix] = pf_mode
    # ... rest of merge ...
```

Then at `optimize_merge` return site, include in `lora_data`:

```python
lora_data["per_prefix_decisions"] = per_prefix_decisions
```

**Step 4: Run test, verify it passes**

Run: `pytest tests/test_per_prefix_capture.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_per_prefix_capture.py
git commit -m "feat: capture per-prefix merge decisions in lora_data

Each prefix's final chosen strategy (after auto-select + overrides +
slerp upgrades + full-rank gates) is recorded in
lora_data['per_prefix_decisions']. Enables per-prefix label collection
for the estimator training set."
```

---

### Task A2: Include per-prefix decisions in HF upload payload

**Files:**
- Modify: `lora_optimizer.py:8394-8416` (`_community_upload_results`, the candidates list construction)
- Modify: `lora_optimizer.py:~9640` (build `top_n` entries — attach decisions to each candidate)
- Test: `tests/test_per_prefix_capture.py` — add test for payload shape

**Step 1: Write failing test**

```python
def test_upload_payload_includes_per_prefix_decisions(self):
    """Candidates in the uploaded config payload must include per_prefix_decisions"""
    # Mock the HfApi.upload_file call and capture the payload
    with mock.patch('huggingface_hub.HfApi.upload_file') as mock_upload:
        # ... run AutoTuner that triggers upload ...
        call_args = mock_upload.call_args_list
        config_upload = next(c for c in call_args
                             if 'config/' in c.kwargs.get('path_in_repo', ''))
        payload_bytes = config_upload.kwargs['path_or_fileobj'].read()
        payload = json.loads(payload_bytes)
        for cand in payload["candidates"]:
            self.assertIn("per_prefix_decisions", cand)
            self.assertIsInstance(cand["per_prefix_decisions"], dict)
```

**Step 2: Run test, verify it fails**

Run: `pytest tests/test_per_prefix_capture.py::TestPerPrefixCapture::test_upload_payload_includes_per_prefix_decisions -v`
Expected: FAIL

**Step 3: Thread decisions through the AutoTuner pipeline**

In the AutoTuner's grid evaluation loop (around line 9640 where `top_n` entries are built), add the decisions captured during each candidate's merge:

```python
# When building each result entry in the sweep:
results.append({
    "rank": rank,
    "score_heuristic": ...,
    "score_measured": ...,
    "score_final": ...,
    "config": config_dict,
    "metrics": metrics,
    "per_prefix_decisions": lora_data.get("per_prefix_decisions", {}),  # NEW
})
```

Then in `tuner_data["top_n"]` construction (~line 9640-9648):

```python
"top_n": [{
    "rank": r["rank"],
    "score_heuristic": r["score_heuristic"],
    # ... existing fields ...
    "per_prefix_decisions": r.get("per_prefix_decisions", {}),  # NEW
} for r in results],
```

And in `_community_upload_results` candidates list (~line 8394):

```python
candidates = [
    {
        "rank": entry.get("rank", idx + 1),
        "config": entry["config"],
        "score_heuristic": entry.get("score_heuristic", 0.0),
        "score_measured": entry.get("score_measured", 0.0),
        "score_final": entry.get("score_final", 0.0),
        "per_prefix_decisions": entry.get("per_prefix_decisions", {}),  # NEW
    }
    for idx, entry in enumerate(tuner_data["top_n"])
    if "config" in entry
]
```

**Step 4: Run test, verify it passes**

Run: `pytest tests/test_per_prefix_capture.py -v`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_per_prefix_capture.py
git commit -m "feat: include per_prefix_decisions in community cache upload payload

Each candidate in the uploaded config.json now carries the winning
per-prefix strategy map. This is the label data future estimator
versions need for per-prefix prediction."
```

---

### Task A3: User reruns the 400 combos (manual step)

**Not Claude's work — user action.**

After A1 and A2 land:
1. User reruns `LoRACombinationGenerator` over the 400 combos with the new instrumentation.
2. Uploads overwrite the old `config/*.config.json` entries (same hash-based keys).
3. The 2 legacy sd_unet configs are dropped (non-Zimage noise per memory).

No code action needed; this is an operational step. Phase B work can proceed in parallel — the estimator MVP doesn't need per-prefix data until Phase E.

---

## Phase B — Estimator foundation

### Task B1: Add sklearn dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Check current deps**

Run: `grep -A 20 'dependencies' pyproject.toml`

**Step 2: Add scikit-learn**

Add to the `dependencies` list:
```toml
"scikit-learn>=1.3",
```

**Step 3: Verify install**

Run: `pip install scikit-learn>=1.3 && python -c "from sklearn.neighbors import NearestNeighbors; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add scikit-learn dependency for estimator k-NN"
```

---

### Task B2: EstimatorFeatureExtractor — combo → feature vector

**Files:**
- Create: `lora_estimator.py` (new top-level module, keeps `lora_optimizer.py` from growing)
- Test: `tests/test_estimator_features.py` (new)

**Constants to define in `lora_estimator.py`:**
```python
ESTIMATOR_INDEX_VERSION = "1.0.0"
PAIR_STATS = ["overlap", "conflict", "dot", "norm_a_sq", "norm_b_sq",
              "weighted_total", "weighted_conflict", "expected_conflict",
              "excess_conflict", "subspace_overlap", "subspace_weight"]
LORA_STATS = ["norm_sq", "rank", "strength_sign"]  # numeric-only; skip magnitude_samples etc.
FAMILY_LABELS = ["zimage", "wan", "flux", "qwen_image", "acestep", "ltx",
                 "sdxl", "sd15", "unknown"]
AGG_FUNCS = ["mean", "p90", "max", "std"]  # 4 aggregates per stat
```

**Step 1: Write failing test**

```python
# tests/test_estimator_features.py
import unittest
import numpy as np

class TestFeatureExtractor(unittest.TestCase):
    def test_extracts_fixed_dim_vector(self):
        from lora_estimator import EstimatorFeatureExtractor
        extractor = EstimatorFeatureExtractor()
        # Mock Phase 1 output shape
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
        """Worst-pair feature = stats of the pair with max excess_conflict"""
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
        # Worst pair's excess_conflict mean should be ~0.9, not ~0.5
        ec_idx = extractor.PAIR_STATS.index("excess_conflict")
        self.assertAlmostEqual(worst_slice[ec_idx], 0.9, places=3)

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
```

**Step 2: Run tests, verify they fail**

Run: `pytest tests/test_estimator_features.py -v`
Expected: FAIL with `ModuleNotFoundError: lora_estimator`

**Step 3: Implement `EstimatorFeatureExtractor`**

```python
# lora_estimator.py
import numpy as np

ESTIMATOR_INDEX_VERSION = "1.0.0"

class EstimatorFeatureExtractor:
    PAIR_STATS = ["overlap", "conflict", "dot", "norm_a_sq", "norm_b_sq",
                  "weighted_total", "weighted_conflict", "expected_conflict",
                  "excess_conflict", "subspace_overlap", "subspace_weight"]
    LORA_STATS = ["norm_sq", "rank", "strength_sign"]
    FAMILY_LABELS = ["zimage", "wan", "flux", "qwen_image", "acestep", "ltx",
                     "sdxl", "sd15", "unknown"]
    AGG_FUNCS = ["mean", "p90", "max", "std"]
    COMBO_SIZE_BUCKETS = [2, 3, 4, 5]  # pair, triple, quad, 5+

    # Dim: (pair_stats * aggs) + (lora_stats * aggs) + (worst_pair * pair_stats) + size_onehot + family_onehot
    PAIR_DIM = len(PAIR_STATS) * len(AGG_FUNCS)       # 44
    LORA_DIM = len(LORA_STATS) * len(AGG_FUNCS)       # 12
    WORST_DIM = len(PAIR_STATS)                       # 11
    SIZE_DIM = len(COMBO_SIZE_BUCKETS)                # 4
    FAM_DIM = len(FAMILY_LABELS)                      # 9
    DIM = PAIR_DIM + LORA_DIM + WORST_DIM + SIZE_DIM + FAM_DIM  # 80 (under 120; remaining budget for expansion)

    def featurize(self, phase1):
        """phase1 = {'pair_stats': {(i,j): {'per_prefix': {...}}}, 'lora_stats': ..., 'combo_size': int, 'base_model_family': str}"""
        parts = []
        parts.append(self._aggregate_pairs(phase1["pair_stats"]))
        parts.append(self._aggregate_loras(phase1["lora_stats"]))
        parts.append(self._worst_pair(phase1["pair_stats"]))
        parts.append(self._size_onehot(phase1["combo_size"]))
        parts.append(self._family_onehot(phase1.get("base_model_family", "unknown")))
        vec = np.concatenate(parts).astype(np.float32)
        assert vec.shape == (self.DIM,), f"expected {self.DIM}, got {vec.shape}"
        return vec

    def _aggregate_pairs(self, pair_stats):
        """For each pair stat, compute mean/p90/max/std across all pairs × all prefixes."""
        all_values = {s: [] for s in self.PAIR_STATS}
        for _, pair_data in pair_stats.items():
            per_prefix = pair_data.get("per_prefix", {})
            for _, stats in per_prefix.items():
                for s in self.PAIR_STATS:
                    all_values[s].append(stats.get(s, 0.0))
        out = np.zeros(self.PAIR_DIM, dtype=np.float32)
        for i, s in enumerate(self.PAIR_STATS):
            vals = np.array(all_values[s], dtype=np.float32) if all_values[s] else np.zeros(1, dtype=np.float32)
            out[i*4 + 0] = vals.mean()
            out[i*4 + 1] = np.quantile(vals, 0.9)
            out[i*4 + 2] = vals.max()
            out[i*4 + 3] = vals.std()
        return out

    def _aggregate_loras(self, lora_stats):
        """Same aggregation for per-lora stats."""
        all_values = {s: [] for s in self.LORA_STATS}
        for _, lora_data in lora_stats.items():
            per_prefix = lora_data.get("per_prefix", {})
            for _, stats in per_prefix.items():
                for s in self.LORA_STATS:
                    v = stats.get(s)
                    if isinstance(v, (int, float)):
                        all_values[s].append(float(v))
        out = np.zeros(self.LORA_DIM, dtype=np.float32)
        for i, s in enumerate(self.LORA_STATS):
            vals = np.array(all_values[s], dtype=np.float32) if all_values[s] else np.zeros(1, dtype=np.float32)
            out[i*4 + 0] = vals.mean()
            out[i*4 + 1] = np.quantile(vals, 0.9)
            out[i*4 + 2] = vals.max()
            out[i*4 + 3] = vals.std()
        return out

    def _worst_pair(self, pair_stats):
        """Find pair with max mean excess_conflict; return its per-pair mean stats."""
        if not pair_stats:
            return np.zeros(self.WORST_DIM, dtype=np.float32)
        worst_key = None
        worst_ec = -np.inf
        for key, pair_data in pair_stats.items():
            per_prefix = pair_data.get("per_prefix", {})
            ecs = [stats.get("excess_conflict", 0.0) for _, stats in per_prefix.items()]
            mean_ec = np.mean(ecs) if ecs else 0.0
            if mean_ec > worst_ec:
                worst_ec = mean_ec
                worst_key = key
        if worst_key is None:
            return np.zeros(self.WORST_DIM, dtype=np.float32)
        per_prefix = pair_stats[worst_key].get("per_prefix", {})
        out = np.zeros(self.WORST_DIM, dtype=np.float32)
        for i, s in enumerate(self.PAIR_STATS):
            vals = [stats.get(s, 0.0) for _, stats in per_prefix.items()]
            out[i] = np.mean(vals) if vals else 0.0
        return out

    def _size_onehot(self, size):
        out = np.zeros(self.SIZE_DIM, dtype=np.float32)
        idx = min(max(0, size - 2), self.SIZE_DIM - 1)
        out[idx] = 1.0
        return out

    def _family_onehot(self, family):
        out = np.zeros(self.FAM_DIM, dtype=np.float32)
        if family in self.FAMILY_LABELS:
            out[self.FAMILY_LABELS.index(family)] = 1.0
        else:
            out[self.FAMILY_LABELS.index("unknown")] = 1.0
        return out

    def worst_pair_slice(self, vec):
        start = self.PAIR_DIM + self.LORA_DIM
        return vec[start:start + self.WORST_DIM]

    def family_slice(self, vec):
        start = self.PAIR_DIM + self.LORA_DIM + self.WORST_DIM + self.SIZE_DIM
        return vec[start:start + self.FAM_DIM]

    def size_slice(self, vec):
        start = self.PAIR_DIM + self.LORA_DIM + self.WORST_DIM
        return vec[start:start + self.SIZE_DIM]
```

**Step 4: Run tests, verify they pass**

Run: `pytest tests/test_estimator_features.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add lora_estimator.py tests/test_estimator_features.py
git commit -m "feat: add EstimatorFeatureExtractor for combo feature vectors

80-dim vector combining aggregated pair/lora stats, worst-pair summary,
combo-size one-hot, and family one-hot. Deterministic, dtype=float32,
serialisable via numpy."
```

---

## Phase C — Index build pipeline

### Task C1: Index builder script

**Files:**
- Create: `scripts/build_estimator_index.py`
- Test: `tests/test_index_builder.py`

**Script responsibilities:**
1. `snapshot_download` config/pair/lora from HF
2. For each config, locate matching pair.json + lora.json files, parse as Phase 1 stats
3. Call `EstimatorFeatureExtractor.featurize()` → feature matrix
4. Fit z-score params (mean/std per dim)
5. Fit `NearestNeighbors(metric="cosine", n_neighbors=10)` on z-scored matrix
6. Pickle: `{'index': nn_model, 'features': matrix, 'labels': [per-config candidate lists], 'zscore': (mean, std), 'meta': {...}}`
7. Write `meta.json` with `hf_commit_sha`, `ESTIMATOR_INDEX_VERSION`, `n_samples`, `feature_dim`, `build_timestamp`

**Step 1: Write failing test**

```python
# tests/test_index_builder.py
import unittest
import json
import pickle
from pathlib import Path
import tempfile

class TestIndexBuilder(unittest.TestCase):
    def test_build_from_fake_dataset(self):
        """Build index from an in-memory mini dataset, verify outputs."""
        from scripts.build_estimator_index import build_index
        with tempfile.TemporaryDirectory() as tmp:
            # Create a fake dataset dir matching HF layout
            ds = Path(tmp) / "hf"
            (ds / "config").mkdir(parents=True)
            (ds / "pair").mkdir()
            (ds / "lora").mkdir()
            # Write 2 minimal configs + referenced pair/lora caches
            # ... (fixture setup) ...
            out = Path(tmp) / "out"
            build_index(local_dataset_dir=ds, out_dir=out, hf_commit_sha="testsha")
            self.assertTrue((out / "index.pkl").exists())
            self.assertTrue((out / "meta.json").exists())
            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["hf_commit_sha"], "testsha")
            self.assertEqual(meta["n_samples"], 2)
            data = pickle.loads((out / "index.pkl").read_bytes())
            self.assertIn("index", data)
            self.assertIn("zscore", data)
            self.assertEqual(len(data["labels"]), 2)
```

**Step 2: Run, verify fail**

Run: `pytest tests/test_index_builder.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `scripts/build_estimator_index.py`**

```python
#!/usr/bin/env python3
"""Build estimator k-NN index from the HF community cache dataset."""
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
from lora_estimator import EstimatorFeatureExtractor, ESTIMATOR_INDEX_VERSION


def parse_config_name(fname):
    """Extract (sorted_hashes, arch_preset) from 'h1_h2_h3_dit.config.json'"""
    stem = fname.replace(".config.json", "")
    parts = stem.split("_")
    arch = parts[-1] if parts[-1] in ("dit", "sd_unet", "acestep_dit", "llm") else "_".join(parts[-2:])
    hash_count = len(parts) - (1 if arch in ("dit", "sd_unet", "llm") else 2)
    hashes = parts[:hash_count]
    return sorted(hashes), arch


def load_phase1_for_config(dataset_dir, sorted_hashes):
    """Assemble phase1 dict by loading pair.json for each pair and lora.json for each lora."""
    pair_stats = {}
    lora_stats = {}
    for i, h in enumerate(sorted_hashes):
        lora_path = dataset_dir / "lora" / f"{h}.lora.json"
        if lora_path.exists():
            lora_stats[i] = json.loads(lora_path.read_text())
    for i, ha in enumerate(sorted_hashes):
        for j, hb in enumerate(sorted_hashes):
            if i >= j:
                continue
            key_a, key_b = sorted([ha, hb])
            pair_path = dataset_dir / "pair" / f"{key_a}_{key_b}.pair.json"
            if pair_path.exists():
                pair_stats[(i, j)] = json.loads(pair_path.read_text())
    return pair_stats, lora_stats


def build_index(local_dataset_dir, out_dir, hf_commit_sha):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = EstimatorFeatureExtractor()

    feature_rows = []
    labels = []
    for cfg_path in sorted((Path(local_dataset_dir) / "config").glob("*.config.json")):
        cfg = json.loads(cfg_path.read_text())
        sorted_hashes, arch = parse_config_name(cfg_path.name)
        pair_stats, lora_stats = load_phase1_for_config(Path(local_dataset_dir), sorted_hashes)
        if not pair_stats and len(sorted_hashes) >= 2:
            continue  # skip configs with no pair stats available
        phase1 = {
            "pair_stats": pair_stats,
            "lora_stats": lora_stats,
            "combo_size": len(sorted_hashes),
            "base_model_family": cfg.get("base_model_family", "unknown"),
        }
        vec = extractor.featurize(phase1)
        feature_rows.append(vec)
        labels.append({
            "arch_preset": cfg.get("arch_preset"),
            "base_model_family": cfg.get("base_model_family", "unknown"),
            "combo_size": len(sorted_hashes),
            "config": cfg.get("config"),
            "candidates": cfg.get("candidates", []),
            "score_final": cfg.get("score", 0.0),
        })

    if not feature_rows:
        raise RuntimeError("no valid configs found to build index")

    X = np.stack(feature_rows).astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid div-by-zero on constant dims
    X_norm = (X - mean) / std

    nn = NearestNeighbors(metric="cosine", n_neighbors=min(10, len(X)))
    nn.fit(X_norm)

    with open(out_dir / "index.pkl", "wb") as f:
        pickle.dump({
            "index": nn,
            "features": X_norm,
            "raw_features": X,
            "labels": labels,
            "zscore": (mean.tolist(), std.tolist()),
        }, f)

    meta = {
        "hf_commit_sha": hf_commit_sha,
        "estimator_index_version": ESTIMATOR_INDEX_VERSION,
        "n_samples": len(feature_rows),
        "feature_dim": int(X.shape[1]),
        "build_timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./models/estimator")
    p.add_argument("--local-dataset-dir", default=None,
                   help="Use local dataset (skip HF download)")
    p.add_argument("--hf-repo", default="ethanfel/lora-optimizer-community-cache")
    args = p.parse_args()

    if args.local_dataset_dir:
        sha = "local"
        ds_dir = args.local_dataset_dir
    else:
        from huggingface_hub import snapshot_download, HfApi
        info = HfApi().dataset_info(args.hf_repo)
        sha = info.sha
        ds_dir = snapshot_download(repo_id=args.hf_repo, repo_type="dataset",
                                   allow_patterns=["config/*", "pair/*", "lora/*"])

    meta = build_index(ds_dir, args.out, sha)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
```

**Step 4: Run test, verify pass**

Run: `pytest tests/test_index_builder.py -v`
Expected: PASS (need to complete fixture setup in the test first — write minimal valid config/pair/lora JSONs)

**Step 5: Commit**

```bash
git add scripts/build_estimator_index.py tests/test_index_builder.py
git commit -m "feat: index builder script for estimator k-NN

Offline pipeline: snapshot_download HF dataset, parse
config/pair/lora JSONs, featurize, z-score, fit NearestNeighbors,
pickle to disk with meta.json sidecar (hf_commit_sha, version,
dims, timestamp)."
```

---

### Task C2: Freshness check utility

**Files:**
- Modify: `lora_estimator.py` — add `check_index_freshness()` + `ensure_index_fresh()`
- Test: `tests/test_estimator_freshness.py`

**Step 1: Write failing test**

```python
# tests/test_estimator_freshness.py
import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

class TestFreshness(unittest.TestCase):
    def test_stale_sha_triggers_rebuild(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"hf_commit_sha": "abc"}
            (Path(tmp) / "meta.json").write_text(json.dumps(meta))
            with mock.patch("lora_estimator._fetch_hf_head_sha", return_value="xyz") as m:
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_called_once()

    def test_fresh_sha_no_rebuild(self):
        from lora_estimator import ensure_index_fresh
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"hf_commit_sha": "abc"}
            (Path(tmp) / "meta.json").write_text(json.dumps(meta))
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
            with mock.patch("lora_estimator._fetch_hf_head_sha", side_effect=Exception("no network")):
                rebuild_fn = mock.Mock()
                ensure_index_fresh(Path(tmp), rebuild_fn=rebuild_fn, mode="auto")
                rebuild_fn.assert_not_called()  # graceful fallback
```

**Step 2: Run, verify fail**

Run: `pytest tests/test_estimator_freshness.py -v`
Expected: FAIL (function not defined)

**Step 3: Implement freshness check**

Add to `lora_estimator.py`:

```python
import json
import logging
from pathlib import Path

HF_REPO_ID = "ethanfel/lora-optimizer-community-cache"


def _fetch_hf_head_sha(repo_id=HF_REPO_ID):
    """Get current HEAD commit SHA of the dataset repo. Raises on network failure."""
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo_id)
    return info.sha


def ensure_index_fresh(index_dir, rebuild_fn, mode="auto", repo_id=HF_REPO_ID):
    """Trigger rebuild_fn() if the index is stale.

    mode: 'auto' (check SHA), 'force' (always rebuild), 'skip' (never rebuild)
    Graceful: on network failure in 'auto', falls back to cache with a warning.
    """
    index_dir = Path(index_dir)
    if mode == "skip":
        return
    if mode == "force":
        rebuild_fn()
        return
    # mode == "auto"
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        rebuild_fn()
        return
    try:
        current_sha = _fetch_hf_head_sha(repo_id)
    except Exception as e:
        logging.warning(f"[LoRA Estimator] HF SHA check failed — using cached index: {e}")
        return
    cached = json.loads(meta_path.read_text())
    if cached.get("hf_commit_sha") != current_sha:
        logging.info(f"[LoRA Estimator] Dataset updated ({cached.get('hf_commit_sha', '?')[:8]} → "
                     f"{current_sha[:8]}), rebuilding index…")
        rebuild_fn()
```

**Step 4: Run, verify pass**

Run: `pytest tests/test_estimator_freshness.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add lora_estimator.py tests/test_estimator_freshness.py
git commit -m "feat: index freshness check with graceful offline fallback

SHA-based refresh: auto (check + rebuild on change), force (always),
skip (never). On HF network failure, 'auto' falls back to cache with
a warning rather than blocking the merge."
```

---

## Phase D — Estimator runtime

### Task D1: `LoRAEstimator` — load index, retrieve, aggregate

**Files:**
- Modify: `lora_estimator.py` — add `LoRAEstimator` class
- Test: `tests/test_estimator_runtime.py`

**Step 1: Write failing test**

```python
# tests/test_estimator_runtime.py
import unittest
import numpy as np

class TestLoRAEstimator(unittest.TestCase):
    def test_retrieve_filters_by_family_and_size(self):
        """Retrieval must only return neighbors matching family + combo_size"""
        from lora_estimator import LoRAEstimator
        est = LoRAEstimator.from_memory(
            features=np.random.rand(10, 80).astype(np.float32),
            labels=[
                {"base_model_family": "zimage", "combo_size": 3, "candidates": []}
                if i < 5 else
                {"base_model_family": "wan", "combo_size": 3, "candidates": []}
                for i in range(10)
            ],
            zscore=(np.zeros(80), np.ones(80)),
        )
        query_vec = np.random.rand(80).astype(np.float32)
        neighbors = est.retrieve(query_vec, base_model_family="zimage", combo_size=3, k=3)
        self.assertEqual(len(neighbors), 3)
        for n in neighbors:
            self.assertEqual(n["label"]["base_model_family"], "zimage")
            self.assertEqual(n["label"]["combo_size"], 3)

    def test_aggregate_candidates(self):
        """Candidates with identical config tuples get their weighted scores summed"""
        from lora_estimator import LoRAEstimator
        neighbors = [
            {"distance": 0.1, "label": {"candidates": [
                {"rank": 1, "config": {"merge_mode": "ties", "sparsification": "disabled"},
                 "score_final": 0.8},
                {"rank": 2, "config": {"merge_mode": "weighted_average", "sparsification": "disabled"},
                 "score_final": 0.7},
            ]}},
            {"distance": 0.2, "label": {"candidates": [
                {"rank": 1, "config": {"merge_mode": "ties", "sparsification": "disabled"},
                 "score_final": 0.85},
            ]}},
        ]
        top_n = LoRAEstimator.aggregate_candidates(neighbors, top_n=2)
        self.assertEqual(len(top_n), 2)
        # 'ties' wins: 0.8 * 1/1.1 + 0.85 * 1/1.2 > 0.7 * 1/1.1
        self.assertEqual(top_n[0]["config"]["merge_mode"], "ties")

    def test_emit_tuner_data_shape(self):
        """Estimator output must match tuner_data schema the optimizer expects"""
        from lora_estimator import LoRAEstimator
        top_n = [
            {"rank": 1, "config": {"merge_mode": "ties", "sparsification": "disabled",
                                    "sparsification_density": 0.7, "dare_dampening": 0.0,
                                    "merge_refinement": "none", "auto_strength": True,
                                    "optimization_mode": "per_prefix", "strategy_set": "full"},
             "score_final": 0.8, "score_heuristic": 0.75, "score_measured": 0.82},
        ]
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
```

**Step 2: Run, verify fail**

Run: `pytest tests/test_estimator_runtime.py -v`
Expected: FAIL

**Step 3: Implement `LoRAEstimator`**

```python
# Appended to lora_estimator.py
import pickle
import numpy as np

class LoRAEstimator:
    def __init__(self, index, features, labels, zscore):
        self.index = index
        self.features = features
        self.labels = labels
        self.zscore_mean, self.zscore_std = zscore

    @classmethod
    def from_disk(cls, index_path):
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        mean, std = data["zscore"]
        return cls(data["index"], data["features"], data["labels"],
                   (np.asarray(mean), np.asarray(std)))

    @classmethod
    def from_memory(cls, features, labels, zscore):
        """Test helper — avoids pickle round-trip."""
        from sklearn.neighbors import NearestNeighbors
        mean, std = zscore
        features = np.asarray(features, dtype=np.float32)
        X_norm = (features - mean) / np.where(std < 1e-8, 1.0, std)
        nn = NearestNeighbors(metric="cosine", n_neighbors=min(10, len(features)))
        nn.fit(X_norm)
        return cls(nn, X_norm, labels, (np.asarray(mean), np.asarray(std)))

    def retrieve(self, feature_vec, base_model_family, combo_size, k=5):
        """Return k nearest neighbors filtered by family+size. Fewer than k if not enough matches."""
        mask = np.array([
            lbl.get("base_model_family") == base_model_family and lbl.get("combo_size") == combo_size
            for lbl in self.labels
        ])
        if mask.sum() == 0:
            return []
        # Normalize query
        q = (feature_vec - self.zscore_mean) / np.where(self.zscore_std < 1e-8, 1.0, self.zscore_std)
        q = q.astype(np.float32).reshape(1, -1)
        # Subset to matching entries, find k nearest within subset
        subset_features = self.features[mask]
        subset_labels = [lbl for lbl, m in zip(self.labels, mask) if m]
        from sklearn.neighbors import NearestNeighbors
        sub_nn = NearestNeighbors(metric="cosine", n_neighbors=min(k, len(subset_features)))
        sub_nn.fit(subset_features)
        distances, indices = sub_nn.kneighbors(q)
        out = []
        for d, i in zip(distances[0], indices[0]):
            out.append({"distance": float(d), "label": subset_labels[i]})
        return out

    @staticmethod
    def aggregate_candidates(neighbors, top_n=3):
        """Sum inverse-distance-weighted scores across neighbors' candidates. Return top-N."""
        if not neighbors:
            return []
        import hashlib
        def _config_key(cfg):
            return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        pool = {}
        for n in neighbors:
            w = 1.0 / (1.0 + n["distance"])
            for cand in n["label"].get("candidates", []):
                cfg = cand.get("config")
                if not cfg:
                    continue
                key = _config_key(cfg)
                entry = pool.setdefault(key, {
                    "config": cfg,
                    "score_final": 0.0,
                    "score_heuristic": 0.0,
                    "score_measured": 0.0,
                    "weight_sum": 0.0,
                })
                entry["score_final"] += w * cand.get("score_final", 0.0)
                entry["score_heuristic"] += w * cand.get("score_heuristic", 0.0)
                entry["score_measured"] += w * cand.get("score_measured", 0.0)
                entry["weight_sum"] += w
        ranked = sorted(pool.values(), key=lambda e: e["score_final"], reverse=True)
        out = []
        for rank, e in enumerate(ranked[:top_n], start=1):
            w = e["weight_sum"] if e["weight_sum"] > 0 else 1.0
            out.append({
                "rank": rank,
                "config": e["config"],
                "score_final": e["score_final"] / w,
                "score_heuristic": e["score_heuristic"] / w,
                "score_measured": e["score_measured"] / w,
                "score_external": None,
                "metrics": {},
                "external_details": None,
            })
        return out

    @staticmethod
    def emit_tuner_data(top_n, source_loras, analysis_summary, normalize_keys="disabled",
                        architecture_preset="auto", auto_strength_floor=-1.0,
                        decision_smoothing=0.25):
        """Produce a tuner_data dict that matches the optimizer's settings_source=from_tuner_data path."""
        return {
            "version": 1,
            "lora_hash": "estimator",
            "source_loras": source_loras,
            "normalize_keys": normalize_keys,
            "architecture_preset": architecture_preset,
            "auto_strength_floor": auto_strength_floor,
            "decision_smoothing": decision_smoothing,
            "analysis_summary": analysis_summary,
            "top_n": top_n,
            "_estimator": True,  # marker so downstream code can tell this is predicted
        }
```

**Step 4: Run, verify pass**

Run: `pytest tests/test_estimator_runtime.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lora_estimator.py tests/test_estimator_runtime.py
git commit -m "feat: LoRAEstimator runtime — retrieve, aggregate, emit tuner_data

Hard-filter retrieval (family + combo_size), cosine k-NN over
z-scored features, inverse-distance-weighted candidate aggregation,
tuner_data emission matching the optimizer's settings_source input."
```

---

## Phase E — Per-prefix retrieval extension

### Task E1: Emit per_prefix_decisions in predicted tuner_data

**Precondition:** Phase A complete, user has rerun the 400 combos, HF dataset now contains `per_prefix_decisions` on each candidate.

**Files:**
- Modify: `lora_estimator.py` — `aggregate_candidates` picks per-prefix decisions from the highest-weighted neighbor; `emit_tuner_data` passes them through
- Test: extend `tests/test_estimator_runtime.py`

**Step 1: Write failing test**

```python
def test_aggregate_includes_per_prefix_decisions(self):
    from lora_estimator import LoRAEstimator
    neighbors = [
        {"distance": 0.1, "label": {"candidates": [
            {"rank": 1,
             "config": {"merge_mode": "ties", "sparsification": "disabled"},
             "score_final": 0.8,
             "per_prefix_decisions": {"layer.0": "ties", "layer.1": "weighted_average"}},
        ]}},
        {"distance": 0.2, "label": {"candidates": [
            {"rank": 1,
             "config": {"merge_mode": "ties", "sparsification": "disabled"},
             "score_final": 0.85,
             "per_prefix_decisions": {"layer.0": "weighted_average", "layer.1": "weighted_average"}},
        ]}},
    ]
    top_n = LoRAEstimator.aggregate_candidates(neighbors, top_n=1)
    self.assertIn("per_prefix_decisions", top_n[0])
    # Decision should come from the higher-weighted neighbor (distance=0.1)
    self.assertEqual(top_n[0]["per_prefix_decisions"]["layer.0"], "ties")
```

**Step 2: Run, verify fail**

Run: `pytest tests/test_estimator_runtime.py::TestLoRAEstimator::test_aggregate_includes_per_prefix_decisions -v`
Expected: FAIL

**Step 3: Extend `aggregate_candidates`**

Inside `aggregate_candidates`, track the highest-weighted contributor per config key and lift its `per_prefix_decisions`:

```python
# Inside the pool-building loop, record the best-weight contributor:
entry = pool.setdefault(key, {
    "config": cfg,
    "score_final": 0.0, "score_heuristic": 0.0, "score_measured": 0.0,
    "weight_sum": 0.0,
    "best_weight": 0.0,
    "per_prefix_decisions": {},
})
if w > entry["best_weight"]:
    entry["best_weight"] = w
    entry["per_prefix_decisions"] = cand.get("per_prefix_decisions", {})
```

And include it in the output dict.

**Step 4: Run, verify pass**

Run: `pytest tests/test_estimator_runtime.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add lora_estimator.py tests/test_estimator_runtime.py
git commit -m "feat: propagate per_prefix_decisions from retrieved neighbors

Per-prefix strategy map is pulled from the highest-weighted neighbor
contributing to each aggregated candidate, passed through to the
emitted tuner_data so the optimizer can replay the predicted
per-prefix policy verbatim."
```

---

### Task E2: Optimizer reads per_prefix_decisions from tuner_data

**Files:**
- Modify: `lora_optimizer.py:6756-6765` — when `tuner_data.top_n[0]` has `per_prefix_decisions`, use the predicted strategy instead of calling `_auto_select_params`
- Test: `tests/test_per_prefix_replay.py`

**Step 1: Write failing test**

```python
# tests/test_per_prefix_replay.py
import unittest
from unittest import mock

class TestPerPrefixReplay(unittest.TestCase):
    def test_optimizer_uses_predicted_decisions_when_present(self):
        """When tuner_data carries per_prefix_decisions, skip _auto_select_params."""
        # ... setup optimizer ...
        tuner_data = {
            "top_n": [{
                "config": {"merge_mode": "per_prefix_auto", "optimization_mode": "per_prefix",
                           # ... other config fields ...
                           },
                "per_prefix_decisions": {"layer.0.attn": "ties"},
            }],
        }
        with mock.patch.object(opt, "_auto_select_params") as mock_select:
            opt.optimize_merge(..., tuner_data=tuner_data, settings_source="from_tuner_data")
            mock_select.assert_not_called()  # bypassed because predictions provided
```

**Step 2: Run, verify fail**

Run: `pytest tests/test_per_prefix_replay.py -v`
Expected: FAIL

**Step 3: Implement replay path**

In `_merge_one_group` at `lora_optimizer.py:6747`, before calling `_auto_select_params`:

```python
elif optimization_mode == "per_prefix" and label_prefix in prefix_stats:
    # Check for predicted per-prefix decision from estimator
    predicted_decisions = (tuner_data or {}).get("_predicted_per_prefix_decisions", {})
    if label_prefix in predicted_decisions:
        pf_mode = predicted_decisions[label_prefix]
        pf_density = 0.5
        pf_sign = "frequency"
        # Skip to override block
    else:
        # Existing _auto_select_params call path
        pf = prefix_stats[label_prefix]
        # ...
```

Thread `tuner_data` into `_merge_one_group` via the closure or parameter.

When the optimizer receives `tuner_data` with `settings_source=from_tuner_data`, it should extract `top_n[0]["per_prefix_decisions"]` and stash it in a closure-accessible location for `_merge_one_group`.

**Step 4: Run, verify pass**

Run: `pytest tests/test_per_prefix_replay.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_per_prefix_replay.py
git commit -m "feat: replay predicted per-prefix decisions in optimizer

When tuner_data from the estimator carries per_prefix_decisions,
the optimizer uses them directly in _merge_one_group instead of
recomputing via _auto_select_params."
```

---

## Phase F — ComfyUI node surface

### Task F1: Register Estimator ComfyUI node

**Files:**
- Modify: `lora_optimizer.py` — add `LoRAEstimatorNode` class + register in `NODE_CLASS_MAPPINGS`
- Test: `tests/test_estimator_node.py`

**Step 1: Write failing test**

```python
# tests/test_estimator_node.py
import unittest
# ... standard stub setup ...

class TestEstimatorNode(unittest.TestCase):
    def test_node_registered(self):
        from lora_optimizer import NODE_CLASS_MAPPINGS
        self.assertIn("LoRAEstimator", NODE_CLASS_MAPPINGS)

    def test_input_types(self):
        from lora_optimizer import NODE_CLASS_MAPPINGS
        node_cls = NODE_CLASS_MAPPINGS["LoRAEstimator"]
        inputs = node_cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("model", required)
        self.assertIn("lora_stack", required)
        self.assertIn("k", required)
        self.assertIn("rebuild_index", required)

    def test_return_types(self):
        from lora_optimizer import NODE_CLASS_MAPPINGS
        node_cls = NODE_CLASS_MAPPINGS["LoRAEstimator"]
        self.assertEqual(node_cls.RETURN_TYPES, ("TUNER_DATA", "STRING"))
        self.assertEqual(node_cls.RETURN_NAMES, ("tuner_data", "estimator_report"))
```

**Step 2: Run, verify fail**

Run: `pytest tests/test_estimator_node.py -v`
Expected: FAIL

**Step 3: Implement node**

```python
# lora_optimizer.py, near the other NODE_CLASS_MAPPINGS entries
class LoRAEstimatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_stack": ("LORA_STACK",),
                "k": ("INT", {"default": 5, "min": 1, "max": 20}),
                "rebuild_index": (["auto", "force", "skip"], {"default": "auto"}),
                "top_n_output": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("TUNER_DATA", "STRING")
    RETURN_NAMES = ("tuner_data", "estimator_report")
    FUNCTION = "estimate"
    CATEGORY = "LoRA/Optimization"

    def estimate(self, model, lora_stack, k, rebuild_index, top_n_output, clip=None):
        from lora_estimator import LoRAEstimator, EstimatorFeatureExtractor, ensure_index_fresh
        import folder_paths
        from pathlib import Path

        index_dir = Path(folder_paths.models_dir) / "estimator"
        index_dir.mkdir(parents=True, exist_ok=True)

        def _rebuild():
            from scripts.build_estimator_index import build_index
            from huggingface_hub import snapshot_download, HfApi
            info = HfApi().dataset_info("ethanfel/lora-optimizer-community-cache")
            ds_dir = snapshot_download(
                repo_id="ethanfel/lora-optimizer-community-cache",
                repo_type="dataset",
                allow_patterns=["config/*", "pair/*", "lora/*"],
            )
            build_index(ds_dir, index_dir, info.sha)

        if not (index_dir / "index.pkl").exists():
            _rebuild()
        else:
            ensure_index_fresh(index_dir, _rebuild, mode=rebuild_index)

        # Run Phase 1 via shared AutoTuner machinery
        # (this reuses existing AutoTuner Phase 1 code — instantiate the tuner,
        # call its analysis pass, pull pair_stats/lora_stats from the result)
        tuner = LoRAAutoTuner()
        phase1 = tuner._run_phase1_for_estimator(model, clip, lora_stack)

        extractor = EstimatorFeatureExtractor()
        feature_vec = extractor.featurize(phase1)

        estimator = LoRAEstimator.from_disk(index_dir / "index.pkl")
        neighbors = estimator.retrieve(
            feature_vec,
            base_model_family=phase1["base_model_family"],
            combo_size=phase1["combo_size"],
            k=k,
        )

        if not neighbors:
            report = (f"[Estimator] No neighbors matched (family={phase1['base_model_family']}, "
                      f"combo_size={phase1['combo_size']}). Fall back to AutoTuner.")
            return (None, report)

        top_n = LoRAEstimator.aggregate_candidates(neighbors, top_n=top_n_output)
        tuner_data = LoRAEstimator.emit_tuner_data(
            top_n=top_n,
            source_loras=[{"name": item.get("name", ""), "strength": item.get("strength", 1.0)}
                          for item in lora_stack],
            analysis_summary={
                "n_prefixes": len(phase1["pair_stats"]),
                "combo_size": phase1["combo_size"],
                "base_model_family": phase1["base_model_family"],
                "mean_neighbor_distance": float(np.mean([n["distance"] for n in neighbors])),
            },
        )
        report = self._build_report(neighbors, top_n)
        return (tuner_data, report)

    def _build_report(self, neighbors, top_n):
        lines = ["=== LoRA Merge Estimator Report ==="]
        lines.append(f"Retrieved {len(neighbors)} neighbor(s):")
        for i, n in enumerate(neighbors, 1):
            lines.append(f"  #{i}: dist={n['distance']:.3f}  score={n['label'].get('score_final', 0):.3f}")
        lines.append(f"\nPredicted top {len(top_n)} config(s):")
        for e in top_n:
            lines.append(f"  rank {e['rank']}: {e['config'].get('merge_mode', '?')}  "
                         f"pred_score={e['score_final']:.3f}")
        return "\n".join(lines)


NODE_CLASS_MAPPINGS["LoRAEstimator"] = LoRAEstimatorNode
NODE_DISPLAY_NAME_MAPPINGS["LoRAEstimator"] = "LoRA Merge Estimator"
```

**Step 4: Need to implement `_run_phase1_for_estimator`**

Extract Phase 1 logic in AutoTuner into a reusable method. Locate the current Phase 1 code inside `optimize_merge` (~line 9056 area) and refactor into `_run_phase1_for_estimator(model, clip, lora_stack)` that returns `{pair_stats, lora_stats, combo_size, base_model_family}`.

**Step 5: Run, verify pass**

Run: `pytest tests/test_estimator_node.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_estimator_node.py
git commit -m "feat: register LoRAEstimator ComfyUI node

Pre-node that runs Phase 1 (shared with AutoTuner), featurizes the
combo, retrieves similar past merges from the HF-backed k-NN index,
and emits a tuner_data struct the optimizer consumes via existing
settings_source=from_tuner_data plug. Freshness-aware, falls back
gracefully on missing index / no neighbors."
```

---

### Task F2: End-to-end integration test

**Files:**
- Create: `tests/test_estimator_integration.py`

**Test flow:**
1. Build a minimal in-memory HF-like dataset (2-3 configs)
2. Run the builder against it → index on disk
3. Instantiate the node, invoke with mock lora_stack
4. Assert tuner_data is returned with expected shape
5. Pass tuner_data to optimizer mock, verify `settings_source=from_tuner_data` path is hit

**Step 1: Write test**

```python
# tests/test_estimator_integration.py
import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

class TestEstimatorEndToEnd(unittest.TestCase):
    def test_pipeline_produces_valid_tuner_data(self):
        # Build a 2-entry fake dataset
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "hf"
            (ds / "config").mkdir(parents=True)
            (ds / "pair").mkdir()
            (ds / "lora").mkdir()
            # ... write minimal config/pair/lora JSONs ...
            out = Path(tmp) / "index"
            from scripts.build_estimator_index import build_index
            build_index(ds, out, "testsha")

            # Instantiate node, patch _run_phase1_for_estimator to return fake stats
            from lora_optimizer import NODE_CLASS_MAPPINGS
            node_cls = NODE_CLASS_MAPPINGS["LoRAEstimator"]
            node = node_cls()
            with mock.patch("folder_paths.models_dir", str(tmp)):
                with mock.patch.object(node, "_run_phase1_for_estimator") as mock_p1:
                    mock_p1.return_value = {
                        "pair_stats": {...}, "lora_stats": {...},
                        "combo_size": 2, "base_model_family": "zimage",
                    }
                    tuner_data, report = node.estimate(
                        model=mock.Mock(), lora_stack=[{"name": "a", "strength": 1.0},
                                                       {"name": "b", "strength": 1.0}],
                        k=2, rebuild_index="skip", top_n_output=2,
                    )
            self.assertIsNotNone(tuner_data)
            self.assertIn("top_n", tuner_data)
            self.assertGreater(len(tuner_data["top_n"]), 0)
            self.assertIn("[Estimator]", report)
```

**Step 2: Run, debug, verify pass**

Run: `pytest tests/test_estimator_integration.py -v`
Expected: PASS (may need iteration on fixture details)

**Step 3: Commit**

```bash
git add tests/test_estimator_integration.py
git commit -m "test: estimator end-to-end integration (build → retrieve → emit)"
```

---

## Phase G — Polish

### Task G1: Manual smoke test

**Not an automated test — user-performed verification.**

1. Pull latest branch in docker.
2. In ComfyUI, wire: `LoRA Stack → LoRA Merge Estimator → (tuner_data) → LoRA Optimizer (settings_source=from_tuner_data)`.
3. Confirm first run triggers index build (~30-60s).
4. Confirm subsequent runs are fast (~1-2s for retrieval).
5. Inspect `estimator_report` output — neighbors + predicted configs sensible?
6. Let merge complete, verify merged model is sensible.

If the build fails with an sklearn import, install the dep and confirm.

### Task G2: Update README/wiki

**Files:**
- Modify: `README.md` (add a short section on the estimator node)
- Modify: `docs/wiki/*` if such a structure exists

**Step 1:** Add a "LoRA Merge Estimator" section to README covering:
- What the node does (one-paragraph pitch)
- Workflow wiring diagram
- Input parameters
- Report interpretation
- When to use vs AutoTuner

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Estimator pre-node section to README"
```

---

## Summary task list (for executing-plans skill)

| # | Task | Files | Depends on |
|---|------|-------|------------|
| A1 | Capture per-prefix decisions in AutoTuner | `lora_optimizer.py`, `tests/test_per_prefix_capture.py` | — |
| A2 | Add decisions to upload payload | `lora_optimizer.py`, same test file | A1 |
| A3 | *(user reruns 400 combos)* | — | A2 |
| B1 | Add sklearn to pyproject | `pyproject.toml` | — |
| B2 | EstimatorFeatureExtractor | `lora_estimator.py`, `tests/test_estimator_features.py` | B1 |
| C1 | Index builder script | `scripts/build_estimator_index.py`, `tests/test_index_builder.py` | B2 |
| C2 | Freshness check | `lora_estimator.py`, `tests/test_estimator_freshness.py` | C1 |
| D1 | LoRAEstimator runtime | `lora_estimator.py`, `tests/test_estimator_runtime.py` | C1 |
| E1 | Per-prefix aggregation | `lora_estimator.py`, `tests/test_estimator_runtime.py` | D1, A3 |
| E2 | Optimizer replay path | `lora_optimizer.py`, `tests/test_per_prefix_replay.py` | E1 |
| F1 | ComfyUI node | `lora_optimizer.py`, `tests/test_estimator_node.py` | D1 |
| F2 | Integration test | `tests/test_estimator_integration.py` | F1 |
| G1 | Manual smoke test | — | F1 |
| G2 | README update | `README.md` | F1 |

**Parallelism:** Tasks A1-A2 (instrumentation) and B1-F1 (estimator) are independent except where noted. Phase E depends on A3 (user rerun) completing.

**Done criteria:** Full pipeline works end-to-end in a fresh ComfyUI environment, all tests pass, manual smoke test produces sensible predictions on a 3-LoRA Zimage combo, estimator report is intelligible.
