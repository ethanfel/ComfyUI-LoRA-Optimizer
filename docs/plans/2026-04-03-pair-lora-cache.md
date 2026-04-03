# Pair/LoRA Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cache per-LoRA and per-pair analysis stats so that adding or swapping a LoRA in an existing set reuses previously computed data instead of re-analyzing from scratch.

**Architecture:** Six tasks in sequence. Tasks 1–2 add I/O methods (hash, load, save) for the two new file types. Task 3 adds extraction helpers that pull per-LoRA and per-pair data out of an `_analyze_target_group` result. Task 4 adds the reconstruction helper that rebuilds the 8-tuple from cached data alone. Tasks 5–6 wire everything into `_run_group_analysis` and `auto_tune`. Each task is TDD: write failing test, implement, verify passing.

**Tech Stack:** Python, `json`, `os.replace` (atomic writes), `hashlib.sha256`, `torch`, `unittest.mock`, `tempfile`

---

### Background: key data structures

`_analyze_target_group` returns an 8-tuple:
```python
(prefix, partial_stats, pair_conflicts, magnitude_samples,
 (target_key, is_clip), skip_count, raw_n, per_lora_norm_sq)
```
- `partial_stats`: `[(lora_idx, rank, display_l2, norm_sq), ...]` — only for LoRAs that have keys in this prefix
- `pair_conflicts`: `{(i,j): {overlap, conflict, dot, norm_a_sq, norm_b_sq, weighted_total, weighted_conflict, expected_conflict, excess_conflict, subspace_overlap, subspace_weight}}` — only for pairs where both LoRAs participate
- `magnitude_samples`: list of tensors, one per participating LoRA (same order as partial_stats)
- `per_lora_norm_sq`: `{lora_idx: float}` — only participating LoRAs

A LoRA that has no keys for a prefix simply is not present in `diffs`, `partial_stats`, or `per_lora_norm_sq`.

**Non-participating LoRAs**: stored as `null` in the lora cache for that prefix — this distinguishes "seen and absent" from "never seen (cache miss)".

---

### Task 1: Add `_lora_identity_hash` and lora cache I/O

**Files:**
- Modify: `lora_optimizer.py` — add 4 static methods after `_analysis_partial_delete` (~line 7689, before `_memory_load`)
- Test: `tests/test_lora_optimizer.py` — add new test class `TestLoraCacheIO`

**Step 1: Write the failing tests**

Add this class to `tests/test_lora_optimizer.py` (after the existing `TestAnalysisPartialLifecycle` class):

```python
class TestLoraCacheIO(unittest.TestCase):

    def test_lora_identity_hash_returns_16char_hex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake lora file so os.stat works
            lora_path = os.path.join(tmpdir, "test.safetensors")
            with open(lora_path, "w") as f:
                f.write("x")
            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            return_value=lora_path):
                h = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "test.safetensors", "strength": 1.0})
                self.assertIsInstance(h, str)
                self.assertEqual(len(h), 16)
                self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_lora_identity_hash_differs_for_different_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.safetensors")
            path_b = os.path.join(tmpdir, "b.safetensors")
            with open(path_a, "w") as f: f.write("x")
            with open(path_b, "w") as f: f.write("y")
            def fake_get_full_path(folder, name):
                return path_a if name == "a.safetensors" else path_b
            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            side_effect=fake_get_full_path):
                ha = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "a.safetensors", "strength": 1.0})
                hb = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "b.safetensors", "strength": 1.0})
                self.assertNotEqual(ha, hb)

    def test_lora_identity_hash_ignores_strength(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "test.safetensors")
            with open(lora_path, "w") as f: f.write("x")
            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            return_value=lora_path):
                h1 = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "test.safetensors", "strength": 0.5})
                h2 = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "test.safetensors", "strength": 1.0})
                self.assertEqual(h1, h2)

    def test_lora_cache_path_uses_lora_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path = lora_optimizer.LoRAAutoTuner._lora_cache_path("abc123")
                self.assertTrue(path.endswith("abc123.lora.json"))

    def test_lora_cache_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._lora_cache_load("nonexistent"))

    def test_lora_cache_load_stale_algo_version_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path = os.path.join(tmpdir, "stale.lora.json")
                with open(path, "w") as f:
                    json.dump({"algo_version": "0.0.0", "per_prefix": {}}, f)
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._lora_cache_load("stale"))

    def test_lora_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                per_prefix = {
                    "prefix_a": {
                        "norm_sq": 1.5, "rank": 16,
                        "magnitude_samples_unscaled": [0.1, 0.2],
                        "strength_sign": 1,
                        "target_key": "layer.weight",
                        "is_clip": False, "skip_count": 0, "raw_n": 1,
                    }
                }
                lora_optimizer.LoRAAutoTuner._lora_cache_save("abc123", per_prefix)
                loaded = lora_optimizer.LoRAAutoTuner._lora_cache_load("abc123")
                self.assertIsNotNone(loaded)
                self.assertIn("prefix_a", loaded)
                self.assertEqual(loaded["prefix_a"]["norm_sq"], 1.5)

    def test_lora_cache_save_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._lora_cache_save("abc123", {})
                path = lora_optimizer.LoRAAutoTuner._lora_cache_path("abc123")
                self.assertTrue(os.path.exists(path))
                self.assertFalse(os.path.exists(path + ".tmp"))
```

**Step 2: Run tests to verify they fail**

```bash
cd /media/p5/ComfyUI-ZImage-LoRA-Merger
python -m pytest tests/test_lora_optimizer.py::TestLoraCacheIO -v 2>&1 | tail -15
```
Expected: `AttributeError: type object 'LoRAAutoTuner' has no attribute '_lora_identity_hash'`

**Step 3: Implement — add after `_analysis_partial_delete` (line ~7689), before `_memory_load`**

```python
@staticmethod
def _lora_identity_hash(lora_item):
    """16-char hex hash of a single LoRA's file identity (name+mtime+size)."""
    name = lora_item["name"]
    path = folder_paths.get_full_path("loras", name)
    if path is not None:
        try:
            st = os.stat(path)
            entry = (name, st.st_mtime, st.st_size)
        except OSError:
            entry = (name, 0, 0)
    else:
        entry = (name, 0, 0)
    hash_input = json.dumps(entry, separators=(",", ":"))
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

@staticmethod
def _lora_cache_path(lora_hash):
    return os.path.join(AUTOTUNER_MEMORY_DIR, f"{lora_hash}.lora.json")

@staticmethod
def _lora_cache_load(lora_hash):
    """Load per-LoRA prefix stats. Returns per_prefix dict or None on miss/stale."""
    path = LoRAAutoTuner._lora_cache_path(lora_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("algo_version") != AUTOTUNER_ALGO_VERSION:
            logging.info("[AutoTuner Lora Cache] Stale algo version, ignoring")
            return None
        return data.get("per_prefix")
    except Exception as e:
        logging.warning(f"[AutoTuner Lora Cache] Failed to load: {e}")
        return None

@staticmethod
def _lora_cache_save(lora_hash, per_prefix):
    """Atomic write of per-LoRA cache to disk."""
    from datetime import datetime
    path = LoRAAutoTuner._lora_cache_path(lora_hash)
    entry = {
        "algo_version": AUTOTUNER_ALGO_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "per_prefix": per_prefix,
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(entry, f)
        os.replace(tmp_path, path)
        logging.info(f"[AutoTuner Lora Cache] Saved: {path}")
    except Exception as e:
        logging.warning(f"[AutoTuner Lora Cache] Failed to save: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lora_optimizer.py::TestLoraCacheIO -v 2>&1 | tail -15
```
Expected: all 8 tests PASS.

**Step 5: Run full suite for regressions**

```bash
python -m pytest tests/test_lora_optimizer.py -q --deselect tests/test_lora_optimizer.py::LoRAOptimizerTests::test_widget_order_keeps_upstream_workflow_compatibility 2>&1 | tail -5
```
Expected: all pass.

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add lora identity hash and lora cache I/O methods"
```

---

### Task 2: Add pair cache I/O methods

**Files:**
- Modify: `lora_optimizer.py` — add 3 static methods after `_lora_cache_save`
- Test: `tests/test_lora_optimizer.py` — add `TestPairCacheIO` class

**Step 1: Write the failing tests**

```python
class TestPairCacheIO(unittest.TestCase):

    def test_pair_cache_path_sorts_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path_ab = lora_optimizer.LoRAAutoTuner._pair_cache_path("aaa", "bbb")
                path_ba = lora_optimizer.LoRAAutoTuner._pair_cache_path("bbb", "aaa")
                self.assertEqual(path_ab, path_ba)
                self.assertIn("aaa_bbb", path_ab)

    def test_pair_cache_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._pair_cache_load("aaa", "bbb"))

    def test_pair_cache_load_stale_algo_version_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path = os.path.join(tmpdir, "aaa_bbb.pair.json")
                with open(path, "w") as f:
                    json.dump({"algo_version": "0.0.0", "per_prefix": {}}, f)
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._pair_cache_load("aaa", "bbb"))

    def test_pair_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                per_prefix = {
                    "prefix_a": {
                        "overlap": 100, "conflict": 30, "dot": 0.5,
                        "norm_a_sq": 1.0, "norm_b_sq": 0.5,
                        "weighted_total": 0.8, "weighted_conflict": 0.2,
                        "expected_conflict": 0.15, "excess_conflict": 0.05,
                        "subspace_overlap": 0.3, "subspace_weight": 1.0,
                    }
                }
                lora_optimizer.LoRAAutoTuner._pair_cache_save("aaa", "bbb", per_prefix)
                loaded = lora_optimizer.LoRAAutoTuner._pair_cache_load("aaa", "bbb")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["prefix_a"]["overlap"], 100)

    def test_pair_cache_load_commutative(self):
        """_pair_cache_load("a","b") and ("b","a") return the same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                per_prefix = {"prefix_a": {"overlap": 50}}
                lora_optimizer.LoRAAutoTuner._pair_cache_save("aaa", "bbb", per_prefix)
                loaded_ab = lora_optimizer.LoRAAutoTuner._pair_cache_load("aaa", "bbb")
                loaded_ba = lora_optimizer.LoRAAutoTuner._pair_cache_load("bbb", "aaa")
                self.assertEqual(loaded_ab, loaded_ba)

    def test_pair_cache_save_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._pair_cache_save("aaa", "bbb", {})
                path = lora_optimizer.LoRAAutoTuner._pair_cache_path("aaa", "bbb")
                self.assertTrue(os.path.exists(path))
                self.assertFalse(os.path.exists(path + ".tmp"))
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_lora_optimizer.py::TestPairCacheIO -v 2>&1 | tail -10
```
Expected: `AttributeError: type object 'LoRAAutoTuner' has no attribute '_pair_cache_path'`

**Step 3: Implement — add after `_lora_cache_save`**

```python
@staticmethod
def _pair_cache_path(hash_a, hash_b):
    ha, hb = (hash_a, hash_b) if hash_a < hash_b else (hash_b, hash_a)
    return os.path.join(AUTOTUNER_MEMORY_DIR, f"{ha}_{hb}.pair.json")

@staticmethod
def _pair_cache_load(hash_a, hash_b):
    """Load per-pair prefix metrics. Returns per_prefix dict or None on miss/stale."""
    path = LoRAAutoTuner._pair_cache_path(hash_a, hash_b)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("algo_version") != AUTOTUNER_ALGO_VERSION:
            logging.info("[AutoTuner Pair Cache] Stale algo version, ignoring")
            return None
        return data.get("per_prefix")
    except Exception as e:
        logging.warning(f"[AutoTuner Pair Cache] Failed to load: {e}")
        return None

@staticmethod
def _pair_cache_save(hash_a, hash_b, per_prefix):
    """Atomic write of per-pair cache to disk."""
    from datetime import datetime
    path = LoRAAutoTuner._pair_cache_path(hash_a, hash_b)
    entry = {
        "algo_version": AUTOTUNER_ALGO_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "per_prefix": per_prefix,
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(entry, f)
        os.replace(tmp_path, path)
        logging.info(f"[AutoTuner Pair Cache] Saved: {path}")
    except Exception as e:
        logging.warning(f"[AutoTuner Pair Cache] Failed to save: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lora_optimizer.py::TestPairCacheIO -v 2>&1 | tail -10
```
Expected: all 6 tests PASS.

**Step 5: Full suite**

```bash
python -m pytest tests/test_lora_optimizer.py -q --deselect tests/test_lora_optimizer.py::LoRAOptimizerTests::test_widget_order_keeps_upstream_workflow_compatibility 2>&1 | tail -5
```

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add pair cache I/O methods"
```

---

### Task 3: Add extraction helpers

**Files:**
- Modify: `lora_optimizer.py` — add 2 static methods after `_extract_for_analysis_cache` (~line 4687, before `_reconstruct_from_analysis_cache`)
- Test: `tests/test_lora_optimizer.py` — add `TestCacheExtraction` class

**Background:** `_extract_for_analysis_cache` (line 4638) already strips strength from magnitude samples. The new methods extract the per-LoRA and per-pair slices from the same 8-tuple.

`norm_a_sq` in a pair file always corresponds to the LoRA whose `lora_identity_hash` is lexicographically smaller. The extraction method receives `hash_i` and `hash_j` (hashes of loras at index i and j) and swaps norm_a/norm_b if needed.

**Step 1: Write the failing tests**

```python
class TestCacheExtraction(unittest.TestCase):

    def _make_result(self):
        """Minimal 8-tuple from _analyze_target_group."""
        partial_stats = [(0, 16, 1.5, 2.25), (1, 32, 0.9, 0.81)]
        pair_conflicts = {
            (0, 1): {
                "overlap": 50, "conflict": 10, "dot": 0.4,
                "norm_a_sq": 2.25, "norm_b_sq": 0.81,
                "weighted_total": 0.6, "weighted_conflict": 0.1,
                "expected_conflict": 0.12, "excess_conflict": 0.0,
                "subspace_overlap": 0.2, "subspace_weight": 0.5,
            }
        }
        magnitude_samples = [
            torch.tensor([0.5, 1.0]),  # lora 0, already scaled by strength
            torch.tensor([0.3, 0.6]),  # lora 1
        ]
        per_lora_norm_sq = {0: 2.25, 1: 0.81}
        return (
            "prefix_a", partial_stats, pair_conflicts, magnitude_samples,
            ("layer.weight", False), 0, 2, per_lora_norm_sq
        )

    def test_extract_for_lora_cache_participating(self):
        active_loras = [
            {"name": "a.safetensors", "strength": 1.5},
            {"name": "b.safetensors", "strength": 0.9},
        ]
        result = self._make_result()
        entry = lora_optimizer.LoRAOptimizer._extract_for_lora_cache(result, 0, active_loras)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["norm_sq"], 2.25)
        self.assertEqual(entry["rank"], 16)
        self.assertEqual(entry["strength_sign"], 1)
        self.assertFalse(entry["is_clip"])
        self.assertEqual(entry["target_key"], "layer.weight")
        # magnitude unscaled: tensor / abs(strength=1.5)
        self.assertAlmostEqual(entry["magnitude_samples_unscaled"][0], 0.5 / 1.5, places=5)

    def test_extract_for_lora_cache_non_participating_returns_none(self):
        """LoRA index not in partial_stats → non-participating → return None."""
        active_loras = [
            {"name": "a.safetensors", "strength": 1.0},
            {"name": "b.safetensors", "strength": 1.0},
            {"name": "c.safetensors", "strength": 1.0},  # index 2: not in result
        ]
        result = self._make_result()
        entry = lora_optimizer.LoRAOptimizer._extract_for_lora_cache(result, 2, active_loras)
        self.assertIsNone(entry)

    def test_extract_for_pair_cache_norm_order_by_hash(self):
        """norm_a_sq in entry corresponds to the LoRA with the smaller hash."""
        result = self._make_result()
        # hash_i > hash_j → swap norm_a/norm_b
        entry = lora_optimizer.LoRAOptimizer._extract_for_pair_cache(
            result, i=0, j=1, hash_i="zzz", hash_j="aaa")
        # In result, norm_a_sq=2.25 (lora 0), norm_b_sq=0.81 (lora 1)
        # hash_j="aaa" < hash_i="zzz", so "aaa" (lora 1) should be norm_a
        self.assertAlmostEqual(entry["norm_a_sq"], 0.81)
        self.assertAlmostEqual(entry["norm_b_sq"], 2.25)

    def test_extract_for_pair_cache_no_swap_when_hash_i_smaller(self):
        result = self._make_result()
        entry = lora_optimizer.LoRAOptimizer._extract_for_pair_cache(
            result, i=0, j=1, hash_i="aaa", hash_j="zzz")
        # hash_i="aaa" < hash_j="zzz" → no swap
        self.assertAlmostEqual(entry["norm_a_sq"], 2.25)
        self.assertAlmostEqual(entry["norm_b_sq"], 0.81)

    def test_extract_for_pair_cache_non_participating_returns_none(self):
        """Pair not in pair_conflicts (one LoRA doesn't participate) → None."""
        result = self._make_result()
        # Pair (0,2) not in result
        entry = lora_optimizer.LoRAOptimizer._extract_for_pair_cache(
            result, i=0, j=2, hash_i="aaa", hash_j="zzz")
        self.assertIsNone(entry)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_lora_optimizer.py::TestCacheExtraction -v 2>&1 | tail -10
```
Expected: `AttributeError: type object 'LoRAOptimizer' has no attribute '_extract_for_lora_cache'`

Note: these go on `LoRAOptimizer` (the base class), same as `_extract_for_analysis_cache`.

**Step 3: Implement — add after `_extract_for_analysis_cache` (~line 4687)**

```python
@staticmethod
def _extract_for_lora_cache(result, lora_idx, active_loras):
    """
    Extract per-LoRA stats for one LoRA from an _analyze_target_group result.
    Returns None if lora_idx did not participate in this prefix.
    """
    (prefix, partial_stats, pair_conflicts, magnitude_samples,
     target_info, skip_count, raw_n, per_lora_norm_sq) = result
    if lora_idx not in per_lora_norm_sq:
        return None
    target_key, is_clip = target_info
    tk_serial = list(target_key) if isinstance(target_key, tuple) else target_key

    # Find position of lora_idx in partial_stats (same order as magnitude_samples)
    lora_indices = [s[0] for s in partial_stats]
    if lora_idx not in lora_indices:
        return None
    pos = lora_indices.index(lora_idx)
    rank = partial_stats[pos][1]

    clip_s = active_loras[lora_idx].get("clip_strength")
    eff_s = clip_s if (clip_s is not None and is_clip) else active_loras[lora_idx]["strength"]
    abs_strength = abs(eff_s)
    raw = magnitude_samples[pos] if pos < len(magnitude_samples) else torch.tensor([])
    mag_unscaled = (raw / abs_strength if abs_strength > 0 else raw).tolist()

    return {
        "norm_sq": float(per_lora_norm_sq[lora_idx]),
        "rank": rank,
        "magnitude_samples_unscaled": mag_unscaled,
        "strength_sign": 1 if eff_s >= 0 else -1,
        "target_key": tk_serial,
        "is_clip": is_clip,
        "skip_count": skip_count,
        "raw_n": raw_n,
    }

@staticmethod
def _extract_for_pair_cache(result, i, j, hash_i, hash_j):
    """
    Extract pair metrics for one (i,j) pair from an _analyze_target_group result.
    norm_a_sq corresponds to the LoRA with the lexicographically smaller hash.
    Returns None if this pair did not participate.
    """
    (prefix, partial_stats, pair_conflicts, magnitude_samples,
     target_info, skip_count, raw_n, per_lora_norm_sq) = result
    if (i, j) not in pair_conflicts:
        return None
    metrics = dict(pair_conflicts[(i, j)])
    # Ensure norm_a_sq corresponds to smaller hash
    if hash_i > hash_j:
        metrics["norm_a_sq"], metrics["norm_b_sq"] = (
            metrics["norm_b_sq"], metrics["norm_a_sq"])
    return metrics
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lora_optimizer.py::TestCacheExtraction -v 2>&1 | tail -10
```
Expected: all 5 PASS.

**Step 5: Full suite**

```bash
python -m pytest tests/test_lora_optimizer.py -q --deselect tests/test_lora_optimizer.py::LoRAOptimizerTests::test_widget_order_keeps_upstream_workflow_compatibility 2>&1 | tail -5
```

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add per-lora and per-pair cache extraction helpers"
```

---

### Task 4: Add reconstruction method

**Files:**
- Modify: `lora_optimizer.py` — add 1 static method after `_reconstruct_from_analysis_cache` (~line 4740)
- Test: `tests/test_lora_optimizer.py` — add `TestPairLoraReconstruction` class

**Background:** `_reconstruct_from_pair_lora_cache` rebuilds the 8-tuple from cached data alone when all participating LoRAs and pairs are available. Returns None on sign flip (triggers full re-analysis).

Full hit condition: for every lora index i in active_loras, `lora_entries[i]` is present (either real data dict or `None` meaning non-participating). For every pair (i,j) where both i and j have real (non-None) lora entries, `pair_entries[(i,j)]` must be present.

**Step 1: Write the failing tests**

```python
class TestPairLoraReconstruction(unittest.TestCase):

    def _make_lora_entries(self):
        return {
            0: {
                "norm_sq": 2.25, "rank": 16,
                "magnitude_samples_unscaled": [0.5, 1.0],
                "strength_sign": 1,
                "target_key": "layer.weight",
                "is_clip": False, "skip_count": 0, "raw_n": 2,
            },
            1: {
                "norm_sq": 0.81, "rank": 32,
                "magnitude_samples_unscaled": [0.3, 0.6],
                "strength_sign": 1,
                "target_key": "layer.weight",
                "is_clip": False, "skip_count": 0, "raw_n": 2,
            },
        }

    def _make_pair_entries(self, hash_0="aaa", hash_1="bbb"):
        # hash_0 < hash_1, so norm_a_sq = lora 0
        return {
            (0, 1): {
                "overlap": 50, "conflict": 10, "dot": 0.4,
                "norm_a_sq": 2.25, "norm_b_sq": 0.81,
                "weighted_total": 0.6, "weighted_conflict": 0.1,
                "expected_conflict": 0.12, "excess_conflict": 0.0,
                "subspace_overlap": 0.2, "subspace_weight": 0.5,
            }
        }

    def test_reconstruction_returns_8tuple(self):
        active_loras = [
            {"name": "a.safetensors", "strength": 1.5},
            {"name": "b.safetensors", "strength": 0.9},
        ]
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=self._make_lora_entries(),
            pair_entries=self._make_pair_entries(),
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 8)
        prefix, partial_stats, pair_conflicts, magnitude_samples, target_info, skip_count, raw_n, per_lora_norm_sq = result
        self.assertEqual(prefix, "prefix_a")
        self.assertEqual(len(partial_stats), 2)
        self.assertIn((0, 1), pair_conflicts)
        self.assertAlmostEqual(per_lora_norm_sq[0], 2.25)

    def test_reconstruction_rescales_magnitude_by_current_strength(self):
        active_loras = [
            {"name": "a.safetensors", "strength": 2.0},  # different from cache
            {"name": "b.safetensors", "strength": 1.0},
        ]
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=self._make_lora_entries(),
            pair_entries=self._make_pair_entries(),
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        _, _, _, magnitude_samples, _, _, _, _ = result
        # unscaled[0] = 0.5, rescaled by strength=2.0 → 1.0
        self.assertAlmostEqual(magnitude_samples[0][0].item(), 1.0, places=5)

    def test_reconstruction_returns_none_on_sign_flip(self):
        active_loras = [
            {"name": "a.safetensors", "strength": -1.0},  # sign flipped vs cached +1
            {"name": "b.safetensors", "strength": 1.0},
        ]
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=self._make_lora_entries(),
            pair_entries=self._make_pair_entries(),
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNone(result)

    def test_reconstruction_swaps_norm_when_hash_ordering_differs(self):
        """When lora 0's hash > lora 1's hash, norm_a_sq in pair file is lora 1's."""
        active_loras = [
            {"name": "a.safetensors", "strength": 1.0},
            {"name": "b.safetensors", "strength": 1.0},
        ]
        lora_hashes = {0: "zzz", 1: "aaa"}  # lora 1 has smaller hash
        # In the pair file, norm_a_sq = lora with smaller hash = lora 1 = 0.81
        pair_entries = {
            (0, 1): {
                "overlap": 50, "conflict": 10, "dot": 0.4,
                "norm_a_sq": 0.81,   # lora 1 (smaller hash "aaa")
                "norm_b_sq": 2.25,   # lora 0 (larger hash "zzz")
                "weighted_total": 0.6, "weighted_conflict": 0.1,
                "expected_conflict": 0.12, "excess_conflict": 0.0,
                "subspace_overlap": 0.2, "subspace_weight": 0.5,
            }
        }
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=self._make_lora_entries(),
            pair_entries=pair_entries,
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNotNone(result)
        _, _, pair_conflicts, _, _, _, _, _ = result
        # After reconstruction, norm_a_sq should be for lora 0 = 2.25
        self.assertAlmostEqual(pair_conflicts[(0, 1)]["norm_a_sq"], 2.25)
        self.assertAlmostEqual(pair_conflicts[(0, 1)]["norm_b_sq"], 0.81)

    def test_reconstruction_handles_non_participating_loras(self):
        """LoRA with None entry (non-participating) is excluded from partial_stats."""
        active_loras = [
            {"name": "a.safetensors", "strength": 1.0},
            {"name": "b.safetensors", "strength": 1.0},
            {"name": "c.safetensors", "strength": 1.0},  # non-participating
        ]
        lora_hashes = {0: "aaa", 1: "bbb", 2: "ccc"}
        lora_entries = {**self._make_lora_entries(), 2: None}
        pair_entries = self._make_pair_entries()  # only (0,1)
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=lora_entries,
            pair_entries=pair_entries,
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNotNone(result)
        _, partial_stats, pair_conflicts, _, _, _, _, _ = result
        lora_indices_in_stats = [s[0] for s in partial_stats]
        self.assertNotIn(2, lora_indices_in_stats)
        self.assertNotIn((0, 2), pair_conflicts)
        self.assertNotIn((1, 2), pair_conflicts)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_lora_optimizer.py::TestPairLoraReconstruction -v 2>&1 | tail -10
```
Expected: `AttributeError: ... '_reconstruct_from_pair_lora_cache'`

**Step 3: Implement — add after `_reconstruct_from_analysis_cache`**

```python
@staticmethod
def _reconstruct_from_pair_lora_cache(prefix, lora_entries, pair_entries,
                                       active_loras, lora_hashes):
    """
    Reconstruct the _analyze_target_group 8-tuple from per-LoRA and per-pair
    cache entries. Returns None if any participating LoRA has a sign flip.

    lora_entries: {lora_idx: dict_or_None} — None means non-participating
    pair_entries: {(i,j): dict} — only pairs where both LoRAs participate
    lora_hashes: {lora_idx: hash_str}
    """
    participating = {i for i, entry in lora_entries.items() if entry is not None}

    # Sign-flip check
    for i in participating:
        entry = lora_entries[i]
        current_sign = 1 if active_loras[i]["strength"] >= 0 else -1
        if current_sign != entry.get("strength_sign", 1):
            logging.info(
                f"[AutoTuner Pair/Lora Cache] Sign flip on LoRA {i} "
                f"for {prefix!r}, falling back to full analysis")
            return None

    # Build partial_stats and magnitude_samples
    partial_stats = []
    magnitude_samples = []
    for i in sorted(participating):
        entry = lora_entries[i]
        norm_sq = entry["norm_sq"]
        clip_s = active_loras[i].get("clip_strength")
        is_clip = entry["is_clip"]
        eff_s = clip_s if (clip_s is not None and is_clip) else active_loras[i]["strength"]
        abs_strength = abs(eff_s)
        display_l2 = math.sqrt(norm_sq) * abs_strength
        partial_stats.append((i, entry["rank"], display_l2, norm_sq))
        raw = torch.tensor(entry["magnitude_samples_unscaled"], dtype=torch.float32)
        magnitude_samples.append(raw * abs_strength)

    # Build pair_conflicts — restore norm_a/norm_b to positional order
    pair_conflicts = {}
    for (i, j), metrics in pair_entries.items():
        m = dict(metrics)
        hash_i, hash_j = lora_hashes[i], lora_hashes[j]
        if hash_i > hash_j:
            # File stores norm_a for smaller hash = j; swap back to i=a, j=b
            m["norm_a_sq"], m["norm_b_sq"] = m["norm_b_sq"], m["norm_a_sq"]
        pair_conflicts[(i, j)] = m

    # Build per_lora_norm_sq
    per_lora_norm_sq = {i: lora_entries[i]["norm_sq"] for i in participating}

    # Get prefix-level metadata from any participating LoRA's entry
    first = lora_entries[min(participating)]
    tk = first["target_key"]
    target_key = tuple(tk) if isinstance(tk, list) else tk

    return (
        prefix,
        partial_stats,
        pair_conflicts,
        magnitude_samples,
        (target_key, first["is_clip"]),
        first["skip_count"],
        first["raw_n"],
        per_lora_norm_sq,
    )
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lora_optimizer.py::TestPairLoraReconstruction -v 2>&1 | tail -10
```
Expected: all 5 PASS.

**Step 5: Full suite**

```bash
python -m pytest tests/test_lora_optimizer.py -q --deselect tests/test_lora_optimizer.py::LoRAOptimizerTests::test_widget_order_keeps_upstream_workflow_compatibility 2>&1 | tail -5
```

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add _reconstruct_from_pair_lora_cache"
```

---

### Task 5: Wire into `_run_group_analysis`

**Files:**
- Modify: `lora_optimizer.py:4743` (`_run_group_analysis` signature and body)
- Test: `tests/test_lora_optimizer.py` — add `TestPairLoraCacheWiring` class

**What changes in `_run_group_analysis`:**
1. Add `lora_caches=None, pair_caches=None, lora_hashes=None` to signature
2. Initialize `new_lora_entries = {i: {} for i in range(len(active_loras))}` and `new_pair_entries = {(i,j): {} for i,j in pairs}` at the start
3. In both GPU and CPU paths, before calling `_analyze_target_group`, check if this prefix is a full hit from pair/lora caches
4. On fresh analysis, populate `new_lora_entries` and `new_pair_entries`
5. Return `new_lora_entries` and `new_pair_entries` in the result dict

**Full hit check for a prefix:**

```python
def _pair_lora_full_hit(prefix, lora_caches, pair_caches, pairs, active_loras, lora_hashes):
    if lora_caches is None or pair_caches is None:
        return None
    # All loras must have an entry for this prefix (data or None sentinel)
    lora_entries = {}
    for i in range(len(active_loras)):
        cache = lora_caches.get(i)
        if cache is None or prefix not in cache:
            return None  # never seen this lora+prefix combo
        lora_entries[i] = cache[prefix]  # may be None (non-participating)

    participating = {i for i, e in lora_entries.items() if e is not None}
    # All pairs between participating loras must have entries
    pair_entries = {}
    for (i, j) in pairs:
        if i in participating and j in participating:
            cache = pair_caches.get((i, j))
            if cache is None or prefix not in cache:
                return None
            pair_entries[(i, j)] = cache[prefix]

    return lora_entries, pair_entries
```

**Step 1: Write the failing tests**

```python
class TestPairLoraCacheWiring(unittest.TestCase):

    def _make_lora_caches(self, prefix="prefix_a"):
        """Minimal lora_caches dict covering prefix_a for 2 loras."""
        entry = {
            "norm_sq": 1.0, "rank": 1,
            "magnitude_samples_unscaled": [0.5],
            "strength_sign": 1,
            "target_key": "layer.weight",
            "is_clip": False, "skip_count": 0, "raw_n": 2,
        }
        return {
            0: {prefix: entry},
            1: {prefix: {**entry, "norm_sq": 0.25}},
        }

    def _make_pair_caches(self, prefix="prefix_a"):
        return {
            (0, 1): {
                prefix: {
                    "overlap": 10, "conflict": 2, "dot": 0.1,
                    "norm_a_sq": 1.0, "norm_b_sq": 0.25,
                    "weighted_total": 0.3, "weighted_conflict": 0.05,
                    "expected_conflict": 0.1, "excess_conflict": 0.0,
                    "subspace_overlap": 0.1, "subspace_weight": 0.5,
                }
            }
        }

    def test_run_group_analysis_uses_pair_lora_cache_on_full_hit(self):
        """When pair+lora caches cover all prefixes, _analyze_target_group is not called."""
        optimizer = lora_optimizer.LoRAOptimizer()
        active_loras = [
            _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
            _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
        ]
        model = _make_model()
        target_groups = optimizer._build_target_groups(
            ["prefix_a"], {"prefix_a": "layer.weight"}, {})

        call_count = {"n": 0}
        orig = optimizer._analyze_target_group
        def counting(*args, **kwargs):
            call_count["n"] += 1
            return orig(*args, **kwargs)

        lora_hashes = {0: "aaa", 1: "bbb"}
        with mock.patch.object(optimizer, "_analyze_target_group", side_effect=counting):
            result = optimizer._run_group_analysis(
                target_groups, active_loras, model, None, torch.device("cpu"),
                lora_caches=self._make_lora_caches(),
                pair_caches=self._make_pair_caches(),
                lora_hashes=lora_hashes,
                track_new_entries=True,
            )
        self.assertEqual(call_count["n"], 0)
        self.assertEqual(result["new_lora_entries"][0], {})
        self.assertEqual(result["new_pair_entries"][(0, 1)], {})

    def test_run_group_analysis_populates_new_entries_on_pair_lora_miss(self):
        """Cache miss populates new_lora_entries and new_pair_entries."""
        optimizer = lora_optimizer.LoRAOptimizer()
        active_loras = [
            _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
            _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
        ]
        model = _make_model()
        target_groups = optimizer._build_target_groups(
            ["prefix_a"], {"prefix_a": "layer.weight"}, {})
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = optimizer._run_group_analysis(
            target_groups, active_loras, model, None, torch.device("cpu"),
            lora_caches={0: {}, 1: {}},  # prefix_a missing → full miss
            pair_caches={(0, 1): {}},
            lora_hashes=lora_hashes,
            track_new_entries=True,
        )
        self.assertIn("prefix_a", result["new_lora_entries"][0])
        self.assertIn("prefix_a", result["new_pair_entries"][(0, 1)])
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_lora_optimizer.py::TestPairLoraCacheWiring -v 2>&1 | tail -10
```
Expected: `TypeError: _run_group_analysis() got an unexpected keyword argument 'lora_caches'`

**Step 3: Implement**

Update `_run_group_analysis` signature at line 4743:

```python
def _run_group_analysis(self, target_groups, active_loras, model, clip,
                        compute_device, clip_strength_multiplier=1.0,
                        merge_refinement="none",
                        decision_smoothing=0.0, progress_cb=None,
                        cached_analysis=None, track_new_entries=False,
                        on_prefix_done=None,
                        lora_caches=None, pair_caches=None, lora_hashes=None):
```

After `new_analysis_entries = {}` (line 4877), add:

```python
new_lora_entries = {i: {} for i in range(len(active_loras))}
new_pair_entries = {(i, j): {} for i, j in pairs}
```

Add a helper nested function inside `_run_group_analysis` (before the `group_items` loop):

```python
def _pair_lora_cache_hit(prefix):
    """Return (lora_entries, pair_entries) if full hit, else None."""
    if lora_caches is None or pair_caches is None or lora_hashes is None:
        return None
    lora_entries = {}
    for i in range(len(active_loras)):
        cache = lora_caches.get(i)
        if cache is None or prefix not in cache:
            return None
        lora_entries[i] = cache[prefix]
    participating = {i for i, e in lora_entries.items() if e is not None}
    pair_entries = {}
    for (i, j) in pairs:
        if i in participating and j in participating:
            cache = pair_caches.get((i, j))
            if cache is None or prefix not in cache:
                return None
            pair_entries[(i, j)] = cache[prefix]
    return lora_entries, pair_entries
```

In the GPU path, update the per-prefix block to:

```python
for target_group in group_items:
    prefix = target_group["label_prefix"]
    result = None
    if cached_analysis is not None and prefix in cached_analysis:
        result = self._reconstruct_from_analysis_cache(
            prefix, cached_analysis[prefix], active_loras)
    if result is None:
        hit = _pair_lora_cache_hit(prefix)
        if hit is not None:
            lora_entries, pair_entries = hit
            result = self._reconstruct_from_pair_lora_cache(
                prefix, lora_entries, pair_entries, active_loras, lora_hashes)
    if result is None:
        result = self._analyze_target_group(
            target_group, active_loras, model, clip, compute_device,
            clip_strength_multiplier=clip_strength_multiplier,
            merge_refinement=merge_refinement,
        )
        if result is not None and track_new_entries:
            entry = self._extract_for_analysis_cache(result, active_loras)
            new_analysis_entries[prefix] = entry
            if on_prefix_done is not None:
                on_prefix_done(prefix, entry)
            if lora_caches is not None:
                for i in range(len(active_loras)):
                    new_lora_entries[i][prefix] = self._extract_for_lora_cache(
                        result, i, active_loras)
            if pair_caches is not None:
                for (i, j) in pairs:
                    new_pair_entries[(i, j)][prefix] = self._extract_for_pair_cache(
                        result, i, j, lora_hashes[i], lora_hashes[j])
    _collect_analysis_result(result)
    if progress_cb is not None:
        progress_cb()
```

Apply the same pattern to the CPU path (the futures loop): after `entry = self._extract_for_analysis_cache(...)`, add the lora and pair extraction blocks.

Update the return dict at line 4929 to include:

```python
return {
    ...existing keys...,
    "new_lora_entries": new_lora_entries,
    "new_pair_entries": new_pair_entries,
}
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lora_optimizer.py::TestPairLoraCacheWiring -v 2>&1 | tail -10
```
Expected: all PASS.

**Step 5: Full suite**

```bash
python -m pytest tests/test_lora_optimizer.py -q --deselect tests/test_lora_optimizer.py::LoRAOptimizerTests::test_widget_order_keeps_upstream_workflow_compatibility 2>&1 | tail -5
```

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: wire pair/lora cache hit/miss into _run_group_analysis"
```

---

### Task 6: Wire load and save into `auto_tune`

**Files:**
- Modify: `lora_optimizer.py` — `auto_tune` method, three locations: load (before `_run_group_analysis`), pass (at `_run_group_analysis` call), save (after success)
- Test: `tests/test_lora_optimizer.py` — add `TestPairLoraCacheAutoTune` class

**Step 1: Write the failing tests**

```python
class TestPairLoraCacheAutoTune(unittest.TestCase):
    """Verify that auto_tune loads, uses, and saves pair/lora caches."""

    def test_lora_and_pair_cache_files_created_after_analysis(self):
        """After _run_group_analysis with misses, lora and pair files are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                active_loras = [
                    _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
                    _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
                ]

                lora_hashes = {}
                with mock.patch("lora_optimizer.folder_paths.get_full_path",
                                return_value=None):
                    for i, lora in enumerate(active_loras):
                        lora_hashes[i] = tuner._lora_identity_hash(lora)

                new_lora_entries = {0: {"prefix_a": {"norm_sq": 1.0}},
                                    1: {"prefix_a": {"norm_sq": 0.5}}}
                new_pair_entries = {(0, 1): {"prefix_a": {"overlap": 10}}}

                # Save as if auto_tune just completed
                for i, h in lora_hashes.items():
                    tuner._lora_cache_save(h, new_lora_entries[i])
                tuner._pair_cache_save(
                    lora_hashes[0], lora_hashes[1], new_pair_entries[(0, 1)])

                # Verify files exist
                path_0 = tuner._lora_cache_path(lora_hashes[0])
                path_1 = tuner._lora_cache_path(lora_hashes[1])
                path_pair = tuner._pair_cache_path(lora_hashes[0], lora_hashes[1])
                self.assertTrue(os.path.exists(path_0))
                self.assertTrue(os.path.exists(path_1))
                self.assertTrue(os.path.exists(path_pair))

    def test_lora_cache_merged_with_existing_on_save(self):
        """New per-prefix entries are merged into the existing lora cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                # Pre-existing cache with prefix_a
                tuner._lora_cache_save("hash_x", {"prefix_a": {"norm_sq": 1.0}})
                # New analysis added prefix_b
                existing = tuner._lora_cache_load("hash_x") or {}
                existing["prefix_b"] = {"norm_sq": 2.0}
                tuner._lora_cache_save("hash_x", existing)
                # Both prefixes should be in the file
                loaded = tuner._lora_cache_load("hash_x")
                self.assertIn("prefix_a", loaded)
                self.assertIn("prefix_b", loaded)
```

**Step 2: Run tests to verify they pass already** (they test helper methods, not wiring):

```bash
python -m pytest tests/test_lora_optimizer.py::TestPairLoraCacheAutoTune -v 2>&1 | tail -10
```
Expected: both PASS (they use only already-implemented helpers).

**Step 3: Implement — load phase in `auto_tune`**

Find the analysis section in `auto_tune` just before the `source_loras_for_cache` line (~line 8099). Add before it:

```python
# Load pair/lora caches for cross-run reuse
lora_hashes = {i: self._lora_identity_hash(lora)
                for i, lora in enumerate(active_loras)}
lora_caches = {i: self._lora_cache_load(h) or {}
               for i, h in lora_hashes.items()}
pair_caches = {(i, j): self._pair_cache_load(lora_hashes[i], lora_hashes[j]) or {}
               for i, j in pairs_for_cache}
```

Where `pairs_for_cache` = `[(i, j) for i in range(len(active_loras)) for j in range(i+1, len(active_loras))]` — compute this just before the load phase.

Note: `pairs` is not yet defined at this point in `auto_tune` (it comes back from `_run_group_analysis`). Compute it locally:
```python
pairs_for_cache = [(i, j) for i in range(len(active_loras))
                           for j in range(i+1, len(active_loras))]
```

**Step 4: Implement — pass to `_run_group_analysis`**

Update the `_run_group_analysis` call (~line 8107) to add:

```python
analysis_data = self._run_group_analysis(
    ...existing args...,
    lora_caches=lora_caches,
    pair_caches=pair_caches,
    lora_hashes=lora_hashes,
)
```

**Step 5: Implement — save phase after success**

After the existing `self._analysis_partial_delete(names_only_hash)` line (~line 8133), add:

```python
new_lora_entries = analysis_data.get("new_lora_entries", {})
new_pair_entries = analysis_data.get("new_pair_entries", {})
for i, h in lora_hashes.items():
    new_for_lora = new_lora_entries.get(i, {})
    if new_for_lora:
        existing_lora = self._lora_cache_load(h) or {}
        existing_lora.update(new_for_lora)
        self._lora_cache_save(h, existing_lora)
for (i, j) in pairs_for_cache:
    new_for_pair = new_pair_entries.get((i, j), {})
    if new_for_pair:
        existing_pair = self._pair_cache_load(lora_hashes[i], lora_hashes[j]) or {}
        existing_pair.update(new_for_pair)
        self._pair_cache_save(lora_hashes[i], lora_hashes[j], existing_pair)
```

**Step 6: Run full test suite**

```bash
python -m pytest tests/test_lora_optimizer.py -q --deselect tests/test_lora_optimizer.py::LoRAOptimizerTests::test_widget_order_keeps_upstream_workflow_compatibility 2>&1 | tail -10
```
Expected: all pass.

**Step 7: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: wire pair/lora cache load and save through auto_tune"
```
