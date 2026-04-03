# Analysis Resume Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Save per-prefix analysis results to a `.partial` file after each prefix completes, so a crash (OOM etc.) can be resumed on the next run without restarting from scratch.

**Architecture:** Three-task TDD sequence: (1) add four partial-file static methods mirroring the existing analysis cache I/O methods, (2) add `on_prefix_done` callback to `_run_group_analysis` that fires after each fresh prefix, (3) wire the callback and partial lifecycle (load-on-miss, save-per-prefix, delete-on-success) into `auto_tune`.

**Tech Stack:** Python, `json`, `os.replace` atomic writes, `unittest.mock`, `tempfile`

---

### Task 1: Add partial file I/O static methods to `LoRAAutoTuner`

**Files:**
- Modify: `lora_optimizer.py:7583-7627` (after `_analysis_cache_save`, before `_memory_load`)
- Test: `tests/test_lora_optimizer.py` (after `test_analysis_cache_load_stale_algo_version_returns_none` ~line 1261)

**Step 1: Write the failing tests**

Add to the existing analysis cache test class in `tests/test_lora_optimizer.py` after line 1261:

```python
def test_analysis_partial_path_uses_partial_suffix(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            path = lora_optimizer.LoRAAutoTuner._analysis_partial_path("abc123")
            self.assertTrue(path.endswith("abc123.analysis.partial.json"))

def test_analysis_partial_load_missing_returns_none(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            self.assertIsNone(
                lora_optimizer.LoRAAutoTuner._analysis_partial_load("nonexistent"))

def test_analysis_partial_load_stale_algo_version_returns_none(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            path = os.path.join(tmpdir, "stale.analysis.partial.json")
            with open(path, "w") as f:
                json.dump({"algo_version": "0.0.0", "per_prefix": {}}, f)
            self.assertIsNone(
                lora_optimizer.LoRAAutoTuner._analysis_partial_load("stale"))

def test_analysis_partial_roundtrip(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            per_prefix = {"prefix_a": {"ranks": {"0": 16}, "is_clip": False}}
            source_loras = [{"name": "a.safetensors"}]
            lora_optimizer.LoRAAutoTuner._analysis_partial_save(
                "abc123", per_prefix, source_loras)
            loaded = lora_optimizer.LoRAAutoTuner._analysis_partial_load("abc123")
            self.assertIsNotNone(loaded)
            self.assertIn("prefix_a", loaded)

def test_analysis_partial_save_is_atomic(self):
    """Save uses tmp+replace so a crash mid-write doesn't corrupt the file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            lora_optimizer.LoRAAutoTuner._analysis_partial_save(
                "abc123", {"p": {}}, [])
            partial_path = lora_optimizer.LoRAAutoTuner._analysis_partial_path("abc123")
            tmp_path = partial_path + ".tmp"
            self.assertTrue(os.path.exists(partial_path))
            self.assertFalse(os.path.exists(tmp_path))

def test_analysis_partial_delete_removes_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            lora_optimizer.LoRAAutoTuner._analysis_partial_save(
                "abc123", {}, [])
            path = lora_optimizer.LoRAAutoTuner._analysis_partial_path("abc123")
            self.assertTrue(os.path.exists(path))
            lora_optimizer.LoRAAutoTuner._analysis_partial_delete("abc123")
            self.assertFalse(os.path.exists(path))

def test_analysis_partial_delete_silent_on_missing(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            # Should not raise
            lora_optimizer.LoRAAutoTuner._analysis_partial_delete("nonexistent")
```

**Step 2: Run tests to verify they fail**

```bash
cd /media/p5/ComfyUI-ZImage-LoRA-Merger
python -m pytest tests/test_lora_optimizer.py -k "partial" -v 2>&1 | tail -20
```

Expected: `AttributeError: type object 'LoRAAutoTuner' has no attribute '_analysis_partial_path'`

**Step 3: Implement the four static methods**

In `lora_optimizer.py`, after `_analysis_cache_save` (after line 7627), add:

```python
@staticmethod
def _analysis_partial_path(names_only_hash):
    return os.path.join(AUTOTUNER_MEMORY_DIR,
                        f"{names_only_hash}.analysis.partial.json")

@staticmethod
def _analysis_partial_load(names_only_hash):
    """Load partial analysis checkpoint. Returns per_prefix dict or None."""
    path = LoRAAutoTuner._analysis_partial_path(names_only_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("algo_version") != AUTOTUNER_ALGO_VERSION:
            logging.info("[AutoTuner Analysis Cache] Partial file stale algo version, ignoring")
            return None
        return data.get("per_prefix")
    except Exception as e:
        logging.warning(f"[AutoTuner Analysis Cache] Failed to load partial: {e}")
        return None

@staticmethod
def _analysis_partial_save(names_only_hash, per_prefix, source_loras):
    """Atomic write of partial analysis checkpoint to disk."""
    from datetime import datetime
    path = LoRAAutoTuner._analysis_partial_path(names_only_hash)
    entry = {
        "analysis_version": 1,
        "algo_version": AUTOTUNER_ALGO_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_loras": source_loras,
        "per_prefix": per_prefix,
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(entry, f)
        os.replace(tmp_path, path)
    except Exception as e:
        logging.warning(f"[AutoTuner Analysis Cache] Failed to save partial: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

@staticmethod
def _analysis_partial_delete(names_only_hash):
    """Delete partial checkpoint file, silently ignoring missing files."""
    path = LoRAAutoTuner._analysis_partial_path(names_only_hash)
    try:
        os.unlink(path)
    except OSError:
        pass
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lora_optimizer.py -k "partial" -v 2>&1 | tail -20
```

Expected: all 7 partial tests PASS

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add analysis partial checkpoint I/O methods"
```

---

### Task 2: Add `on_prefix_done` callback to `_run_group_analysis`

**Files:**
- Modify: `lora_optimizer.py:4743-4941` (`_run_group_analysis`)
- Test: `tests/test_lora_optimizer.py`

**Step 1: Write the failing tests**

Add after `test_run_group_analysis_populates_new_entries_on_miss` (~line 1424):

```python
def test_run_group_analysis_calls_on_prefix_done_for_fresh_prefixes(self):
    """on_prefix_done fires once per freshly-computed prefix."""
    optimizer = lora_optimizer.LoRAOptimizer()
    active_loras = [
        _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
        _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
    ]
    model = _make_model()
    target_groups = optimizer._build_target_groups(
        ["prefix_a"], {"prefix_a": "layer.weight"}, {})

    calls = []
    device = torch.device("cpu")
    optimizer._run_group_analysis(
        target_groups, active_loras, model, None, device,
        cached_analysis={},
        track_new_entries=True,
        on_prefix_done=lambda prefix, entry: calls.append((prefix, entry)),
    )
    self.assertEqual(len(calls), 1)
    prefix, entry = calls[0]
    self.assertEqual(prefix, "prefix_a")
    self.assertIn("ranks", entry)

def test_run_group_analysis_does_not_call_on_prefix_done_for_cache_hits(self):
    """on_prefix_done is NOT called for prefixes already in cached_analysis."""
    optimizer = lora_optimizer.LoRAOptimizer()
    active_loras = [
        _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
        _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
    ]
    model = _make_model()
    target_groups = optimizer._build_target_groups(
        ["prefix_a"], {"prefix_a": "layer.weight"}, {})

    fake_cached = {
        "prefix_a": {
            "pair_conflicts": {
                "0,1": {"overlap": 10, "conflict": 2, "dot": 0.1,
                        "norm_a_sq": 1.0, "norm_b_sq": 0.25,
                        "weighted_total": 0.3, "weighted_conflict": 0.05,
                        "expected_conflict": 0.1, "excess_conflict": 0.0,
                        "subspace_overlap": 0.1, "subspace_weight": 0.5}
            },
            "per_lora_norm_sq": {"0": 1.0, "1": 0.25},
            "magnitude_samples_unscaled": {"0": [0.5], "1": [0.3]},
            "ranks": {"0": 1, "1": 1},
            "target_key": "layer.weight",
            "is_clip": False,
            "raw_n": 2,
            "skip_count": 0,
            "strength_signs": {"0": 1, "1": 1},
        }
    }

    calls = []
    device = torch.device("cpu")
    optimizer._run_group_analysis(
        target_groups, active_loras, model, None, device,
        cached_analysis=fake_cached,
        track_new_entries=True,
        on_prefix_done=lambda prefix, entry: calls.append(prefix),
    )
    self.assertEqual(calls, [])
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_lora_optimizer.py -k "on_prefix_done" -v 2>&1 | tail -10
```

Expected: `TypeError: _run_group_analysis() got an unexpected keyword argument 'on_prefix_done'`

**Step 3: Add the parameter and callback calls**

In `lora_optimizer.py`, update the signature at line 4743:

```python
def _run_group_analysis(self, target_groups, active_loras, model, clip,
                        compute_device, clip_strength_multiplier=1.0,
                        merge_refinement="none",
                        decision_smoothing=0.0, progress_cb=None,
                        cached_analysis=None, track_new_entries=False,
                        on_prefix_done=None):
```

In the GPU path (around line 4891-4893), after the `new_analysis_entries[prefix] = entry` assignment:

```python
if result is not None and track_new_entries:
    entry = self._extract_for_analysis_cache(result, active_loras)
    new_analysis_entries[prefix] = entry
    if on_prefix_done is not None:
        on_prefix_done(prefix, entry)
```

In the CPU path (around line 4918-4922), after the equivalent assignment:

```python
if result is not None and track_new_entries:
    prefix = result[0]
    if cached_analysis is None or prefix not in cached_analysis:
        entry = self._extract_for_analysis_cache(result, active_loras)
        new_analysis_entries[prefix] = entry
        if on_prefix_done is not None:
            on_prefix_done(prefix, entry)
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lora_optimizer.py -k "on_prefix_done or new_entries or cache_hit" -v 2>&1 | tail -15
```

Expected: all targeted tests PASS

**Step 5: Run full test suite to verify no regressions**

```bash
python -m pytest tests/test_lora_optimizer.py -x -q 2>&1 | tail -10
```

Expected: all tests pass

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add on_prefix_done callback to _run_group_analysis"
```

---

### Task 3: Wire partial lifecycle into `auto_tune`

**Files:**
- Modify: `lora_optimizer.py:7887-8052` (`auto_tune` analysis section)
- Test: `tests/test_lora_optimizer.py`

**Step 1: Write the failing tests**

Add a new test class after the existing analysis cache tests:

```python
class TestAnalysisPartialLifecycle(unittest.TestCase):
    """Integration tests for partial checkpoint create/resume/delete lifecycle."""

    def _make_minimal_autotuner_mocks(self, tmpdir):
        """Return a patched LoRAAutoTuner and minimal run args."""
        import lora_optimizer
        tuner = lora_optimizer.LoRAAutoTuner()
        return tuner

    def test_partial_file_created_during_analysis(self):
        """A .partial file exists on disk while analysis runs (simulated by checking after)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                active_loras = [
                    _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
                    _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
                ]
                names_only_hash, _ = tuner._compute_names_only_hash(active_loras)
                source_loras = [{"name": l["name"]} for l in active_loras]

                written_prefixes = []
                partial_accumulated = {}

                def on_prefix_done(prefix, entry):
                    partial_accumulated[prefix] = entry
                    tuner._analysis_partial_save(names_only_hash, partial_accumulated, source_loras)
                    written_prefixes.append(prefix)

                model = _make_model()
                target_groups = tuner._build_target_groups(
                    ["prefix_a"], {"prefix_a": "layer.weight"}, {})
                device = torch.device("cpu")
                tuner._run_group_analysis(
                    target_groups, active_loras, model, None, device,
                    cached_analysis={},
                    track_new_entries=True,
                    on_prefix_done=on_prefix_done,
                )

                self.assertIn("prefix_a", written_prefixes)
                partial_path = tuner._analysis_partial_path(names_only_hash)
                self.assertTrue(os.path.exists(partial_path))

    def test_partial_file_deleted_after_successful_save(self):
        """After full analysis cache is saved, the .partial file is removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                hash_val = "testhash99"
                tuner._analysis_partial_save(hash_val, {"prefix_a": {}}, [])
                partial_path = tuner._analysis_partial_path(hash_val)
                self.assertTrue(os.path.exists(partial_path))

                tuner._analysis_cache_save(hash_val, {"prefix_a": {}}, [])
                tuner._analysis_partial_delete(hash_val)

                self.assertFalse(os.path.exists(partial_path))
                full_path = tuner._analysis_cache_path(hash_val)
                self.assertTrue(os.path.exists(full_path))

    def test_partial_file_loaded_as_cached_analysis_on_resume(self):
        """If no full cache but .partial exists, it is loaded as cached_analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                hash_val = "resumehash1"
                per_prefix = {"prefix_b": {"ranks": {"0": 16}, "is_clip": False}}
                tuner._analysis_partial_save(hash_val, per_prefix, [])

                # Full cache miss
                self.assertIsNone(tuner._analysis_cache_load(hash_val))
                # Partial hit
                loaded = tuner._analysis_partial_load(hash_val)
                self.assertIsNotNone(loaded)
                self.assertIn("prefix_b", loaded)
```

**Step 2: Run tests to verify they pass already (they test helpers, not wiring)**

```bash
python -m pytest tests/test_lora_optimizer.py -k "TestAnalysisPartialLifecycle" -v 2>&1 | tail -15
```

Expected: all 3 pass (they only test helper methods, not `auto_tune` wiring)

**Step 3: Wire partial load into `auto_tune`**

In `lora_optimizer.py`, replace the analysis cache load block at line 7887-7892:

```python
cached_analysis = self._analysis_cache_load(names_only_hash)
if cached_analysis is not None:
    logging.info(
        f"[AutoTuner Analysis Cache] HIT — {len(cached_analysis)} prefixes cached")
else:
    cached_analysis = self._analysis_partial_load(names_only_hash)
    if cached_analysis is not None:
        logging.info(
            f"[AutoTuner Analysis Cache] Partial resume — "
            f"{len(cached_analysis)} prefixes already done")
    else:
        logging.info("[AutoTuner Analysis Cache] MISS — will run full analysis")
```

**Step 4: Wire partial delete in `clear_and_run` mode**

In `lora_optimizer.py`, inside the `clear_and_run` block at line 7936-7941, add one line:

```python
if memory_mode == "clear_and_run":
    self._memory_clear(memory_lora_hash, settings_hash)
    analysis_path = self._analysis_cache_path(names_only_hash)
    if os.path.exists(analysis_path):
        os.unlink(analysis_path)
    self._analysis_partial_delete(names_only_hash)   # <-- add this
    cached_analysis = None
```

**Step 5: Wire callback and delete-on-success around `_run_group_analysis` call**

In `lora_optimizer.py`, replace the `_run_group_analysis` call at line 8027-8052:

```python
source_loras_for_cache = [{"name": item["name"]} for item in active_loras]
partial_accumulated = dict(cached_analysis or {})

def _on_prefix_done(prefix, entry):
    partial_accumulated[prefix] = entry
    self._analysis_partial_save(names_only_hash, partial_accumulated, source_loras_for_cache)

analysis_data = self._run_group_analysis(
    target_groups, active_loras, model, clip, compute_device,
    clip_strength_multiplier=clip_strength_multiplier,
    merge_refinement="none",
    decision_smoothing=decision_smoothing,
    progress_cb=lambda: pbar.update(1),
    cached_analysis=cached_analysis,
    track_new_entries=True,
    on_prefix_done=_on_prefix_done,
)
```

Then update the post-analysis save block (line 8047-8052) to also delete the partial:

```python
new_analysis_entries = analysis_data.get("new_analysis_entries", {})
if new_analysis_entries:
    merged = dict(cached_analysis or {})
    merged.update(new_analysis_entries)
    self._analysis_cache_save(names_only_hash, merged, source_loras_for_cache)
    self._analysis_partial_delete(names_only_hash)
```

Note: remove the local `source_loras` variable that was defined just before `_analysis_cache_save` (it's now `source_loras_for_cache` defined earlier).

**Step 6: Run full test suite**

```bash
python -m pytest tests/test_lora_optimizer.py -x -q 2>&1 | tail -10
```

Expected: all tests pass

**Step 7: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: wire analysis partial checkpoint lifecycle into auto_tune"
```
