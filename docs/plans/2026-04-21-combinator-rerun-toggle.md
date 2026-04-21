# Combinator Rerun Toggle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a temporary `rerun_mode` boolean to `LoRACombinationGenerator` that routes progress tracking to a separate file so the existing 400-combo collection can be re-run (to backfill per-prefix decisions) without clobbering the original progress state.

**Architecture:** Single new BOOLEAN input on the node. When `True`, the progress load/save paths resolve to `combo_progress_rerun.json` instead of `combo_progress.json`. Iteration order is unchanged (driven by `shuffle_order`), so the rerun follows the same sequence as the original collection. Once the rerun finishes, the toggle and the side file are deleted.

**Tech Stack:** Python, ComfyUI node API, unittest.

**Prerequisite:** Phase A of the estimator plan (`docs/plans/2026-04-21-estimator-prenode.md`) must land first — the rerun only has value once `per_prefix_decisions` is being captured.

**Scope:** Two tasks.
1. Toggle + separate progress file.
2. HF-aware skip: in rerun mode, before processing a combo, check HF for a matching enriched config (any candidate carries `per_prefix_decisions`); if found, mark the combo complete and move on.

---

## Task 1: Add `rerun_mode` input and progress routing

**Files:**
- Modify: `lora_optimizer.py:12144-12242` (`LoRACombinationGenerator`)
- Modify: `tests/test_lora_optimizer.py` (`TestLoRACombinationGenerator`, around line 2625)

**Step 1: Write the failing tests**

Append to `TestLoRACombinationGenerator` in `tests/test_lora_optimizer.py`:

```python
    # -- rerun mode --

    def test_input_types_has_rerun_mode(self):
        inputs = lora_optimizer.LoRACombinationGenerator.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("rerun_mode", req)
        spec = req["rerun_mode"]
        self.assertEqual(spec[0], "BOOLEAN")
        self.assertEqual(spec[1]["default"], False)

    def test_resolve_progress_path_default(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        path = gen._resolve_progress_path(rerun_mode=False)
        self.assertTrue(path.endswith("combo_progress.json"))
        self.assertFalse(path.endswith("combo_progress_rerun.json"))

    def test_resolve_progress_path_rerun(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        path = gen._resolve_progress_path(rerun_mode=True)
        self.assertTrue(path.endswith("combo_progress_rerun.json"))

    def test_resolve_progress_path_rerun_same_dir_as_default(self):
        """Rerun progress file lives next to the default one."""
        gen = lora_optimizer.LoRACombinationGenerator()
        default_dir = os.path.dirname(gen._resolve_progress_path(rerun_mode=False))
        rerun_dir = os.path.dirname(gen._resolve_progress_path(rerun_mode=True))
        self.assertEqual(default_dir, rerun_dir)
```

**Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v -k "rerun or resolve_progress_path"`

Expected: 4 failures — `rerun_mode` key absent, `_resolve_progress_path` method absent.

**Step 3: Implement the minimal changes**

In `lora_optimizer.py`, modify `LoRACombinationGenerator`:

Add to `INPUT_TYPES` required dict (after `folder_filter`):

```python
                "rerun_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "TEMPORARY: re-run all combos into a separate "
                               "progress file (combo_progress_rerun.json). "
                               "Use once to backfill per-prefix decisions, "
                               "then disable and delete the side file."}),
```

Add a helper method on the class:

```python
    def _resolve_progress_path(self, rerun_mode):
        if rerun_mode:
            return os.path.join(
                os.path.dirname(self._progress_path),
                "combo_progress_rerun.json",
            )
        return self._progress_path
```

Update `get_next_combo` signature and body:

```python
    def get_next_combo(self, shuffle_order, strength, combo_size,
                       folder_filter="", rerun_mode=False):
        # ... existing prefix/lora loading unchanged ...

        progress_path = self._resolve_progress_path(rerun_mode)
        combos = self._generate_combos(lora_names, combo_size)
        shuffled = self._shuffle_combos(combos, shuffle_order)
        completed, _ = self._load_progress(progress_path)
        total = len(shuffled)
        logging.info("[LoRA Combo] shuffle_order=%d, pool=%d LoRAs, %d combos, "
                     "%d already completed, rerun_mode=%s, progress file: %s",
                     shuffle_order, len(lora_names), total, len(completed),
                     rerun_mode, progress_path)
        # ... rest of method unchanged, but replace every
        #     self._save_progress(self._progress_path, ...)
        # with
        #     self._save_progress(progress_path, ...)
```

Also update `IS_CHANGED` to accept the new kwarg so ComfyUI doesn't cache stale invalidations:

```python
    @classmethod
    def IS_CHANGED(cls, shuffle_order, strength, combo_size,
                   folder_filter="", rerun_mode=False):
        return float("nan")
```

**Step 4: Run the tests to verify they pass**

Run the focused set:
```
pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v
```

Expected: all tests in the class pass (new ones green, old ones still green).

Then run the whole file to catch regressions:
```
pytest tests/test_lora_optimizer.py -v
```

Expected: PASS across the board (no imports changed, no existing behavior altered for `rerun_mode=False`).

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat(combinator): add temporary rerun_mode toggle

Routes progress tracking to combo_progress_rerun.json when enabled, so
the existing 400-combo collection can be re-run end-to-end (to backfill
per-prefix decisions) without disturbing combo_progress.json. Iteration
order is unchanged. Intended to be reverted once the rerun completes."
```

---

## Task 2: HF-enrichment skip check (rerun mode only)

**Files:**
- Modify: `lora_optimizer.py` (`LoRACombinationGenerator.__init__`, `get_next_combo`, new helpers)
- Modify: `tests/test_lora_optimizer.py` (`TestLoRACombinationGenerator`)

**Design:**
- `__init__` gains two per-instance caches: `_enrichment_cache` (joined_hashes → bool) and `_hf_files_cache` (list[str] | None) so we pay the HF file-list round-trip at most once per combinator instance.
- `_list_hf_config_files()` uses `huggingface_hub.HfApi().list_repo_files` against `COMMUNITY_CACHE_REPO` (dataset), returns the cached `config/*.config.json` subset; on failure returns `[]` and logs a warning (rerun proceeds without the skip).
- `_combo_already_enriched(combo)` computes each member's content hash via `LoRAAutoTuner._lora_content_hash({"name": name})` (reuses the existing file-mtime cache), builds `prefix = f"config/{joined}_"`, filters the cached HF file list, downloads matching files via `LoRAAutoTuner._community_download`, and returns `True` iff any `candidates[*].per_prefix_decisions` is non-empty.
- In `get_next_combo`, when `rerun_mode=True` and the combo passed the existing skip checks, call `_combo_already_enriched(combo)`; if True, mark the combo hash in `completed`, persist the rerun progress file, and loop to the next combo via the existing skip pattern.

**Step 1: Write the failing tests**

Append to `TestLoRACombinationGenerator`:

```python
    # -- rerun HF-enrichment skip --

    def test_hf_file_list_is_memoized(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        call_count = {"n": 0}

        def fake_list(*args, **kwargs):
            call_count["n"] += 1
            return ["config/aaa_bbb_dit.config.json", "lora/aaa.lora.json"]

        with unittest.mock.patch(
            "lora_optimizer.HfApi",
            create=True,
            return_value=unittest.mock.MagicMock(list_repo_files=fake_list),
        ):
            first = gen._list_hf_config_files()
            second = gen._list_hf_config_files()
        self.assertEqual(first, ["config/aaa_bbb_dit.config.json"])
        self.assertEqual(second, first)
        self.assertEqual(call_count["n"], 1)

    def test_combo_already_enriched_true_when_any_candidate_has_decisions(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = ["config/aaa_bbb_dit.config.json"]
        enriched = {
            "candidates": [
                {"per_prefix_decisions": {}},
                {"per_prefix_decisions": {"layer.0": "ties"}},
            ],
        }
        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            side_effect=lambda item: {"a": "aaa", "b": "bbb"}[item["name"]],
        ), unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_community_download",
            return_value=enriched,
        ):
            self.assertTrue(gen._combo_already_enriched(("a", "b")))

    def test_combo_already_enriched_false_when_candidates_lack_decisions(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = ["config/aaa_bbb_dit.config.json"]
        not_enriched = {"candidates": [{"per_prefix_decisions": {}}]}
        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            side_effect=lambda item: {"a": "aaa", "b": "bbb"}[item["name"]],
        ), unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_community_download",
            return_value=not_enriched,
        ):
            self.assertFalse(gen._combo_already_enriched(("a", "b")))

    def test_combo_already_enriched_false_when_no_matching_hf_config(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = ["config/zzz_yyy_dit.config.json"]
        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            side_effect=lambda item: {"a": "aaa", "b": "bbb"}[item["name"]],
        ):
            self.assertFalse(gen._combo_already_enriched(("a", "b")))

    def test_combo_already_enriched_memoizes_result(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = ["config/aaa_bbb_dit.config.json"]
        download_calls = {"n": 0}

        def fake_download(path):
            download_calls["n"] += 1
            return {"candidates": [{"per_prefix_decisions": {"l": "ties"}}]}

        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            side_effect=lambda item: {"a": "aaa", "b": "bbb"}[item["name"]],
        ), unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_community_download",
            side_effect=fake_download,
        ):
            self.assertTrue(gen._combo_already_enriched(("a", "b")))
            self.assertTrue(gen._combo_already_enriched(("a", "b")))
        self.assertEqual(download_calls["n"], 1)

    def test_combo_already_enriched_handles_missing_content_hash(self):
        """If any LoRA hash cannot be computed, fall back to 'not enriched'
        so the combo still runs."""
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = []
        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            return_value=None,
        ):
            self.assertFalse(gen._combo_already_enriched(("a", "b")))
```

Make sure `import unittest.mock` is in scope (add at top of the test file if missing).

**Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v -k "hf_file_list or combo_already_enriched"`

Expected: 6 failures — `_list_hf_config_files` and `_combo_already_enriched` not defined; `_hf_files_cache` / `_enrichment_cache` attributes missing.

**Step 3: Implement**

In `lora_optimizer.py`:

Update `__init__`:

```python
    def __init__(self):
        self._progress_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "combo_progress.json"
        )
        self._enrichment_cache = {}
        self._hf_files_cache = None
```

Add helpers on the class:

```python
    def _list_hf_config_files(self):
        if self._hf_files_cache is not None:
            return self._hf_files_cache
        try:
            from huggingface_hub import HfApi
            files = HfApi().list_repo_files(
                repo_id=COMMUNITY_CACHE_REPO, repo_type="dataset",
            )
            self._hf_files_cache = [f for f in files if f.startswith("config/")]
        except Exception as exc:
            logging.warning("[LoRA Combo] HF file list failed (%s) — "
                            "skip check disabled for this session.", exc)
            self._hf_files_cache = []
        return self._hf_files_cache

    def _combo_already_enriched(self, combo):
        content_hashes = []
        for name in combo:
            ch = LoRAAutoTuner._lora_content_hash({"name": name})
            if ch is None:
                return False
            content_hashes.append(ch)
        joined = "_".join(sorted(content_hashes))
        if joined in self._enrichment_cache:
            return self._enrichment_cache[joined]
        prefix = f"config/{joined}_"
        matching = [f for f in self._list_hf_config_files()
                    if f.startswith(prefix)]
        enriched = False
        for path in matching:
            data = LoRAAutoTuner._community_download(path)
            if not data:
                continue
            for cand in data.get("candidates", []):
                if cand.get("per_prefix_decisions"):
                    enriched = True
                    break
            if enriched:
                break
        self._enrichment_cache[joined] = enriched
        return enriched
```

Extend the skip loop in `get_next_combo` (inside `while combo is not None:`, after the LoRA-loading block that sets `skip`):

```python
            if not skip and rerun_mode:
                if self._combo_already_enriched(combo):
                    logging.info("[LoRA Combo] Already enriched on HF — "
                                 "skipping: %s", " + ".join(combo))
                    completed.add(self._combo_hash(combo))
                    self._save_progress(progress_path, completed, total)
                    skip = True

            if not skip:
                break
            combo = self._find_next(shuffled, completed)
```

**Step 4: Run the tests to verify they pass**

```
pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v
pytest tests/test_lora_optimizer.py -v
```

Expected: all green. Existing tests still pass because the enrichment check is gated on `rerun_mode=True`.

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat(combinator): skip combos already enriched on HF during rerun

When rerun_mode is on, before running a combo the combinator checks the
HF community cache for a matching config whose candidates carry
per_prefix_decisions. Enriched combos are marked complete in the rerun
progress file and skipped. HF file list and per-combo results are cached
per-instance so the extra traffic is bounded."
```

---

## Usage (after Phase A lands)

1. Delete the side file if it exists: `rm -f combo_progress_rerun.json`.
2. In the workflow, set `rerun_mode=True` on `LoRACombinationGenerator`.
3. Run until the combinator raises `InterruptProcessingException` ("All N combinations completed").
4. Confirm the HF dataset now shows `per_prefix_decisions` on the re-uploaded configs.
5. Revert this commit (or manually remove the input + helper) and delete `combo_progress_rerun.json`.

## Out of scope

- Adapting the default workflow JSONs. The new input has a default of `False`, so existing saved workflows are unaffected.
- Cross-architecture family filtering during the enrichment check. The `config/{joined}_*.config.json` wildcard covers every arch the same content-hash set could have been uploaded under, which is the intended behavior.
