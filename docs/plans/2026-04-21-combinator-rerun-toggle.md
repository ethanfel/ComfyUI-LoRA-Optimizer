# Combinator Rerun Toggle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a temporary `rerun_mode` boolean to `LoRACombinationGenerator` that routes progress tracking to a separate file so the existing 400-combo collection can be re-run (to backfill per-prefix decisions) without clobbering the original progress state.

**Architecture:** Single new BOOLEAN input on the node. When `True`, the progress load/save paths resolve to `combo_progress_rerun.json` instead of `combo_progress.json`. Iteration order is unchanged (driven by `shuffle_order`), so the rerun follows the same sequence as the original collection. Once the rerun finishes, the toggle and the side file are deleted.

**Tech Stack:** Python, ComfyUI node API, unittest.

**Prerequisite:** Phase A of the estimator plan (`docs/plans/2026-04-21-estimator-prenode.md`) must land first — the rerun only has value once `per_prefix_decisions` is being captured.

**Scope boundary:** No HF-aware "skip if already enriched" logic. A separate local progress file is enough: interrupted reruns resume from their own progress, and the original `combo_progress.json` is untouched.

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

## Usage (after Phase A lands)

1. Delete the side file if it exists: `rm -f combo_progress_rerun.json`.
2. In the workflow, set `rerun_mode=True` on `LoRACombinationGenerator`.
3. Run until the combinator raises `InterruptProcessingException` ("All N combinations completed").
4. Confirm the HF dataset now shows `per_prefix_decisions` on the re-uploaded configs.
5. Revert this commit (or manually remove the input + helper) and delete `combo_progress_rerun.json`.

## Out of scope

- HF-aware skip logic (checking each combo's current HF state before running). A separate progress file plus the same seed is sufficient.
- Adapting the default workflow JSONs. The new input has a default of `False`, so existing saved workflows are unaffected.
