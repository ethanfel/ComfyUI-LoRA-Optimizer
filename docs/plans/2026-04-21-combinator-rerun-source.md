# Combinator `rerun_source` Filter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `LoRACombinationGenerator` with a `rerun_source` enum (`shuffle` | `original_progress`) that, when `rerun_mode=True`, can restrict iteration to combos that appear in the original `combo_progress.json`. Any combos that were never processed originally are ignored; any LoRAs added since the original run don't leak into the rerun.

**Architecture:** New required `rerun_source` enum on the node (active only when `rerun_mode=True`). When set to `original_progress`, `get_next_combo` loads the *original* `combo_progress.json` and filters `shuffled` down to combos whose hash is in that set before `_find_next` runs. The rerun progress file and HF-enrichment skip are unchanged.

**Tech Stack:** Python, ComfyUI node API, unittest.

**Prerequisite:** The `rerun_mode` toggle from `docs/plans/2026-04-21-combinator-rerun-toggle.md` is already landed (commits `07f1fe5`, `c6665f1` on branch `feat/acestep-merge-optimization`).

---

## Task 1: Add `rerun_source` input and progress-intersection filter

**Files:**
- Modify: `lora_optimizer.py` (`LoRACombinationGenerator.INPUT_TYPES`, `IS_CHANGED`, `get_next_combo`)
- Modify: `tests/test_lora_optimizer.py` (`TestLoRACombinationGenerator`)

**Step 1: Write the failing tests**

Append to `TestLoRACombinationGenerator` in `tests/test_lora_optimizer.py`:

```python
    # -- rerun source filter --

    def test_input_types_has_rerun_source(self):
        inputs = lora_optimizer.LoRACombinationGenerator.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("rerun_source", req)
        spec = req["rerun_source"]
        self.assertEqual(spec[0], ["shuffle", "original_progress"])
        self.assertEqual(spec[1]["default"], "shuffle")

    def test_filter_shuffled_by_original_progress_keeps_only_completed(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        shuffled = [("a", "b"), ("c", "d"), ("e", "f")]
        ab_hash = gen._combo_hash(("a", "b"))
        ef_hash = gen._combo_hash(("e", "f"))
        filtered = gen._filter_by_original_progress(
            shuffled, original_completed={ab_hash, ef_hash},
        )
        self.assertEqual(filtered, [("a", "b"), ("e", "f")])

    def test_filter_shuffled_by_original_progress_preserves_order(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        shuffled = [("c", "d"), ("a", "b"), ("e", "f")]
        completed = {gen._combo_hash(c) for c in shuffled}
        filtered = gen._filter_by_original_progress(shuffled, completed)
        self.assertEqual(filtered, shuffled)

    def test_filter_shuffled_by_original_progress_empty_original(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        shuffled = [("a", "b"), ("c", "d")]
        filtered = gen._filter_by_original_progress(shuffled, original_completed=set())
        self.assertEqual(filtered, [])
```

**Step 2: Run the tests to verify they fail**

Run:
```
pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v -k "rerun_source or filter_shuffled"
```

Expected: 4 failures — `rerun_source` key absent and `_filter_by_original_progress` method absent.

**Step 3: Implement the minimal changes**

In `lora_optimizer.py`, modify `LoRACombinationGenerator.INPUT_TYPES` — add after `rerun_mode`:

```python
                "rerun_source": (["shuffle", "original_progress"], {
                    "default": "shuffle",
                    "tooltip": "Only meaningful when rerun_mode=True. "
                               "'shuffle' iterates all combos in shuffle order. "
                               "'original_progress' restricts iteration to combos "
                               "present in the original combo_progress.json (replay "
                               "exactly what was previously processed)."}),
```

Update `IS_CHANGED` signature:

```python
    @classmethod
    def IS_CHANGED(cls, shuffle_order, strength, combo_size,
                   folder_filter="", rerun_mode=False, rerun_source="shuffle"):
        return float("nan")
```

Add a helper method on the class (static — no instance state needed):

```python
    @staticmethod
    def _filter_by_original_progress(shuffled, original_completed):
        """Return *shuffled* with order preserved, keeping only combos whose
        hash is in *original_completed*."""
        return [c for c in shuffled
                if LoRACombinationGenerator._combo_hash(c) in original_completed]
```

Update `get_next_combo` signature and wire the filter between `_shuffle_combos` and `_find_next`:

```python
    def get_next_combo(self, shuffle_order, strength, combo_size,
                        folder_filter="", rerun_mode=False,
                        rerun_source="shuffle"):
        lora_names = folder_paths.get_filename_list("loras")
        if not folder_filter:
            raise ValueError("folder_filter is required — specify one or more "
                             "comma-separated prefixes (e.g. 'zit/,zib/').")
        prefixes = tuple(p.strip() for p in folder_filter.split(",") if p.strip())
        lora_names = [n for n in lora_names if n.startswith(prefixes)]
        if len(lora_names) < 2:
            raise ValueError(f"Need at least 2 LoRAs matching filter '{folder_filter}', "
                             f"found {len(lora_names)}: {lora_names}")

        progress_path = self._resolve_progress_path(rerun_mode)
        combos = self._generate_combos(lora_names, combo_size)
        shuffled = self._shuffle_combos(combos, shuffle_order)

        if rerun_mode and rerun_source == "original_progress":
            original_completed, _ = self._load_progress(self._progress_path)
            pre_filter_count = len(shuffled)
            shuffled = self._filter_by_original_progress(
                shuffled, original_completed)
            logging.info("[LoRA Combo] rerun_source=original_progress — "
                         "filtered %d combos down to %d (present in %s).",
                         pre_filter_count, len(shuffled), self._progress_path)

        completed, _ = self._load_progress(progress_path)
        total = len(shuffled)
        logging.info("[LoRA Combo] shuffle_order=%d, pool=%d LoRAs, %d combos, "
                     "%d already completed, rerun_mode=%s, rerun_source=%s, "
                     "progress file: %s",
                     shuffle_order, len(lora_names), total, len(completed),
                     rerun_mode, rerun_source, progress_path)
        # ... rest of method unchanged
```

**Step 4: Run the tests to verify they pass**

```
pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v
pytest tests/test_lora_optimizer.py -v
```

Expected: all green. `rerun_source` defaults to `"shuffle"` so every pre-existing behavior (and test) is untouched.

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat(combinator): add rerun_source=original_progress filter

When rerun_mode=True and rerun_source='original_progress', restrict the
shuffled combo list to hashes present in the original combo_progress.json
before iteration. Lets the backfill rerun replay exactly the combos that
were processed originally, ignoring any LoRAs added since and any combos
that were never run."
```

---

## Out of scope

- Skipping the filter when the original progress file is missing — if the file is absent, `_load_progress` returns `set()` and the filter yields an empty list (no combos). This is deliberate: if there's no original progress, `original_progress` mode has nothing to replay.
- Cross-check with the rerun progress file. The HF-enrichment skip already handles "was this combo enriched in a prior rerun session", so we don't need to layer another filter.
