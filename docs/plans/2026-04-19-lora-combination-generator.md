# LoRA Combination Generator — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A ComfyUI node that generates all 2/3-way LoRA combinations in shuffled order, outputting one per execution as a LORA_STACK, with persistent tracking to avoid duplicates.

**Architecture:** Single node class `LoRACombinationGenerator` added to `lora_optimizer.py`. Combo generation uses `itertools.combinations` + `random.Random(seed).shuffle`. Progress tracked in a JSON file (`combo_progress.json`) in the plugin directory, storing only completed combo hashes — the full queue is regenerated deterministically each run.

**Tech Stack:** Python stdlib (itertools, random, hashlib, json), ComfyUI node API (folder_paths, comfy.utils)

---

### Task 1: Core combo generation + tracking logic (tests)

**Files:**
- Create tests in: `tests/test_lora_optimizer.py` (append to existing)

**Step 1: Write failing tests for combo generation and tracking**

Add a new test class at the end of `tests/test_lora_optimizer.py`:

```python
class TestLoRACombinationGenerator(unittest.TestCase):
    """Tests for LoRACombinationGenerator node."""

    def test_generates_all_pairs_from_lora_list(self):
        """4 LoRAs should produce C(4,2) = 6 pairs."""
        loras = ["a.safetensors", "b.safetensors", "c.safetensors", "d.safetensors"]
        combos = LoRACombinationGenerator._generate_combos(loras, "2")
        self.assertEqual(len(combos), 6)
        # Each combo is a tuple of sorted names
        self.assertIn(("a.safetensors", "b.safetensors"), combos)

    def test_generates_all_triples_from_lora_list(self):
        """4 LoRAs should produce C(4,3) = 4 triples."""
        loras = ["a.safetensors", "b.safetensors", "c.safetensors", "d.safetensors"]
        combos = LoRACombinationGenerator._generate_combos(loras, "3")
        self.assertEqual(len(combos), 4)

    def test_generates_both_pairs_and_triples(self):
        """4 LoRAs with '2_and_3' should produce 6 + 4 = 10 combos."""
        loras = ["a.safetensors", "b.safetensors", "c.safetensors", "d.safetensors"]
        combos = LoRACombinationGenerator._generate_combos(loras, "2_and_3")
        self.assertEqual(len(combos), 10)

    def test_shuffle_is_deterministic_with_seed(self):
        """Same seed produces same order."""
        loras = ["a.safetensors", "b.safetensors", "c.safetensors", "d.safetensors"]
        combos1 = LoRACombinationGenerator._generate_combos(loras, "2_and_3")
        combos2 = LoRACombinationGenerator._generate_combos(loras, "2_and_3")
        order1 = LoRACombinationGenerator._shuffle_combos(combos1, seed=42)
        order2 = LoRACombinationGenerator._shuffle_combos(combos2, seed=42)
        self.assertEqual(order1, order2)

    def test_different_seeds_produce_different_order(self):
        """Different seeds produce different order."""
        loras = ["a.safetensors", "b.safetensors", "c.safetensors", "d.safetensors"]
        combos1 = LoRACombinationGenerator._generate_combos(loras, "2_and_3")
        combos2 = LoRACombinationGenerator._generate_combos(loras, "2_and_3")
        order1 = LoRACombinationGenerator._shuffle_combos(combos1, seed=42)
        order2 = LoRACombinationGenerator._shuffle_combos(combos2, seed=99)
        self.assertNotEqual(order1, order2)

    def test_combo_hash_is_order_independent(self):
        """Hash of (a, b) should equal hash of (b, a)."""
        h1 = LoRACombinationGenerator._combo_hash(("a.safetensors", "b.safetensors"))
        h2 = LoRACombinationGenerator._combo_hash(("b.safetensors", "a.safetensors"))
        self.assertEqual(h1, h2)

    def test_find_next_skips_completed(self):
        """Should skip combos whose hash is in the completed set."""
        loras = ["a.safetensors", "b.safetensors", "c.safetensors"]
        combos = LoRACombinationGenerator._generate_combos(loras, "2")
        shuffled = LoRACombinationGenerator._shuffle_combos(combos, seed=0)
        first = shuffled[0]
        first_hash = LoRACombinationGenerator._combo_hash(first)
        completed = {first_hash}
        result = LoRACombinationGenerator._find_next(shuffled, completed)
        self.assertIsNotNone(result)
        self.assertNotEqual(result, first)

    def test_find_next_returns_none_when_all_done(self):
        """Should return None when all combos are completed."""
        loras = ["a.safetensors", "b.safetensors"]
        combos = LoRACombinationGenerator._generate_combos(loras, "2")
        shuffled = LoRACombinationGenerator._shuffle_combos(combos, seed=0)
        completed = {LoRACombinationGenerator._combo_hash(c) for c in shuffled}
        result = LoRACombinationGenerator._find_next(shuffled, completed)
        self.assertIsNone(result)

    def test_progress_save_and_load(self):
        """Progress file round-trips correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "combo_progress.json")
            LoRACombinationGenerator._save_progress(path, 42, {"abc", "def"}, 10)
            completed, total = LoRACombinationGenerator._load_progress(path, 42)
            self.assertEqual(completed, {"abc", "def"})
            self.assertEqual(total, 10)

    def test_progress_load_missing_file_returns_empty(self):
        """Missing file returns empty set."""
        completed, total = LoRACombinationGenerator._load_progress("/nonexistent/path.json", 42)
        self.assertEqual(completed, set())
        self.assertEqual(total, 0)

    def test_progress_load_different_seed_returns_empty(self):
        """Loading a different seed than saved returns empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "combo_progress.json")
            LoRACombinationGenerator._save_progress(path, 42, {"abc"}, 10)
            completed, total = LoRACombinationGenerator._load_progress(path, 99)
            self.assertEqual(completed, set())
```

**Step 2: Run tests to verify they fail**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v`
Expected: FAIL — `LoRACombinationGenerator` not defined

---

### Task 2: Implement core combo logic (static methods)

**Files:**
- Modify: `lora_optimizer.py` (add class before `NODE_CLASS_MAPPINGS` at line ~12033)

**Step 1: Implement LoRACombinationGenerator static methods**

Add before `NODE_CLASS_MAPPINGS` in `lora_optimizer.py`:

```python
class LoRACombinationGenerator:
    """
    Generates all 2-way and/or 3-way LoRA combinations in a deterministic
    shuffled order. Outputs one combo per execution as a LORA_STACK.
    Tracks completed combos in a JSON file to avoid duplicates across runs.
    """

    @staticmethod
    def _generate_combos(lora_names, combo_size):
        """Generate all combinations of the given size(s) from lora_names."""
        import itertools
        names = sorted(lora_names)
        combos = []
        if combo_size in ("2", "2_and_3"):
            combos.extend(itertools.combinations(names, 2))
        if combo_size in ("3", "2_and_3"):
            combos.extend(itertools.combinations(names, 3))
        return combos

    @staticmethod
    def _shuffle_combos(combos, seed):
        """Shuffle combos deterministically using seed. Returns new list."""
        import random
        rng = random.Random(seed)
        result = list(combos)
        rng.shuffle(result)
        return result

    @staticmethod
    def _combo_hash(combo):
        """Hash a combo tuple by sorted names. Order-independent."""
        import hashlib
        key = "|".join(sorted(combo))
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @staticmethod
    def _find_next(shuffled_combos, completed):
        """Find next combo not in completed set. Returns None if all done."""
        for combo in shuffled_combos:
            if LoRACombinationGenerator._combo_hash(combo) not in completed:
                return combo
        return None

    @staticmethod
    def _load_progress(path, seed):
        """Load completed set for a given seed. Returns (set, total)."""
        import json
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return set(), 0
        key = f"seed_{seed}"
        if key not in data:
            return set(), 0
        entry = data[key]
        return set(entry.get("completed", [])), entry.get("total_generated", 0)

    @staticmethod
    def _save_progress(path, seed, completed, total):
        """Save completed set for a given seed."""
        import json
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        key = f"seed_{seed}"
        data[key] = {
            "completed": sorted(completed),
            "total_generated": total,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

**Step 2: Run tests to verify they pass**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v`
Expected: All 11 tests PASS

**Step 3: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add LoRACombinationGenerator core logic with tests"
```

---

### Task 3: Add ComfyUI node interface

**Files:**
- Modify: `lora_optimizer.py` — add INPUT_TYPES, RETURN_TYPES, main method, IS_CHANGED, and register in NODE_CLASS_MAPPINGS

**Step 1: Write failing test for the node interface**

Add to `TestLoRACombinationGenerator` in `tests/test_lora_optimizer.py`:

```python
    def test_input_types_has_required_fields(self):
        """Node should have seed, strength, combo_size inputs."""
        inputs = LoRACombinationGenerator.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("seed", req)
        self.assertIn("strength", req)
        self.assertIn("combo_size", req)

    def test_return_types(self):
        """Node should return LORA_STACK and STRING."""
        self.assertEqual(LoRACombinationGenerator.RETURN_TYPES, ("LORA_STACK", "STRING"))
```

**Step 2: Run tests to verify they fail**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator::test_input_types_has_required_fields tests/test_lora_optimizer.py::TestLoRACombinationGenerator::test_return_types -v`
Expected: FAIL — INPUT_TYPES / RETURN_TYPES not defined

**Step 3: Add node interface to LoRACombinationGenerator**

Add these class attributes and methods to the existing `LoRACombinationGenerator` class:

```python
    def __init__(self):
        self._progress_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "combo_progress.json"
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                             "tooltip": "Default strength applied to all LoRAs in each combination."}),
                "combo_size": (["2", "3", "2_and_3"], {
                    "default": "2_and_3",
                    "tooltip": "Generate pairs (2), triples (3), or both (2_and_3)."}),
            },
        }

    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("lora_stack", "combo_info")
    FUNCTION = "get_next_combo"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Generates all LoRA combinations for AutoTuner dataset collection. Tracks progress to avoid duplicates."
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, seed, strength, combo_size):
        # Always re-execute to get the next combo
        return float("nan")

    def get_next_combo(self, seed, strength, combo_size):
        lora_names = folder_paths.get_filename_list("loras")
        if len(lora_names) < 2:
            raise ValueError("Need at least 2 LoRAs to generate combinations.")

        combos = self._generate_combos(lora_names, combo_size)
        shuffled = self._shuffle_combos(combos, seed)
        completed, _ = self._load_progress(self._progress_path, seed)
        total = len(shuffled)

        combo = self._find_next(shuffled, completed)
        if combo is None:
            raise InterruptProcessingException(
                f"All {total} combinations completed for seed {seed}."
            )

        # Load LoRAs into stack format
        lora_list = []
        for name in combo:
            lora_path = folder_paths.get_full_path("loras", name)
            if lora_path is None:
                # LoRA was deleted — skip this combo, mark done, recurse
                completed.add(self._combo_hash(combo))
                self._save_progress(self._progress_path, seed, completed, total)
                return self.get_next_combo(seed, strength, combo_size)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            lora_list.append({
                "name": name,
                "lora": lora,
                "strength": strength,
                "conflict_mode": "all",
                "key_filter": "all",
                "metadata": _read_safetensors_metadata(lora_path),
            })

        # Mark completed and save
        completed.add(self._combo_hash(combo))
        self._save_progress(self._progress_path, seed, completed, total)

        done = len(completed)
        info = (f"Combo {done}/{total} | "
                f"{' + '.join(name for name in combo)} | "
                f"Remaining: {total - done}")

        return (lora_list, info)
```

Note: `InterruptProcessingException` — check what ComfyUI uses. If not available, use a simple empty return or raise a generic exception that ComfyUI catches to stop the queue. This will be verified during implementation.

**Step 4: Register the node**

Add to `NODE_CLASS_MAPPINGS` dict:
```python
    "LoRACombinationGenerator": LoRACombinationGenerator,
```

Add to `NODE_DISPLAY_NAME_MAPPINGS` dict:
```python
    "LoRACombinationGenerator": "LoRA Combination Generator",
```

**Step 5: Run tests to verify they pass**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py::TestLoRACombinationGenerator -v`
Expected: All 13 tests PASS

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add LoRACombinationGenerator ComfyUI node interface"
```

---

### Task 4: Verify stop mechanism

**Files:**
- Modify: `lora_optimizer.py` — verify correct exception for stopping ComfyUI queue

**Step 1: Check ComfyUI's interrupt mechanism**

Search the ComfyUI codebase for how nodes signal "stop processing":
- `InterruptProcessingException` from `comfy_execution.exceptions`
- Or `ExecutionBlockedError`
- Or simply returning empty outputs

Grep in ComfyUI source:
```bash
grep -r "InterruptProcessingException\|ExecutionBlockedError" /path/to/ComfyUI/
```

**Step 2: Update the exception if needed**

Replace the exception in `get_next_combo` with whatever ComfyUI actually uses.

**Step 3: Run full test suite**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v`
Expected: All tests PASS (no regressions)

**Step 4: Commit if changed**

```bash
git add lora_optimizer.py
git commit -m "fix: use correct ComfyUI interrupt for combination generator"
```
