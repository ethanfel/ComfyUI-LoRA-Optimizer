# LoRA Combination Generator — Design

## Goal

A ComfyUI node that exhaustively generates all 2-way and 3-way LoRA combinations
from the available LoRA pool, outputting one combo per execution as a `LORA_STACK`
for the AutoTuner. Tracks completed combos to avoid duplicates across runs.

## Node: LoRACombinationGenerator

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | INT | 0 | Seed for shuffle order. Same seed = same deterministic sequence. |
| `strength` | FLOAT | 1.0 | Default strength for all LoRAs in the combo. |
| `combo_size` | `["2", "3", "2_and_3"]` | `"2_and_3"` | Generate pairs, triples, or both. |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `lora_stack` | LORA_STACK | The next combination to merge. |
| `combo_info` | STRING | Which LoRAs, combo index, remaining count. |

### Behavior

1. Scan all LoRAs via `folder_paths.get_filename_list("loras")`.
2. Generate all combinations (itertools.combinations, size 2 and/or 3).
3. Shuffle deterministically using `seed` (random.Random(seed).shuffle).
4. Load completed set from tracking file.
5. Walk the shuffled list, skip any combo whose hash is in the completed set.
6. Take the first non-completed combo, load the LoRAs, output as `LORA_STACK`.
7. After AutoTuner processes the result, add the combo hash to completed set.
8. When all combos are completed, interrupt execution to stop the queue.

### Tracking file

Stored in the plugin directory as `combo_progress.json`. Only stores completed
hashes — the queue is regenerated deterministically from seed + LoRA list each run.

```json
{
  "seed_12345": {
    "completed": ["a1b2c3", "d4e5f6"],
    "total_generated": 500
  }
}
```

Combo hash = hash of sorted LoRA names, so the same pair is never repeated
even across different seeds.

### Edge cases

- **New LoRAs added**: Won't appear in existing seed's sequence. Use a new seed
  or delete the seed entry to regenerate.
- **LoRA deleted**: Skip that combo (file not found), log warning, move to next.
- **Completion**: Raise `ExecutionBlockedError` or return empty to stop the queue.

### Registration

```python
NODE_CLASS_MAPPINGS["LoRACombinationGenerator"] = LoRACombinationGenerator
NODE_DISPLAY_NAME_MAPPINGS["LoRACombinationGenerator"] = "LoRA Combination Generator"
```

Category: `ZImage LoRA Merger`

## Not in scope

- Architecture filtering (LoRAs are already architecture-specific in practice).
- Custom LoRA subsets (user can organize LoRAs into subfolders upstream).
- AutoTuner result collection (handled by existing AutoTuner output/save nodes).
