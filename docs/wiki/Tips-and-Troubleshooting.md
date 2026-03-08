# Tips and Troubleshooting

Best practices, common pitfalls, and solutions for working with the LoRA Optimizer.

---

## What to Merge (and What NOT to Merge)

### Good Candidates for the Optimizer

- **Style LoRAs** — art styles, aesthetic LoRAs, color grading
- **Character LoRAs** — character appearance, costume, features
- **Concept LoRAs** — objects, poses, compositions
- **Detail LoRAs** — texture, sharpness, quality enhancement

These LoRAs modify the model's creative outputs and benefit from intelligent conflict resolution.

### Do NOT Put These in the Optimizer Stack

| LoRA Type | Why Not | What to Do Instead |
|-----------|---------|-------------------|
| **Distillation** (LCM, Lightning, Turbo, Hyper) | Precisely calibrated for step reduction; merging breaks the calibration | Apply via standard **Load LoRA** node upstream |
| **DPO LoRAs** | Trained for preference alignment; weights are calibrated as a unit | Apply via **Load LoRA** upstream |
| **Edit/Instruction LoRAs** (Qwen edit, Klein edit) | Modify fundamental model behavior, not style | Apply via **Load LoRA** upstream |
| **ControlNet LoRAs** | Separate control mechanism, not a style merge | Use ControlNet nodes |

If you must include an edit LoRA in the stack, use `weighted_sum_only` optimization mode and disable sparsification to avoid weight trimming.

---

## Understanding the Analysis Report

### Block Strategy Map

```
--- Block Strategy Map ---
  input_blocks.0   ====  sum  1 LoRA (6x)
  input_blocks.4   ----  avg  12% conflict (6x)
  middle_block.1   ####  TIES 42% conflict (6x)
  output_blocks.3  ----  avg  8% conflict (6x)
  output_blocks.8  ====  sum  1 LoRA (6x)
  Legend: ==== sum (single LoRA)  ---- avg (compatible)  #### TIES (conflict)
```

- `====` (sum) — Only one LoRA touches these layers. Full strength, no dilution.
- `----` (avg) — Multiple LoRAs, but they mostly agree. Fair blending.
- `####` (TIES) — Significant sign conflicts. TIES resolves them via trim/elect/merge.

A healthy merge shows a mix of all three. If everything is `####`, your LoRAs are very different — consider whether they should be merged at all.

### Suggested Max Output Strength

The report shows a suggested maximum for `output_strength`. Going above this value may cause oversaturation. The ceiling is calculated from the magnitude ratio of the merged LoRAs and capped per architecture:
- UNet models: max 3.0
- DiT models: max 5.0
- LLM models: max 3.0

### Auto-Strength Adjustments

When `auto_strength=enabled`, the report shows original vs adjusted strengths:
```
--- Auto-Strength Adjustment ---
  style_lora: 1.0 -> 0.63
  detail_lora: 0.8 -> 0.51
  Scale factor: 0.63
  Avg pairwise cosine similarity: 0.312
```

The scale factor tells you how much the optimizer reduced strengths. A low factor means the LoRAs reinforce each other heavily.

---

## Memory Optimization

### For Large Models (Video, High-Res)

| Setting | Recommended Value | Why |
|---------|------------------|-----|
| `cache_patches` | disabled | Don't keep merge in RAM between runs |
| `compress_patches` | all | Compress everything, even TIES (slight quality loss) |
| `free_vram_between_passes` | enabled | Release GPU cache between analysis and merge |
| `vram_budget` | 0.0 | Keep all patches on CPU |

### For Iterative Workflows (Quick Experimentation)

| Setting | Recommended Value | Why |
|---------|------------------|-----|
| `cache_patches` | enabled | Instant re-execution with same settings |
| `compress_patches` | non_ties | Exact low-rank path on linear ops; only nonlinear paths use SVD compression |
| `vram_budget` | 0.3–0.5 | Keep some patches on GPU for faster sampling |

### Peak VRAM Usage

The optimizer's two-pass architecture usually keeps peak memory near one active target group at a time, but the exact peak still depends on layer size, overlap count, and enabled quality/compression steps. The merged result can still dominate memory. Use `compress_patches` and `vram_budget` to control where the result lives.

---

## Common Issues

### "My merge looks worse than simple stacking"

1. **Check auto-strength** — if enabled, it may be reducing strengths too aggressively for your use case. Try disabling it.
2. **Check the strategy map** — if most prefixes use TIES, the LoRAs may conflict too much for a clean merge. Consider using fewer LoRAs or adjusting strengths.
3. **Try enhanced quality** — `merge_quality=enhanced` adds DO-orthogonalization and TALL-mask protection.
4. **Try the AutoTuner** — let it rank parameter combinations by merge metrics or your external evaluator.

### "Results are too subtle / LoRA effect is weak"

1. **Increase `output_strength`** — but stay below the suggested max
2. **Disable auto-strength** — if it's reducing strengths too much
3. **Check key overlap** — if LoRAs don't share keys, there's nothing to merge and the optimizer passes them through at full strength already
4. **Check key filter** — `shared_only` or `unique_only` may be filtering out keys you want

### "Results are oversaturated / blown out"

1. **Enable auto-strength** — prevents compounding from stacking
2. **Lower `output_strength`** — check the suggested max in the report
3. **Lower individual LoRA strengths** — especially for LoRAs trained at high learning rates
4. **Use `optimization_mode=per_prefix`** — ensures non-conflicting regions aren't over-merged

### "Key normalization picks the wrong architecture"

This can happen if LoRA keys are ambiguous. Set `architecture_preset` manually to the correct value (`sd_unet`, `dit`, or `llm`) to override auto-detection.

### "LoRAs from different trainers don't merge"

Enable `normalize_keys` — this remaps keys from different trainers to a canonical format. Without it, the optimizer sees no key overlap between differently-named weights.

### "AutoTuner runs out of memory"

- Lower `top_n` (fewer candidates to merge)
- Disable `scoring_svd` (skip SVD-based effective rank scoring)
- Set `compress_patches=all`
- Set `vram_budget=0.0`

### "Merge Selector says stack has changed"

The Merge Selector validates that the LoRA stack hasn't changed since the AutoTuner ran (via hash comparison). If you modified the stack (changed LoRAs, strengths, or conflict modes), re-run the AutoTuner.

---

## Performance Tips

### Speed

- `svd_device=gpu` is 10–50x faster than CPU for SVD compression
- `cache_patches=enabled` gives instant re-execution when only changing `output_strength`
- The AutoTuner caches Pass 1 analysis — only Phase 2 merges take time
- `scoring_device=gpu` speeds up AutoTuner quality scoring

### Quality

- `merge_quality=enhanced` with `della_conflict` sparsification is a strong default for quality
- `merge_quality=maximum` adds KnOTS-inspired SVD alignment — strongest quality mode for critical merges
- `behavior_profile=v1.2` gives the optimizer the full algorithm repertoire
- Use the Conflict Editor to understand pairwise conflicts before choosing settings

---

## Limitations

1. **The optimizer only sees its own stack.** LoRAs applied via upstream Load LoRA nodes are invisible — they stack additively on top. This can cause overexposure if you have upstream LoRAs + optimizer LoRAs touching the same weights.

2. **Fully baked checkpoints are invisible.** If a checkpoint already has LoRA weights baked in, the optimizer can't detect or account for them.

3. **TIES produces full-rank results.** The nonlinear operations (trim + sign election) mean TIES output can't be perfectly captured by low-rank SVD compression. Use `compress_patches=non_ties` to only compress linear operations losslessly.

4. **Conflict-aware sparsification needs overlap.** If LoRAs don't share any keys, there are no conflicts to detect and conflict-aware variants behave the same as disabled.
