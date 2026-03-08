# Configuration Guide

Every parameter on the LoRA Optimizer, organized by what it controls. Each section explains what the parameter does, when to change it, and how it interacts with other settings.

---

## Strength Controls

### `output_strength` (0.0 – 10.0, default 1.0)

Master volume for the entire merge. Scales all merged patches uniformly before applying to the model.

- Start at 1.0 and adjust if the merge is too strong/weak
- The analysis report includes a **suggested max output strength** — going above this risks oversaturation
- Does not affect the merge itself — only the final application strength

### `clip_strength_multiplier` (0.0 – 10.0, default 1.0)

Scales CLIP (text encoder) patches relative to the model patches.

- At 1.0, CLIP patches use the same effective strength as model patches
- Lower values reduce text encoder influence while keeping model influence full
- Only relevant when CLIP is connected

---

## Merge Strategy

### `optimization_mode`

Controls whether the optimizer picks strategies per-prefix or globally.

| Value | Behavior | When to Use |
|-------|----------|-------------|
| `per_prefix` (default) | Each weight group gets its own strategy | Almost always — this is the main feature |
| `global` | Single strategy for all prefixes | When you want consistent behavior across all layers |
| `additive` | Force weighted sum everywhere | For distillation/edit LoRAs, or when combined with SVD compression for zero-loss output |

### `strategy_set`

Controls which merge strategies are available during auto-selection. This is about _strategy logic_ — which algorithms can be picked.

| Value | Auto-selectable Strategies | When to Use |
|-------|---------------------------|-------------|
| `full` (default) | TIES, weighted_average, SLERP, consensus | Recommended — full algorithm repertoire |
| `no_slerp` | TIES, weighted_average, consensus | If SLERP produces undesirable results for your LoRAs |
| `basic` | TIES, weighted_average only | Reproduce pre-1.2 behavior |

### `architecture_preset`

Tunes numeric thresholds (density ranges, noise floors, strength caps) to match the model family. This is about _numbers_ — independent of which strategies are available.

| Value | Models | Effect |
|-------|--------|--------|
| `auto` (default) | Auto-detected | Picks the right preset from LoRA key patterns |
| `sd_unet` | SD 1.5, SDXL | Lower density floor (0.1), 10% noise floor, max strength 3.0 |
| `dit` | Flux, Wan, Z-Image, LTX, HunyuanVideo | Higher density floor (0.4), 5% noise floor, max strength 5.0 |
| `llm` | Qwen, LLaMA | Tight density range (0.1–0.8), 15% noise floor, max strength 3.0 |

Generally leave this on `auto`. Override only if auto-detection picks the wrong family.

### `merge_strategy_override` (optional input)

Forces a specific merge strategy, bypassing all auto-selection. Connect the `merge_strategy` output from the **Conflict Editor** node, or type one of: `auto`, `ties`, `consensus`, `slerp`, `weighted_average`, `weighted_sum`.

---

## Auto-Strength

### `auto_strength` (enabled / disabled)

When enabled, reduces all LoRA strengths proportionally to prevent oversaturation. Uses interference-aware energy normalization — accounts for the actual directional alignment between LoRAs, not just their magnitudes.

| Scenario | Effect |
|----------|--------|
| Enabled, 2 aligned LoRAs | ~50% strength each (they reinforce) |
| Enabled, 2 orthogonal LoRAs | ~71% strength each (independent) |
| Enabled, 2 opposing LoRAs | ~100% each (they cancel) |
| Disabled | Strengths used as-is |

**Recommendation:** Enable when stacking 2+ LoRAs at full strength, especially on distilled/turbo models. The Simple optimizer variant enables this by default.

---

## Sparsification

### `sparsification`

Zeroes out a fraction of each LoRA's weights before merging to reduce interference.

| Value | Method | Best For |
|-------|--------|----------|
| `disabled` (default) | No sparsification | When LoRAs don't interfere much |
| `dare` | Random mask, rescale survivors | General purpose denoising |
| `della` | Magnitude-aware mask (keeps important weights) | More surgical than DARE |
| `dare_conflict` | DARE only at positions where LoRAs disagree | Preserve unique features while reducing interference |
| `della_conflict` | DELLA only at conflict positions | Best of both: surgical + targeted |

**Interaction with TIES:** When TIES is the merge strategy, sparsification _replaces_ the TIES trim step. When using other strategies, sparsification runs as preprocessing.

### `sparsification_density` (0.01 – 1.0, default 0.7)

Fraction of parameters to keep. Lower = more aggressive sparsification.

- 0.9 → keep 90%, drop 10% (gentle)
- 0.7 → keep 70%, drop 30% (moderate, default)
- 0.5 → keep 50%, drop 50% (aggressive)

### `dare_dampening` (0.0 – 1.0, default 0.0)

DAREx-q enhancement for DARE modes only. Interpolates the rescaling factor toward 1.0.

- 0.0 → standard DARE rescaling (`1/density`)
- Higher → less noise amplification, slightly biased magnitudes
- Only affects `dare` and `dare_conflict` modes

---

## Merge Quality

### `merge_refinement`

Controls additional processing techniques applied during merging. Higher quality = better conflict resolution at slight compute cost.

| Level | Techniques Added | Cost |
|-------|-----------------|------|
| `none` (default) | Element-wise sign voting only | Baseline |
| `refine` | + DO-orthogonalization, column-wise voting, TALL-mask protection | Minimal extra compute, no extra VRAM |
| `full` | + KnOTS SVD alignment (on top of refine) | More VRAM for SVD decomposition |

**Recommendations:**
- Start with `none` — it's often sufficient
- Try `refine` if you see artifacts or loss of individual LoRA character
- Use `full` for critical merges where quality matters most
- Best combined with conflict-aware sparsification (`dare_conflict` or `della_conflict`)

---

## Compression & Memory

### `patch_compression`

Re-compresses merged full-rank patches to low-rank via SVD, reducing RAM by ~32x.

| Value | Compresses | Quality Loss |
|-------|-----------|-------------|
| `non_ties` (default) | weighted_sum and weighted_average prefixes | None (linear ops are exactly representable) |
| `aggressive` | All prefixes including TIES | Lossy on TIES prefixes (nonlinear ops produce full-rank) |
| `disabled` | Nothing | None, but ~32x more RAM |

### `svd_device` (gpu / cpu)

Device for SVD compression. GPU is 10–50x faster. Use CPU only if GPU memory is tight.

### `cache_patches` (enabled / disabled)

When enabled, keeps merged patches in RAM between executions. Same LoRA stack + same settings = instant re-execution. Disable for video models or when RAM is constrained.

### `free_vram_between_passes` (enabled / disabled)

Releases GPU cache between Pass 1 (analysis) and Pass 2 (merge). Negligible speed cost, slight VRAM reduction.

### `vram_budget` (0.0 – 1.0, default 0.0)

Fraction of free VRAM to use for keeping merged patches on GPU.

- 0.0 → all patches on CPU (default, safest)
- 0.5 → use up to 50% of available free VRAM for patches
- 1.0 → use all available free VRAM

Higher values = faster sampling (less CPU→GPU transfer) but more GPU memory used. Available on both the Optimizer and AutoTuner.

---

## Key Normalization

### `normalize_keys` (enabled / disabled)

Auto-detects model architecture and remaps LoRA keys to a canonical format, enabling correct merge across different trainers.

| When to Enable | When to Leave Disabled |
|---------------|----------------------|
| Mixing LoRAs from different trainers | All LoRAs from the same trainer |
| Using Z-Image LoRAs with fused QKV | Single-trainer workflows |
| WanVideo LoRAs (enabled by default on WanVideo Optimizer) | When auto-detection picks the wrong architecture |

Supported architectures: FLUX, SDXL, Z-Image (Lumina2), Wan 2.1/2.2, LTX Video, Qwen-Image.

---

## AutoTuner-Specific

### `top_n` (1 – 10, default 3)

Number of candidate configurations to actually merge and score in Phase 2. Higher = more thorough but slower.

- 3 → good balance of speed and coverage
- 5–10 → more alternatives to choose from via Merge Selector
- 1 → fastest, but no alternatives

### `scoring_svd` (enabled / disabled)

When enabled, computes SVD-based effective rank as part of the quality score. More thorough but slower.

### `scoring_device` (cpu / gpu)

Device for SVD scoring computations. GPU is 10–50x faster than CPU.

### `record_dataset` (enabled / disabled)

Saves analysis metrics and scoring results to `lora_optimizer_reports/autotuner_dataset.jsonl`. Used for threshold tuning research — not needed for normal use.

---

## Recommended Configurations

### Simple 2-LoRA Merge (Quick)

Use the **LoRA Optimizer** (Simple) node. Everything is handled automatically.

### High-Quality Multi-LoRA Merge

| Parameter | Value |
|-----------|-------|
| `auto_strength` | enabled |
| `optimization_mode` | per_prefix |
| `merge_refinement` | refine |
| `sparsification` | della_conflict |
| `sparsification_density` | 0.7 |
| `patch_compression` | non_ties |

### Maximum Quality (Critical Merge)

| Parameter | Value |
|-----------|-------|
| `auto_strength` | enabled |
| `optimization_mode` | per_prefix |
| `merge_refinement` | full |
| `sparsification` | della_conflict |
| `sparsification_density` | 0.7 |
| `patch_compression` | non_ties |

### Video Models (Low Memory)

| Parameter | Value |
|-----------|-------|
| `cache_patches` | disabled |
| `patch_compression` | aggressive |
| `free_vram_between_passes` | enabled |
| `normalize_keys` | enabled |
| `vram_budget` | 0.0 |

### Let the AutoTuner Decide

Use the **LoRA AutoTuner** node with `top_n=5`. Review the report, then use **Merge Selector** to try alternatives.
