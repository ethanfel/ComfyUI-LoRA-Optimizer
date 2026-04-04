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
| `smart` (default) | weighted_sum and weighted_average prefixes | None (linear ops are exactly representable) |
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

## Decision Smoothing

### `decision_smoothing` (0.0 – 1.0, default 0.25)

Blends each group's decision metrics toward the average of its surrounding block. Reduces jagged layer-to-layer strategy flips when the stack is noisy.

- 0.0 → no smoothing (each prefix decides independently)
- 0.25 → gentle smoothing (default — reduces noise while preserving real transitions)
- 1.0 → full smoothing (all prefixes in a block use the same strategy)

### `smooth_slerp_gate` (true / false, default false)

When enabled, uses per-prefix cosine similarity (computed during Pass 1 analysis) for the SLERP interpolation gate instead of the collection-wide average. This makes the SLERP weight vary per layer based on local alignment.

- `false` → single global SLERP weight from the average cosine similarity across all prefixes
- `true` → per-prefix SLERP weight based on that prefix's local cosine similarity

Enable when LoRAs have varying alignment across different model regions — some layers aligned, others orthogonal.

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

### `scoring_speed` (full / fast / turbo / turbo+, default turbo)

Controls prefix subsampling during the Phase 1 heuristic sweep. Higher speed = fewer prefixes scored per candidate.

| Value | Behavior | Speed |
|-------|----------|-------|
| `full` | Score all prefixes | Baseline |
| `fast` | Score every 2nd prefix | ~2x faster |
| `turbo` (default) | Score every 3rd prefix, biased toward high-conflict prefixes | ~3x faster |
| `turbo+` | Most aggressive subsampling | ~4x faster |

Ranking stays fair because all candidates are scored on the same subset. High-conflict prefixes are prioritized since they contribute most to quality differences.

### `scoring_formula` (v2 / v1, default v2)

Scoring formula version for ranking candidates. `v2` is the current default with improved composite scoring. `v1` is the original formula, kept for comparison.

### `output_mode` (merge / tuning_only, default merge)

Controls what the AutoTuner outputs after the sweep.

| Value | Behavior | When to Use |
|-------|----------|-------------|
| `merge` (default) | Applies the top-ranked config and outputs the merged model | Normal use — the AutoTuner both ranks and applies |
| `tuning_only` | Skips the final merge, passes the base model through | Connect AutoTuner → LoRA Optimizer (Legacy) to apply the winning config via a separate optimizer node |

`tuning_only` is useful when you want the AutoTuner to decide the settings but apply the merge through a dedicated Optimizer node with additional controls (e.g., a custom `merge_strategy_override`).

### `memory_mode` (disabled / auto / read_only / clear_and_run, default auto)

Persistent cache for tuning results. When a matching entry exists on disk, the full Phase 1 + Phase 2 sweep is skipped and the cached rankings are replayed in a single merge (~2–5 seconds instead of 30–120+).

| Value | Behavior |
|-------|----------|
| `disabled` | No caching — always run the full sweep |
| `auto` (default) | Load from cache if available; save new results after a sweep |
| `read_only` | Use cached results but never write new ones |
| `clear_and_run` | Delete the cached entry and re-run from scratch, then save |

The cache key is the LoRA set (names + strengths) + tuning settings. Changing any LoRA strength or tuning parameter causes a cache miss. Cache files live in `models/autotuner_memory/` as `{lora_hash}_{settings_hash}.memory.json`.

Use `clear_and_run` after updating a LoRA file in place (same filename, different content), or when you want fresh results after tweaking settings.

### `selection` (1–10, default 1)

When a memory hit occurs, `selection` controls which ranked config is replayed. `1` = the original top-ranked config. `2` = second-ranked, and so on. Requires the cached entry to have at least `selection` configs stored (determined by `top_n` at the time of the original run).

This lets you explore alternatives from a cached run without re-running the full sweep: set `memory_mode=auto` and increase `selection` to try the 2nd or 3rd ranked config from the last run.

### `community_cache` (disabled / download_only / upload_and_download)

Community-backed cache hosted on Hugging Face. Precomputed analysis results (per-LoRA conflict stats, pairwise metrics, winning merge configs) are hardware-agnostic — the same LoRA files always produce the same output regardless of GPU tier.

| Value | Behavior |
|-------|----------|
| `disabled` (default) | No community interaction — all computation is local |
| `download_only` | Download precomputed results anonymously. If a matching config exists, skips the AutoTuner sweep entirely |
| `upload_and_download` | Also upload your results after a sweep. Requires `HF_TOKEN` env var and `huggingface_hub` installed |

**Privacy:** LoRA filenames are never shared. Only content hashes (SHA256[:16]) are used as keys. Results include conflict metrics and winning configs — no paths, no names.

**Fallback:** Network errors are logged as warnings and silently ignored — the full local sweep runs as normal if download fails.

---

## AutoTuner Caching Infrastructure

The AutoTuner uses three disk-based caches to avoid redundant computation. All cache files live in `models/autotuner_memory/`.

### Persistent Memory (`.memory.json`)

Stores complete tuning results (rankings, configs, scores) keyed by LoRA set + settings. See `memory_mode` above. A cache hit skips the entire sweep — typically 30–120× faster than a full run.

### Analysis Cache (`.analysis.json`)

Stores per-prefix pairwise conflict metrics (overlap, cosine similarity, sign conflicts) keyed by the set of LoRA names. Phase 1 analysis is the most GPU-intensive part of a tuning run — this cache makes repeated runs with the same LoRA set nearly instant.

The analysis cache is populated automatically and invalidated when any LoRA in the set changes. You do not need to configure it.

### Analysis Resume (`.analysis.partial.json`)

If a tuning run is interrupted mid-analysis (OOM, crash, ComfyUI restart), the partial results are saved as a checkpoint. On the next run with the same LoRA set, analysis resumes from where it left off instead of starting over.

The partial file is deleted automatically when analysis completes successfully.

### Clearing Caches

To force a full re-run from scratch:
- **Persistent memory only:** Set `memory_mode=clear_and_run`
- **Analysis cache:** Delete `{names_hash}.analysis.json` from `models/autotuner_memory/`
- **All caches:** Delete all files from `models/autotuner_memory/`

---

## Settings Nodes

The settings architecture uses a 3-tier cascade:

```
LoRA Merge Settings → LoRA Optimizer Settings → LoRA Optimizer (settings input)
                    → LoRA AutoTuner Settings → LoRA AutoTuner
```

### Priority Cascade

1. **Connected settings node** — highest priority. Values from the settings node override built-in defaults.
2. **Built-in defaults** — used when no settings node is connected. The LoRA Optimizer uses sensible defaults (auto_strength=enabled, optimization_mode=per_prefix, etc.).

### When to Use Settings Nodes

- **Never required** — the optimizer works out of the box with sensible defaults
- **Use LoRA Merge Settings** when you want to change shared parameters (architecture preset, smoothing, VRAM budget) that apply to both optimizer and tuner workflows
- **Use LoRA Optimizer Settings** when you want fine-grained control over merge parameters (sparsification, compression, refinement) without cluttering the main optimizer node
- **Use LoRA AutoTuner Settings** when configuring the tuner separately (top_n, scoring options, diff cache)

### Connecting Settings

LoRA Merge Settings outputs `MERGE_SETTINGS`, which can be connected to either LoRA Optimizer Settings or LoRA AutoTuner Settings via their `merge_settings` input. Those settings nodes output `OPTIMIZER_SETTINGS`, which connects to the optimizer's or tuner's `settings` input.

You can also use Optimizer Settings or AutoTuner Settings alone — they have sensible defaults for the shared parameters when no Merge Settings node is connected.

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
| `patch_compression` | smart |

### Maximum Quality (Critical Merge)

| Parameter | Value |
|-----------|-------|
| `auto_strength` | enabled |
| `optimization_mode` | per_prefix |
| `merge_refinement` | full |
| `sparsification` | della_conflict |
| `sparsification_density` | 0.7 |
| `patch_compression` | smart |

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
