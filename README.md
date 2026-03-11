<p align="center">
  <a href="assets/banner.png"><img src="assets/banner.svg" alt="LoRA Optimizer" width="100%"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Custom_Nodes-blue?style=flat-square" alt="ComfyUI">
  <img src="https://img.shields.io/badge/TIES_Merging-NeurIPS_2023-8b5cf6?style=flat-square" alt="TIES">
  <img src="https://img.shields.io/badge/DARE_%7C_DELLA-Sparsification-f59e0b?style=flat-square" alt="DARE/DELLA">
  <img src="https://img.shields.io/badge/Per--Prefix_Adaptive-Merge-e94560?style=flat-square" alt="Per-Prefix">
  <img src="https://img.shields.io/badge/KnOTS_%7C_Column--wise_%7C_TALL--masks-Merge_Refinement-a78bfa?style=flat-square" alt="Merge Refinement">
  <img src="https://img.shields.io/badge/SVD_Patch-Compression-64ffda?style=flat-square" alt="SVD">
  <img src="https://img.shields.io/badge/Architecture--Aware-Key_Normalization-22c55e?style=flat-square" alt="Key Normalization">
  <img src="https://img.shields.io/badge/AutoTuner-Parameter_Sweep-e94560?style=flat-square" alt="AutoTuner">
  <img src="https://img.shields.io/badge/Flux_%7C_SDXL_%7C_Wan_%7C_LTX_%7C_Z--Image_%7C_ACE--Step-Compatible-22c55e?style=flat-square" alt="Compatible">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT">
</p>

---

Stacking multiple LoRAs in ComfyUI often causes oversaturation, artifacts, or lost details. This node suite **automatically figures out the best way to combine your LoRAs** — analyzing where they conflict, resolving those conflicts per model layer, and adjusting strengths so the result looks clean. Just add your LoRAs to a stack, connect the optimizer, and generate.

## The Problem

<p align="center"><img src="assets/the-problem.png" width="720" alt="The Problem with LoRA Stacking vs the Optimizer"></p>

<p align="center"><img src="assets/comparison.png" alt="Before/After Comparison" width="100%"></p>

---

<details>
<summary><b>Should I Merge This LoRA?</b></summary>

<p align="center"><img src="assets/merge-use-cases.png" width="720" alt="Should I merge this LoRA? Decision guide"></p>

</details>

---

### Merge Strategy Guide

How LoRAs relate to each other, what the optimizer does about it, and when to change settings.

<p align="center"><img src="docs/merge-strategy-guide.png" width="720" alt="LoRA Merge Strategy Guide"></p>

---

## Quick Start

1. Add a **LoRA Stack (Dynamic)** node — pick your LoRAs, set strengths
2. Add a **LoRA Optimizer** node — connect `MODEL` (and optionally `CLIP`) from your checkpoint
3. Connect the optimizer's `MODEL`/`CLIP` outputs to your sampler — done

```
LoRA Stack (Dynamic) ──► LoRA Optimizer ──► KSampler
                              ▲
Load Checkpoint ──► MODEL ────┘
```

Everything is automatic. Connect `analysis_report` to a **Show Text** node to see what the optimizer did.

**Want more control?** Add a Settings node → connect to the optimizer's `settings` input.
**Want the best config found for you?** Use the **LoRA AutoTuner** instead of the optimizer.

---

## Installation

### ComfyUI Manager
Search for "LoRA Optimizer" in ComfyUI Manager and install.

<details>
<summary><b>Manual install</b></summary>

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ethanfel/ComfyUI-LoRA-Optimizer.git
```
Restart ComfyUI. Nodes appear under the `loaders` category.

</details>

---

## Nodes at a Glance

| Node | What It Does |
|------|-------------|
| **LoRA Stack** / **(Dynamic)** | Build your list of LoRAs — pick files, set strengths |
| **LoRA Optimizer** | Analyze + merge your stack automatically. Just connect and go |
| **Settings Nodes** | Optional fine-tuning: sparsification, compression, smoothing, etc. |
| **LoRA AutoTuner** | Sweep 2000+ parameter combos, rank the best configs |
| **Merge Selector** | Try alternative ranked configs from AutoTuner results |
| **Compatibility Analyzer** | Check which LoRAs work well together before merging |
| **LoRA Pruner** | Drop blocks, keys, or rank from a single LoRA — drop-in replacement for Load LoRA |
| **Save Merged LoRA** | Export the merge as a standalone `.safetensors` file |
| **Merged LoRA to Hook** | Apply merged LoRA per-conditioning instead of globally |
| **LoRA Optimizer (Legacy)** | All parameters on one node (superseded by Optimizer + Settings) |

Also accepts standard tuple-format stacks `(lora_name, model_strength, clip_strength)` from Efficiency Nodes, Comfyroll, and similar packs.

> Full parameter reference: **[Nodes wiki page](docs/wiki/Nodes.md)** · **[Configuration Guide](docs/wiki/Configuration-Guide.md)** · **[Workflows](docs/wiki/Workflows.md)**

---

<details>
<summary><b>Examples</b></summary>

### Z-Image (Lumina2) — 3 LoRAs merged

<p align="center"><img src="assets/zit_example_1.png" width="720" alt="Z-Image 3-LoRA merge example"></p>

</details>

---

## How It Works

<p align="center"><a href="assets/optimizer-pipeline.png"><img src="assets/optimizer-pipeline.svg" alt="Optimizer Pipeline" width="100%"></a></p>

<details>
<summary><b>Per-Group Adaptive Merge — Deep Dive</b></summary>

The key insight: two LoRAs may overlap in some model blocks but not others. A face LoRA and a style LoRA might only conflict in attention layers 4-7, while the rest of the model is touched by only one of them.

Instead of picking one global strategy (which either wastes TIES trimming on non-overlapping blocks or misses real conflicts), the optimizer decides **per resolved target group**:

<div align="center">

| Condition | Strategy |
|-----------|----------|
| Only 1 LoRA touches this group | `weighted_sum` — full strength, no dilution |
| 2+ LoRAs, low excess conflict + low subspace overlap | `weighted_average` — mostly independent updates |
| 2+ LoRAs, high similarity + low excess conflict | `consensus` — aligned, low-interference merge |
| 2+ LoRAs, excess conflict > 25% with overlapping subspaces | `ties` — resolve real conflicts with trim/elect/merge |
| Magnitude ratio > 2x in the group | `total` sign method (stronger LoRA dominates) |
| Magnitude ratio <= 2x in the group | `frequency` sign method (equal votes) |

</div>

This means non-overlapping regions keep 100% of their LoRA's effect, while genuinely conflicting regions get proper TIES resolution. When `decision_smoothing > 0`, those per-group metrics are softly pulled toward the block average so adjacent layers do not flip strategies due to noisy samples.

<p align="center"><a href="assets/merge-strategies.png"><img src="assets/merge-strategies.svg" alt="Merge Strategies Comparison" width="100%"></a></p>

#### Two-Pass Streaming Architecture

The optimizer uses a **two-pass streaming architecture** for low memory usage:
- **Pass 1 (Analysis):** Resolves trainer aliases to target weights, aggregates alias collisions per LoRA, samples conflict and magnitude statistics per target group, then discards the diffs. Only lightweight scalars are kept.
- **Pass 2 (Merge):** Recomputes diffs per target group, looks up that group's conflict data, picks a strategy for it, and merges. Each group is freed after merging. Standard linear merges stay in exact low-rank form; nonlinear merges and optional compression still use dense/SVD paths.

Peak memory is still roughly "one target group at a time," but the exact peak depends on the largest layer, how many LoRAs hit it, and whether extra quality/compression steps are enabled. GPU-accelerated on both passes.

<details>
<summary><b>What It Analyzes</b></summary>

- Per-LoRA metrics (rank, key count, effective L2 norms)
- Pairwise raw + magnitude-weighted conflict ratios per target group (sampled for efficiency)
- Excess conflict over the cosine baseline, plus low-rank subspace overlap
- Pairwise cosine similarity (directional alignment between LoRAs)
- Magnitude / activation-importance distribution per target group
- Key overlap between LoRAs

</details>

</details>

---

## Optimizer Features

<details>
<summary><b>TIES Merging</b></summary>

The optimizer automatically selects TIES-Merging (Trim, Elect Sign, Disjoint Merge — [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)) on prefixes where sign conflicts are detected between LoRAs.

<p align="center"><a href="assets/ties-diagram.png"><img src="assets/ties-diagram.svg" alt="TIES Merging Pipeline" width="100%"></a></p>

</details>

<details>
<summary><b>DARE / DELLA Sparsification</b></summary>

DARE and DELLA **sparsify each LoRA's diff before merging**, reducing parameter interference between LoRAs. The implementations here are practical LoRA-oriented variants inspired by those papers, not paper-faithful reproductions. Available in two modes: **standard** (drops weights everywhere) and **conflict-aware** (only drops weights where LoRAs actually interfere).

<p align="center"><a href="assets/sparsification-diagram.png"><img src="assets/sparsification-diagram.svg" alt="DARE / DELLA Sparsification" width="100%"></a></p>

| Method | How It Works |
|--------|-------------|
| **DARE** | Bernoulli random mask at given density. Survivors rescaled by 1/density to preserve expected value. Fast and unbiased. |
| **DELLA** | Per-row magnitude ranking. Low-magnitude elements get higher drop probability, high-magnitude elements are kept. More surgical than DARE. |
| **DARE (conflict-aware)** | Same as DARE, but only applied at positions where 2+ LoRAs push in **opposite directions**. Same-sign positions (where LoRAs reinforce each other) are left untouched. |
| **DELLA (conflict-aware)** | Same as DELLA, but only at conflict positions. Unique contributions from each LoRA are fully preserved. |

**Why conflict-aware?** Standard sparsification drops weights everywhere — including positions where only one LoRA contributes, or where multiple LoRAs agree. This destroys useful signal. Conflict-aware variants compute a **sign-conflict mask** first: positions where LoRAs push in opposite directions (actual interference). Only those positions get sparsified. The result: interference is reduced without sacrificing unique features.

**Interaction with merge strategies:**
- **TIES mode:** DARE/DELLA *replaces* the TIES trim step (both achieve sparsification, no need for both)
- **Other modes:** Applied as preprocessing before the merge operation

| Setting | Default | Options |
|---------|---------|---------|
| `sparsification` | disabled | `disabled`, `dare`, `della`, `dare_conflict`, `della_conflict` |
| `sparsification_density` | 0.7 | Fraction of parameters to keep (lower = more aggressive) |

</details>

<details>
<summary><b>Merge Refinement (Refine / Full)</b></summary>

Optional preprocessing steps applied to weight diffs before merging, selectable via the `merge_refinement` dropdown:

<p align="center"><a href="assets/merge-quality-diagram.png"><img src="assets/merge-quality-diagram.svg" alt="Merge Quality Pipeline" width="100%"></a></p>

| Level | What It Adds | Cost |
|-------|-------------|------|
| **none** (default) | Merge as-is, no extra processing | Baseline |
| **refine** | Direction orthogonalization + TALL-mask selfish weight protection | Minimal extra compute, no extra VRAM |
| **full** | KnOTS SVD alignment + orthogonalization + TALL-masks | More VRAM for SVD decomposition |

**TALL-masks** (refine+): Identifies "selfish" weights — positions where one LoRA dominates and others contribute little. These weights are separated from the consensus merge and added back afterward, protecting each LoRA's unique features from being averaged away.

**Direction orthogonalization** (refine+): Projects LoRA diffs to be mutually orthogonal, reducing interference between LoRAs that modify overlapping weight regions.

**KnOTS SVD alignment** (full): Projects all LoRA diffs into a shared singular value basis via truncated SVD before merging. This makes diffs more directly comparable by aligning their representation spaces. Falls back to CPU on GPU OOM, skips gracefully if both fail.

**Interaction with other settings:**
- Works with all merge modes (TIES, weighted_average, SLERP, etc.)
- Combines with DARE/DELLA sparsification — sparsification runs first, then refinement
- Best combination: `maximum` + `della_conflict` (or `dare_conflict`) for full pipeline
- Single-LoRA prefixes: all enhancements short-circuit (no work to do)

| Setting | Default | Options |
|---------|---------|---------|
| `merge_refinement` | none | `none`, `refine`, `full` |

</details>

<details>
<summary><b>Key Filter</b></summary>

<a name="key-filter"></a>

Each LoRA has a per-LoRA `key_filter` setting (available on both **LoRA Stack** and **LoRA Stack (Dynamic)** in advanced mode) that controls which target groups that LoRA contributes to, based on how many LoRAs in the stack share each resolved target:

| Filter | Behavior | Use Case |
|--------|----------|----------|
| `all` (default) | Contribute to all keys | Normal merging |
| `shared_only` | Only contribute to keys present in 2+ LoRAs | Strip variant-specific keys (I2V/VACE) from this LoRA |
| `unique_only` | Only contribute to keys present in exactly 1 LoRA | Extract only the variant-specific adapter keys from this LoRA |

This is especially useful for Wan T2V/I2V/VACE LoRAs, which share ~90% of weights but each variant has unique keys (I2V: `cross_attn.k_img/v_img`, `img_emb`; VACE: `vace_blocks.*`, `vace_patch_embedding`).

Because the filter is per-LoRA, you can apply different filters to different LoRAs in the same stack — e.g., "take only the unique VACE keys from LoRA #2 while merging all keys from LoRA #1".

**Example — making an I2V LoRA T2V-compatible:**
1. Stack a T2V LoRA + an I2V LoRA together
2. Set the I2V LoRA's `key_filter` to `shared_only`
3. The I2V-only keys (`k_img`, `v_img`, `img_emb`, etc.) are skipped for that LoRA since they appear in only 1 LoRA
4. The merged result contains only the shared T2V-compatible weights

**Example — extracting a lightweight I2V adapter:**
1. Same stack (T2V + I2V)
2. Set the I2V LoRA's `key_filter` to `unique_only`
3. Only the I2V-specific keys are contributed by that LoRA — a small adapter with just the variant-specific weights

The filter uses the raw `n_loras` count from Pass 1 (before any filtering) and now participates in analysis as well as Pass 2 merge.

</details>

<details>
<summary><b>Auto-Strength</b></summary>

When `auto_strength` is set to `enabled`, the optimizer automatically reduces per-LoRA strengths before merging to prevent overexposure from stacking. This is especially useful on distilled/turbo models where 2+ LoRAs at full strength cause blown-out results even with strong merge settings.

The algorithm uses **interference-aware energy normalization**: during Pass 1 it streams exact Frobenius norms and pairwise dots for each LoRA branch, then computes the exact vector-sum energy separately for model and CLIP updates. All strengths are uniformly scaled so the total combined energy matches what the strongest single LoRA would contribute alone.

- **Aligned LoRAs** (cos~1) — stronger reduction (they reinforce each other, so combined energy is high)
- **Orthogonal LoRAs** (cos~0) — moderate reduction, optionally clamped by an architecture-aware floor
- **Opposing LoRAs** (cos~-1) — minimal reduction (they cancel out, so combined energy is low)

When orthogonal LoRAs are effectively independent, the optimizer can clamp the scale factor with `auto_strength_floor`:

| Architecture | Default floor |
|-------------|---------------|
| Wan / LTX Video | 1.0 |
| SD / SDXL / Flux / Z-Image | 0.85 |
| LLM-style presets | 0.9 |

`auto_strength_floor = -1` uses the architecture default. Setting `0.0–1.0` overrides it manually.

| Scenario | Result |
|----------|--------|
| 2 aligned LoRAs (cos~1) at strength 1.0 | Each reduced to ~0.50 |
| 2 orthogonal LoRAs (cos~0) at strength 1.0 | Each reduced to ~0.71 before floor-clamping |
| 2 opposing LoRAs (cos~-1) at strength 1.0 | ~1.0 each (they cancel) |
| 1 strong + 1 weak LoRA | Proportional reduction |
| Single LoRA | No change |
| `auto_strength` disabled | No adjustment |

Your original strength ratios are always preserved — the algorithm only scales them down uniformly.

</details>

<details>
<summary><b>Decision Smoothing</b></summary>

**`decision_smoothing`** — blends each group's decision metrics toward the average of its surrounding block. This reduces jagged layer-to-layer mode flips when the stack is noisy.

**`smooth_slerp_gate`** — when enabled, uses per-prefix cosine similarity (computed during analysis) instead of the collection average for the SLERP interpolation gate. This makes the SLERP weight vary per layer based on local alignment rather than using a single global value. Available on the LoRA Merge Settings node or the Legacy optimizer.

</details>

<details>
<summary><b>Architecture-Aware Key Normalization</b></summary>

Different LoRA trainers (Kohya, AI-Toolkit, LyCORIS, diffusers/PEFT) produce LoRAs with **different key naming conventions** for the same model weights. When mixing LoRAs from different trainers, the optimizer sees no key overlap and cannot merge them correctly.

Key normalization auto-detects the model architecture from LoRA key patterns and remaps all keys to a canonical format, enabling correct overlap detection and conflict analysis across trainer formats.

<p align="center"><a href="assets/key-normalization.png"><img src="assets/key-normalization.svg" alt="Architecture-Aware Key Normalization" width="100%"></a></p>

| Architecture | Detected From | Normalization |
|-------------|--------------|---------------|
| **Z-Image** (Lumina2) | `diffusion_model.layers.N.attention`, `single_transformer_blocks` | Prefix standardization, QKV split for per-component analysis, re-fuse after merge |
| **FLUX** | `double_blocks`/`single_blocks`, `transformer.transformer_blocks` | AI-Toolkit / Kohya / diffusers unified to canonical format |
| **Wan** 2.1/2.2 | `blocks.N` with `self_attn`/`cross_attn`/`ffn` | LyCORIS / diffusers / Musubi Tuner unified, RS-LoRA alpha fix |
| **SDXL** | `lora_te1_`/`lora_te2_`, `input_blocks`/`down_blocks` | Text encoder + UNet key unification |
| **LTX Video** | `adaln_single`, `transformer_blocks` with `attn1`/`attn2` | Trainer format unification |
| **ACE-Step** | `layers.N` with `self_attn`/`cross_attn` and `q_proj`/`k_proj`/`v_proj` | Attention key unification |
| **Qwen-Image** | `transformer_blocks` with `img_mlp`/`txt_mlp`/`img_mod`/`txt_mod` | Dual-stream key unification |

**Z-Image QKV handling:** Z-Image LoRAs often fuse Q, K, V projections into a single `attention.qkv` weight. The normalizer splits these into separate `to_q`/`to_k`/`to_v` components for per-component conflict analysis, then **re-fuses** them back to the native format after merging.

| Setting | Default | Effect |
|---------|---------|--------|
| `normalize_keys` | enabled | `disabled` or `enabled`. Recommended for mixed-trainer stacks and required for Z-Image QKV fusion. |

</details>

<details>
<summary><b>Architecture-Aware Behavior Profiles</b></summary>

All numeric thresholds in the optimizer (density estimation, conflict detection, auto-strength scaling, scoring heuristics) are tuned per architecture family. The `architecture_preset` setting selects the appropriate thresholds — `auto` detects from LoRA key patterns.

| Preset | Architectures | Key Differences | Orthogonal floor |
|--------|--------------|-----------------|------------------|
| `sd_unet` | SD 1.5, SDXL | Density range [0.1, 0.9], noise floor 10%, max strength cap 3.0 | 0.85 |
| `dit` | Flux, WAN, Z-Image, LTX, HunyuanVideo | Density range [0.4, 0.95], noise floor 5%, max strength cap 5.0 | 0.85 by default, 1.0 for Wan/LTX |
| `llm` | Qwen-Image, LLaMA-based | Density range [0.1, 0.8], noise floor 15%, max strength cap 3.0 | 0.9 |

**Why it matters:** DiT architectures have denser weight distributions than UNet — with UNet thresholds, the optimizer underestimates density and clips suggested strength too aggressively. LLM-based models are sparser and benefit from lower density ceilings.

| Setting | Default | Options |
|---------|---------|---------|
| `architecture_preset` | auto | `auto`, `sd_unet`, `dit`, `llm`. Auto-detection uses the same key pattern matching as key normalization |

**Note:** This is orthogonal to `strategy_set` (which controls *which strategies* are available — consensus, SLERP, etc.). Architecture preset controls the *numeric thresholds* those strategies use.

</details>

<details>
<summary><b>SVD Patch Compression</b></summary>

After merging, full-rank diff patches consume ~128x more RAM than standard LoRA patches (64MB vs 0.5MB per key for a 4096x4096 weight). The optimizer re-compresses merged patches to low-rank via truncated SVD, dramatically reducing post-merge RAM.

| Mode | What gets compressed | Quality | RAM savings |
|------|---------------------|---------|-------------|
| `smart` (default) | `weighted_sum` and `weighted_average` prefixes only | Lossless — sum of input ranks preserves all merge information | ~32x on compressed prefixes |
| `aggressive` | Everything including TIES | Lossy on TIES prefixes — nonlinear ops (trim, sign election) produce full-rank results that can't be perfectly captured | ~32x on all prefixes |
| `disabled` | Nothing | No loss | No savings |

When dense compression is needed, the compression rank is automatically computed as the sum of all input LoRA ranks. For example, 3 rank-32 LoRAs produce a rank-96 compressed patch — enough to represent the full merge on linear operations when no extra nonlinear processing is involved.

> **Tip:** For video models (LTX, Wan, etc.) with high RAM usage, use `additive` mode + `smart` (or `aggressive`) compression. Every patch gets losslessly compressed with minimal RAM footprint.

</details>

<details>
<summary><b>Optimization Modes</b></summary>

| Mode | Behavior |
|------|----------|
| `per_prefix` (default) | Each weight group picks its own strategy based on local conflict data |
| `global` | Single strategy for all prefixes (original behavior) |
| `additive` | Simple weighted addition — no conflict resolution. Preserves all weights exactly. Use for DPO/edit/distill LoRAs, or with patch compression for minimal RAM |

</details>

<details>
<summary><b>Block Strategy Map</b></summary>

The analysis report includes a visual block-by-block map showing what strategy was used and why:

```
--- Block Strategy Map ---
  input_blocks.0   ====  sum  1 LoRA (6x)
  input_blocks.4   ----  avg  12% conflict (6x)
  middle_block.1   ####  TIES 42% conflict (6x)
  output_blocks.3  ----  avg  8% conflict (6x)
  output_blocks.8  ====  sum  1 LoRA (6x)
  Legend: ==== sum (single LoRA)  ---- avg (compatible)  #### TIES (conflict)
```

</details>

<details>
<summary><b>Memory Options</b></summary>

| Option | Default | Effect |
|--------|---------|--------|
| `cache_patches` | enabled | Cache merged patches in RAM for faster re-execution. Disable to free RAM after merge (recommended for video models) |
| `patch_compression` | smart | SVD re-compression of merged patches (see above) |
| `svd_device` | gpu | Device for SVD compression. GPU is ~10-50x faster than CPU. Use CPU if GPU memory is tight |
| `free_vram_between_passes` | disabled | Release GPU cache between analysis and merge passes. Lowers peak VRAM at negligible speed cost |

</details>

<details>
<summary><b>Example Report</b></summary>

```
==================================================
LORA OPTIMIZER - ANALYSIS REPORT
==================================================
Architecture preset: sd_unet (SD/SDXL UNet)

--- Per-LoRA Analysis ---
  style_lora.safetensors:
    Strength: 1.0
    Keys: 192
    Avg rank: 64
    L2 norm (mean): 0.0847
  detail_lora.safetensors:
    Strength: 0.8
    Keys: 192
    Avg rank: 32
    L2 norm (mean): 0.0423

--- Auto-Strength Adjustment ---
  style_lora.safetensors: 1.0 -> 0.6345
  detail_lora.safetensors: 0.8 -> 0.5076
  Scale factor: 0.6345
  Method: interference-aware energy normalization
    Avg pairwise cosine similarity: 0.312 (mostly aligned (reinforcing))
    Interference-aware energy: 0.1335 (orthogonal assumption: 0.1196)

--- Pairwise Analysis ---
  style_lora.safetensors vs detail_lora.safetensors:
    Overlapping positions: 89420
    Sign conflicts: 31297 (35.0%)
    Cosine similarity: 0.312

--- Collection Statistics ---
  Total LoRAs: 2
  Total unique keys: 196
  Avg sign conflict ratio: 35.0%
  Magnitude ratio (max/min L2): 2.00x

--- Auto-Selected Parameters ---
  Merge mode: ties
  Density: 0.42
  Sign method: frequency
  Sparsification: DARE
  Sparsification density: 0.70 (keep rate)
  For TIES prefixes: replaces trim step; others: preprocessing
  (global fallback — each prefix uses its own parameters)

--- Per-Prefix Strategy ---
  weighted_sum (single LoRA):        28 prefixes (14%)
  weighted_average (low conflict):  120 prefixes (61%)
  ties (high conflict):              48 prefixes (24%)
  Total:                            196 prefixes

--- Block Strategy Map ---
  input_blocks.0   ====  sum  1 LoRA (6x)
  input_blocks.1   ====  sum  1 LoRA (6x)
  input_blocks.4   ----  avg  12% conflict (6x)
  input_blocks.5   ####  TIES 38% conflict (6x)
  middle_block.1   ####  TIES 42% conflict (6x)
  output_blocks.3  ----  avg  15% conflict (6x)
  output_blocks.8  ====  sum  1 LoRA (6x)
  Legend: ==== sum (single LoRA)  ---- avg (compatible)  #### TIES (conflict)

--- Reasoning ---
  Sign conflict ratio 35.0% > 25% threshold -> TIES mode selected
    TIES resolves sign conflicts via trim + elect sign + disjoint merge
  Auto-density estimated at 0.42 from magnitude distribution
  Magnitude ratio 2.00x <= 2x -> 'frequency' sign method (equal voting)
    Similar-strength LoRAs get equal votes

--- Merge Summary ---
  Keys processed: 196
  Model patches: 168
  CLIP patches: 28
  Output strength: 1.0
  CLIP strength: 1.0

==================================================
```

Connect the `STRING` output to a **Show Text** node to see the report in ComfyUI.

</details>

<details>
<summary><b>Important notes & limitations</b></summary>

> **Structural & Edit LoRAs:** Do not put distillation LoRAs (LCM, Lightning, Turbo, Hyper), DPO LoRAs, or **edit model LoRAs** (Qwen edit, Klein edit, instruction-editing LoRAs) in the optimizer stack. These LoRAs modify the model's fundamental behavior — their weights are precisely calibrated and merging them with style LoRAs can break their training. Apply them via a standard **Load LoRA** node upstream, then feed only your style/character LoRAs into the optimizer. If you must include an edit LoRA in the stack, use `additive` mode and disable sparsification to avoid weight trimming.

> **Limitation:** The optimizer only analyzes LoRAs in its own stack. It cannot see LoRA patches applied by upstream nodes (Load LoRA, etc.) — those stack additively on top of the optimizer's output. Fully baked merges (safetensors checkpoints) are indistinguishable from base weights and cannot be detected.

</details>

---

## AutoTuner

Automatically sweeps all merge parameters (mode, sparsification, density, dampening, quality level) and ranks configurations for your LoRA stack. Runs Pass 1 analysis once, scores all parameter combinations via heuristic proxies, then merges the top-N candidates and measures output quality. When an `AUTOTUNER_EVALUATOR` is connected, the built-in score can be blended with external prompt/reference evaluation logic. Outputs the highest-ranked merge directly as `MODEL`/`CLIP`, plus a ranked report and `TUNER_DATA` for exploring alternatives via a **Merge Selector** node.

> Full parameter reference: **[Nodes wiki page](docs/wiki/Nodes.md)**

<details>
<summary><b>Diff Cache</b></summary>

During the parameter sweep, each candidate recomputes raw LoRA diffs (A@B matmul) from scratch — even though diffs depend only on LoRA content, not merge config. The diff cache stores these diffs after the first candidate and reuses them for subsequent candidates, eliminating redundant computation.

| Mode | Behavior |
|------|----------|
| `disabled` | Recomputes diffs each time. No extra memory |
| `auto` | Uses RAM up to `diff_cache_ram_pct` of free memory, then spills to disk. Recommended for most setups |
| `ram` | All diffs in RAM. Fastest, but uses ~1.5 GB (SDXL) to ~6 GB (Flux) |
| `disk` | All diffs to temp files with memory-mapping. Slowest cache mode, but minimal RAM |

When `auto` mode runs out of disk space, it falls back to RAM automatically.

| Setting | Default | Effect |
|---------|---------|--------|
| `diff_cache_mode` | auto | Cache mode selection |
| `diff_cache_ram_pct` | 0.5 | Fraction of free system RAM for `auto` mode (0.1–0.9) |

</details>

<details>
<summary><b>VRAM Budget</b></summary>

The `vram_budget` slider (0.0–1.0) controls what fraction of free VRAM to use for storing merged patches on GPU. Default is 0 (all patches on CPU). Setting it higher keeps patches on GPU, reducing RAM usage on systems with enough VRAM. Available on both LoRA Optimizer and LoRA AutoTuner.

</details>

---

<details>
<summary><b>LoRA Pruner</b></summary>

Drop-in replacement for ComfyUI's **Load LoRA** node that prunes before applying. Useful for slimming down large LoRAs or surgically removing layers that hurt quality — for example, removing transformer blocks in LTX-Video 2.3 LoRAs that slow down motion.

**Workflow:**
```
Load Checkpoint → MODEL ──► LoRA Pruner ──► MODEL ──► KSampler
               → CLIP  ──► (optional)  ──► CLIP
                                        ──► LORA_DATA ──► Save Merged LoRA (optional)
                                        ──► STRING (report)
```

### Pruning Modes

| Mode | What It Does |
|------|-------------|
| **block** | Groups keys by transformer block (e.g. `double_blocks.15`) and drops the least important blocks entirely. Good for removing whole layers that contribute little |
| **key** | Drops individual LoRA key prefixes (e.g. a single attention projection). Finer-grained than block mode |
| **rank** | Keeps all keys but reduces each key's internal rank via SVD. A rank-64 key might become rank-32, preserving the effect everywhere but making it thinner |
| **block+rank** | First drops the least important blocks, then reduces rank on all surviving keys. Recommended for aggressive pruning |

### Settings

| Setting | Default | Effect |
|---------|---------|--------|
| `drop_percentage` | 20% | Percentage of blocks or keys to drop (block/key/block+rank modes) |
| `rank_reduction` | 20% | How much rank to remove from each key (rank/block+rank modes) |
| `importance_metric` | frobenius_norm | Scoring metric: `frobenius_norm` (total magnitude) or `spectral_norm` (peak directional impact) |
| `rank_mode` | energy_threshold | `energy_threshold` adapts per key — keys with concentrated spectra get cut more aggressively. `percentage` applies a flat cut everywhere |
| `strength_model` | 1.0 | How strongly to apply the pruned LoRA to the diffusion model |
| `strength_clip` | 1.0 | How strongly to apply the pruned LoRA to CLIP (only when CLIP is connected) |

### LTX-Video 2.3 Guide

Certain blocks in LTX-Video LoRAs can slow down motion or add unwanted artifacts. Start with:
- **block+rank** mode, **20%** drop / **20%** rank reduction
- If motion is still sluggish, try **30-40%** drop
- Check the report to see which blocks were dropped and how much rank was removed

### Notes

- **LoKr/LoHa keys** are always preserved (exotic format cannot be rank-reduced)
- **LoCon mid tensors** are handled for scoring but dropped during rank reduction (recompressed as standard LoRA)
- **DoRA scale** is preserved for block/key modes, dropped for rank reduction (baked into new factors)
- The **LORA_DATA** output connects to **Save Merged LoRA** to export the pruned result as a new `.safetensors` file
- Scoring uses fast paths: Frobenius via trace identity on [r,r] matrices, spectral norm via QR on [r,r] cores — no dense [out,in] matrix expansion

</details>

<details>
<summary><b>Other Nodes & Workflows</b></summary>

### Merge Selector

Applies a specific configuration from AutoTuner results without re-running the sweep. Connect `TUNER_DATA` from a LoRA AutoTuner (or Load Tuner Data) node and set the `selection` index to choose which ranked configuration to apply (1 = top-ranked, 2 = next-ranked, etc.).

**Workflow:**
```
LoRA AutoTuner → TUNER_DATA → Merge Selector (selection=2) → try the 2nd-ranked config
                      ↓
              Save Tuner Data → (reload later) → Load Tuner Data → Merge Selector
```

---

### AutoTuner → Optimizer Bridge

Chain the AutoTuner and the Legacy optimizer in a single model line for a "rank, then tweak" workflow. Only one node merges at a time — the other passes the model through. The Legacy optimizer's `settings_source` switch controls which node is authoritative, and the UI bridge keeps the paired widgets in sync.

<p align="center">
  <a href="assets/bridge-workflow.png"><img src="assets/bridge-workflow.svg" alt="AutoTuner ↔ Optimizer Bridge workflow" width="700"></a>
</p>

```
[Load Model] → [AutoTuner] → model → [Optimizer (Legacy)] → MODEL → sampler
[LoRA Stack]  → [AutoTuner]
[LoRA Stack]  → [Optimizer (Legacy)]
               [AutoTuner] → tuner_data → [Optimizer (Legacy)]
```

| Legacy Optimizer `settings_source` | What happens |
|----|----|
| `from_autotuner` | AutoTuner merges → Legacy Optimizer passes through. Optimizer widgets show the winning config. |
| `manual` | AutoTuner passes the base model through → Legacy Optimizer merges with its own widget settings. |
| `from_tuner_data` | Legacy Optimizer reads settings from connected `tuner_data` input. |

**Typical flow:**
1. Start with `from_autotuner` — let the AutoTuner find the best config
2. Inspect the Optimizer's widgets to see what won
3. Switch to `manual` — the Optimizer takes over, starting from the AutoTuner's recommendation
4. Tweak settings (merge_refinement, sparsification, etc.) and re-run

Switching between modes is instant — the AutoTuner reuses its cached sweep results.

> **Note:** The bridge workflow requires the **LoRA Optimizer (Legacy)** node. The simplified **LoRA Optimizer** uses Settings nodes and `tuner_data` input instead.

---

### Save / Load Tuner Data

Two utility nodes for persisting AutoTuner results to disk:

**Save Tuner Data** — Saves `TUNER_DATA` into a selected `tuner_data` folder as `.tuner` or `.json`. Subdirectories are allowed; path traversal outside that folder is blocked. Optional overwrite control avoids clobbering previous runs. `OUTPUT_NODE = True`.

**Load Tuner Data** — Dropdown of saved tuner data files. Outputs `TUNER_DATA` ready for Merge Selector. Auto-reloads when the file changes on disk.

---

### Evaluator Utilities

**Build AutoTuner Python Evaluator** — packages a Python module path + callable name into an `AUTOTUNER_EVALUATOR` object. The callable can run prompts, compare references, and return a score in `[0, 1]`.

The evaluator callable receives keyword arguments: `model`, `clip`, `lora_data`, `config`, `context`, and `analysis_summary`.

---

### Save Merged LoRA

Saves the optimizer's merged result as a standalone `.safetensors` file that works with any standard LoRA loader.

Connect the `LORA_DATA` output from LoRA Optimizer to this node.

| Option | Default | Effect |
|--------|---------|--------|
| `save_folder` | first configured LoRA folder | Choose which configured ComfyUI LoRA directory to save into |
| `filename` | `merged_lora` | File name relative to `save_folder`. Subdirectories are allowed (e.g. `merged/my_lora`) |
| `save_rank` | 0 (auto) | 0 = use each layer's existing rank from the merge. Non-zero = force this rank for layers that need compression |
| `bake_strength` | enabled | When on, the saved LoRA reproduces your exact merge at strength 1.0. When off, strengths are not baked in |

**Outputs:** `STRING` (file path)

---

### Merged LoRA to Hook

Wraps the optimizer's merged patches as a **conditioning hook** (`HOOKS`) for per-conditioning LoRA application. Instead of applying the merged LoRA globally to the model, you can attach it to specific conditioning entries using ComfyUI's hook system.

Connect the `LORA_DATA` output from LoRA Optimizer to this node, then connect the `HOOKS` output to a **Cond Set Props** (or similar) node.

**Inputs:** `LORA_DATA` (required), `HOOKS` (optional — chain with existing hooks)

**Outputs:** `HOOKS`

Use this node when you want the merged LoRA to apply **only to specific conditioning** rather than the entire model:

- **Per-prompt LoRA:** Apply different merged LoRAs to positive vs negative conditioning
- **Scheduled application:** Combine with hook keyframes to apply the LoRA only during certain sampling steps
- **Regional conditioning:** Use with area-based conditioning to apply the LoRA to specific image regions
- **Preserving the base model:** Keep the MODEL output clean (unpatched) while still using the merged LoRA through conditioning hooks

**Workflow example:**
```
Load Checkpoint → MODEL ──┬──→ LoRA Optimizer → LORA_DATA → Merged LoRA to Hook → HOOKS
                           │                                                          ↓
                           └──→ KSampler ←──── Conditioning ←──── Cond Set Props
```

The `prev_hooks` input allows chaining multiple hook sources together.

---

### WanVideo LoRA Optimizer

Variant of the LoRA Optimizer for **WanVideo models** (via [kijai's WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)). Accepts `WANVIDEOMODEL` instead of `MODEL`, skips CLIP, and applies merged patches in-memory.

All merging algorithms are inherited — TIES, DARE/DELLA, SVD compression, auto-strength, per-prefix adaptive merge, merge refinement (KnOTS, orthogonalization, TALL-masks), and Wan key normalization (LyCORIS, diffusers, Fun LoRA, finetrainer, RS-LoRA) all work identically.

**Basic workflow:**
```
WanVideoModelLoader → WANVIDEOMODEL → WanVideo LoRA Optimizer → WANVIDEOMODEL → WanVideoSampler
                                               ↑
                        LoRA Stack ─────────────┘
```

**Chaining with individual LoRAs:** Individual (non-merged) LoRAs go through WanVideoLoraSelect → model loader as usual. Our optimizer applies merged LoRAs on top — both coexist in the model patcher.

```
WanVideoLoraSelect → WanVideoModelLoader → WANVIDEOMODEL → WanVideo LoRA Optimizer → Sampler
                                                                    ↑
                                             LoRA Stack ────────────┘
```

**Key defaults differ from the standard optimizer:**
- `normalize_keys` = **enabled** — WanVideo LoRAs come from many trainers, normalization is commonly needed
- `cache_patches` = **disabled** — video models are large, caching uses significant RAM
- `architecture_preset` = **dit** — DiT-tuned thresholds (higher density floor, wider strength range)

</details>

---

<details>
<summary><b>Compatibility</b></summary>

- **Models:** SD 1.5, SDXL, Flux, Z-Image (Lumina2), Wan 2.1/2.2, LTX Video, ACE-Step, Qwen-Image, and other architectures supported by ComfyUI
- **LoRA formats:** Standard LoRA, LoCon, and LoRA/LoCon-style trainer variants whose tensors reduce to up/down(/mid) adapters (including many diffusers/PEFT and LyCORIS naming schemes)
- **Trainers:** Kohya, AI-Toolkit, LyCORIS, Musubi Tuner, diffusers — auto-normalized when `normalize_keys` is enabled
- **Flux sliced weights:** Handled correctly (linear1_qkv offsets)
- **Z-Image fused QKV:** Split for per-component analysis, re-fused after merge
- **Stack formats:** Native LoRA Stack dicts, plus standard tuples from Efficiency Nodes / Comfyroll

</details>

<details>
<summary><b>Credits</b></summary>

- Originally based on [ComfyUI-ZImage-LoRA-Merger](https://github.com/DanrisiUA/ComfyUI-ZImage-LoRA-Merger) by DanrisiUA
- Per-prefix adaptive approach inspired by [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) by shootthesound (per-block LoRA analysis)
- Thanks to Scruffy and Ramonguthrie for suggesting the per-block analysis approach
- TIES-Merging: [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)
- DARE: [Yu et al., ICML 2024](https://arxiv.org/abs/2311.03099) — Drop And REscale for language model merging
- DELLA: [Deep et al., 2024](https://arxiv.org/abs/2406.11617) — magnitude-aware sparsification
- KnOTS: [Ramé et al., 2024](https://arxiv.org/abs/2407.09095) — SVD alignment for model merging
- TALL-masks: [Wang et al., 2024](https://arxiv.org/abs/2406.12832) — selfish weight protection via task-aware masks
- Column-wise merging inspired by ZipLoRA: [Shah et al., 2025](https://arxiv.org/abs/2311.13600) — structural sparsity for LoRA merging

</details>

<details>
<summary><b>Development Timeline</b></summary>

<p align="center"><a href="assets/timeline.png"><img src="assets/timeline.svg" alt="Development Timeline" width="720"></a></p>

</details>

## License

MIT License - see [LICENSE](LICENSE).
