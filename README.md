<p align="center">
  <img src="assets/banner.svg" alt="LoRA Optimizer" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Custom_Nodes-blue?style=flat-square" alt="ComfyUI">
  <img src="https://img.shields.io/badge/TIES_Merging-NeurIPS_2023-8b5cf6?style=flat-square" alt="TIES">
  <img src="https://img.shields.io/badge/DARE_%7C_DELLA-Sparsification-f59e0b?style=flat-square" alt="DARE/DELLA">
  <img src="https://img.shields.io/badge/Per--Prefix_Adaptive-Merge-e94560?style=flat-square" alt="Per-Prefix">
  <img src="https://img.shields.io/badge/SVD_Patch-Compression-64ffda?style=flat-square" alt="SVD">
  <img src="https://img.shields.io/badge/Architecture--Aware-Key_Normalization-22c55e?style=flat-square" alt="Key Normalization">
  <img src="https://img.shields.io/badge/Flux_%7C_SDXL_%7C_Wan_%7C_LTX_%7C_Z--Image-Compatible-22c55e?style=flat-square" alt="Compatible">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT">
</p>

---

A ComfyUI node that **automatically analyzes your LoRA stack** and selects the best merge strategy per weight group — diff-based merging, TIES conflict resolution, DARE/DELLA sparsification, per-prefix adaptive decisions, SVD patch compression, architecture-aware key normalization, and auto-tuned parameters. Two nodes: **LoRA Stack** (build input) and **LoRA Optimizer** (analyze + merge).

## The Problem

Stacking LoRAs in ComfyUI adds their effects together. On distilled/turbo models (SDXL-Turbo, LCM, Lightning, Flux-schnell), the accumulated effect exceeds what the model can handle, causing **overexposure, color blowout, and artifacts**.

```
model += lora1_effect x strength1
model += lora2_effect x strength2
total effect = strength1 + strength2  -->  easily exceeds 1.0
```

The optimizer solves this by computing full weight diffs, detecting sign conflicts per weight group, and merging each group with its optimal strategy.

<p align="center">
  <img src="assets/comparison.png" alt="Before/After Comparison" width="100%">
</p>

<p align="center"><img src="assets/merge-use-cases.png" width="720" alt="Should I merge this LoRA? Decision guide"></p>

---

## Nodes

### LoRA Stack

Builds a list of LoRAs for the optimizer. Chain multiple Stack nodes to add any number of LoRAs.

**Inputs:** LoRA selector, strength, optional previous `LORA_STACK`

**Outputs:** `LORA_STACK`

---

### LoRA Stack (Dynamic)

Single node with adjustable slot count (1–10) — replaces chaining multiple Stack nodes.

| Mode | Behavior |
|------|----------|
| **Simple** | One `strength` slider per LoRA (applies to both model and CLIP) |
| **Advanced** | Separate `model_strength` and `clip_strength` per LoRA for independent control |

Accepts an optional `lora_stack` input to chain with other Stack nodes.

**Outputs:** `LORA_STACK`

---

### LoRA Optimizer

The auto-optimizer. Takes a `LORA_STACK`, analyzes the LoRAs, and automatically selects the best merge mode and parameters **per weight group**. Outputs the merged result plus a detailed analysis report with a block strategy map.

Also accepts standard tuple-format stacks `(lora_name, model_strength, clip_strength)` from Efficiency Nodes, Comfyroll, and similar packs.

Uses a **two-pass streaming architecture** for low memory usage:
- **Pass 1 (Analysis):** Computes weight diffs per prefix, samples conflict and magnitude statistics per prefix, then discards the diffs. Only lightweight scalars are kept.
- **Pass 2 (Merge):** Recomputes diffs per prefix, looks up that prefix's conflict data, picks the optimal strategy for it, and merges. Each prefix is freed after merging. Non-TIES patches are SVD-compressed to low-rank by default.

Peak memory is ~one prefix at a time (~260MB) regardless of LoRA count or model size. GPU-accelerated on both passes.

<p align="center">
  <img src="assets/optimizer-pipeline.svg" alt="Optimizer Pipeline" width="100%">
</p>

#### What It Analyzes

- Per-LoRA metrics (rank, key count, effective L2 norms)
- Pairwise sign conflict ratios per prefix (sampled for efficiency)
- Pairwise cosine similarity (directional alignment between LoRAs)
- Magnitude distribution per prefix
- Key overlap between LoRAs

#### Per-Prefix Adaptive Merge

The key insight: two LoRAs may overlap in some model blocks but not others. A face LoRA and a style LoRA might only conflict in attention layers 4-7, while the rest of the model is touched by only one of them.

Instead of picking one global strategy (which either wastes TIES trimming on non-overlapping blocks or misses real conflicts), the optimizer decides **per weight prefix**:

| Condition | Strategy |
|-----------|----------|
| Only 1 LoRA touches this prefix | `weighted_sum` — full strength, no dilution |
| 2+ LoRAs, sign conflict <= 25% | `weighted_average` — compatible, simple merge |
| 2+ LoRAs, sign conflict > 25% | `ties` — resolve conflicts with trim/elect/merge |
| Magnitude ratio > 2x at prefix | `total` sign method (stronger LoRA dominates) |
| Magnitude ratio <= 2x at prefix | `frequency` sign method (equal votes) |

This means non-overlapping regions keep 100% of their LoRA's effect, while genuinely conflicting regions get proper TIES resolution.

<p align="center">
  <img src="assets/merge-strategies.svg" alt="Merge Strategies Comparison" width="100%">
</p>

#### TIES Merging

The optimizer automatically selects TIES-Merging (Trim, Elect Sign, Disjoint Merge — [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)) on prefixes where sign conflicts are detected between LoRAs.

<p align="center">
  <img src="assets/ties-diagram.svg" alt="TIES Merging Pipeline" width="100%">
</p>

#### DARE / DELLA Sparsification

DARE and DELLA **sparsify each LoRA's diff before merging**, reducing parameter interference between LoRAs. Available in two modes: **standard** (drops weights everywhere) and **conflict-aware** (only drops weights where LoRAs actually interfere).

<p align="center">
  <img src="assets/sparsification-diagram.svg" alt="DARE / DELLA Sparsification" width="100%">
</p>

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

#### Auto-Strength

When `auto_strength` is set to `enabled`, the optimizer automatically reduces per-LoRA strengths before merging to prevent overexposure from stacking. This is especially useful on distilled/turbo models where 2+ LoRAs at full strength cause blown-out results even with optimal merge mode selection.

The algorithm uses **interference-aware energy normalization**: it measures pairwise cosine similarity between LoRAs during analysis to account for directional alignment, then computes the exact vector-sum energy using the formula `||sum(v_i)||^2 = sum(||v_i||^2) + 2 * sum(||v_i|| * ||v_j|| * cos(v_i, v_j))`. All strengths are scaled so the total combined energy matches what the strongest single LoRA would contribute alone.

- **Aligned LoRAs** (cos~1) — stronger reduction (they reinforce each other, so combined energy is high)
- **Orthogonal LoRAs** (cos~0) — moderate reduction (independent contributions add in quadrature)
- **Opposing LoRAs** (cos~-1) — minimal reduction (they cancel out, so combined energy is low)

| Scenario | Result |
|----------|--------|
| 2 aligned LoRAs (cos~1) at strength 1.0 | Each reduced to ~0.50 |
| 2 orthogonal LoRAs (cos~0) at strength 1.0 | Each reduced to ~0.71 |
| 2 opposing LoRAs (cos~-1) at strength 1.0 | ~1.0 each (they cancel) |
| 1 strong + 1 weak LoRA | Proportional reduction |
| Single LoRA | No change |
| `auto_strength` disabled | No adjustment (default) |

Your original strength ratios are always preserved — the algorithm only scales them down uniformly.

#### Architecture-Aware Key Normalization

Different LoRA trainers (Kohya, AI-Toolkit, LyCORIS, diffusers/PEFT) produce LoRAs with **different key naming conventions** for the same model weights. When mixing LoRAs from different trainers, the optimizer sees no key overlap and cannot merge them correctly.

Key normalization auto-detects the model architecture from LoRA key patterns and remaps all keys to a canonical format, enabling correct overlap detection and conflict analysis across trainer formats.

<p align="center">
  <img src="assets/key-normalization.svg" alt="Architecture-Aware Key Normalization" width="100%">
</p>

| Architecture | Detected From | Normalization |
|-------------|--------------|---------------|
| **Z-Image** (Lumina2) | `diffusion_model.layers.N.attention`, `single_transformer_blocks` | Prefix standardization, QKV split for per-component analysis, re-fuse after merge |
| **FLUX** | `double_blocks`/`single_blocks`, `transformer.transformer_blocks` | AI-Toolkit / Kohya / diffusers unified to canonical format |
| **Wan** 2.1/2.2 | `blocks.N` with `self_attn`/`cross_attn`/`ffn` | LyCORIS / diffusers / Musubi Tuner unified, RS-LoRA alpha fix |
| **SDXL** | `lora_te1_`/`lora_te2_`, `input_blocks`/`down_blocks` | Text encoder + UNet key unification |
| **LTX Video** | `adaln_single`, `transformer_blocks` with `attn1`/`attn2` | Trainer format unification |
| **Qwen-Image** | `transformer_blocks` with `img_mlp`/`txt_mlp`/`img_mod`/`txt_mod` | Dual-stream key unification |

**Z-Image QKV handling:** Z-Image LoRAs often fuse Q, K, V projections into a single `attention.qkv` weight. The normalizer splits these into separate `to_q`/`to_k`/`to_v` components for per-component conflict analysis, then **re-fuses** them back to the native format after merging.

| Setting | Default | Effect |
|---------|---------|--------|
| `normalize_keys` | disabled | `disabled` or `enabled`. Enable when mixing LoRAs from different trainers or for Z-Image QKV fusion. |

#### SVD Patch Compression

After merging, full-rank diff patches consume ~128x more RAM than standard LoRA patches (64MB vs 0.5MB per key for a 4096x4096 weight). The optimizer re-compresses merged patches to low-rank via truncated SVD, dramatically reducing post-merge RAM.

| Mode | What gets compressed | Quality | RAM savings |
|------|---------------------|---------|-------------|
| `non_ties` (default) | `weighted_sum` and `weighted_average` prefixes only | Lossless — sum of input ranks preserves all merge information | ~32x on compressed prefixes |
| `all` | Everything including TIES | Lossy on TIES prefixes — nonlinear ops (trim, sign election) produce full-rank results that can't be perfectly captured | ~32x on all prefixes |
| `disabled` | Nothing | No loss | No savings |

The compression rank is automatically computed as the sum of all input LoRA ranks. For example, 3 rank-32 LoRAs produce a rank-96 compressed patch — enough to represent the full merge without quality loss on linear operations.

> **Tip:** For video models (LTX, Wan, etc.) with high RAM usage, use `weighted_sum_only` + `non_ties` (or `all`). Every patch gets losslessly compressed with minimal RAM footprint.

#### Optimization Modes

| Mode | Behavior |
|------|----------|
| `per_prefix` (default) | Each weight group picks its own strategy based on local conflict data |
| `global` | Single strategy for all prefixes (original behavior) |
| `weighted_sum_only` | Forces simple weighted sum everywhere — no TIES, no averaging. Combined with patch compression, all patches are fully compressible with zero quality loss |

#### Block Strategy Map

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

#### Memory Options

| Option | Default | Effect |
|--------|---------|--------|
| `cache_patches` | enabled | Cache merged patches in RAM for faster re-execution. Disable to free RAM after merge (recommended for video models) |
| `compress_patches` | non_ties | SVD re-compression of merged patches (see above) |
| `svd_device` | gpu | Device for SVD compression. GPU is ~10-50x faster than CPU. Use CPU if GPU memory is tight |
| `free_vram_between_passes` | disabled | Release GPU cache between analysis and merge passes. Lowers peak VRAM at negligible speed cost |

#### Inputs / Outputs

**Inputs:** `MODEL`, `CLIP` (optional), `LORA_STACK`, output strength, clip strength multiplier, auto strength, optimization mode, cache patches, compress patches, SVD device, free VRAM between passes, normalize keys, sparsification, sparsification density.

**Outputs:** `MODEL`, `CLIP`, `STRING` (analysis report), `LORA_DATA` (for Save Merged LoRA)

#### Example Report

```
==================================================
LORA OPTIMIZER - ANALYSIS REPORT
==================================================

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

> **Structural & Edit LoRAs:** Do not put distillation LoRAs (LCM, Lightning, Turbo, Hyper), DPO LoRAs, or **edit model LoRAs** (Qwen edit, Klein edit, instruction-editing LoRAs) in the optimizer stack. These LoRAs modify the model's fundamental behavior — their weights are precisely calibrated and merging them with style LoRAs can break their training. Apply them via a standard **Load LoRA** node upstream, then feed only your style/character LoRAs into the optimizer. If you must include an edit LoRA in the stack, use `weighted_sum_only` mode and disable sparsification to avoid weight trimming.

> **Limitation:** The optimizer only analyzes LoRAs in its own stack. It cannot see LoRA patches applied by upstream nodes (Load LoRA, etc.) — those stack additively on top of the optimizer's output. Fully baked merges (safetensors checkpoints) are indistinguishable from base weights and cannot be detected.

---

### Save Merged LoRA

Saves the optimizer's merged result as a standalone `.safetensors` file that works with any standard LoRA loader.

Connect the `LORA_DATA` output from LoRA Optimizer to this node.

| Option | Default | Effect |
|--------|---------|--------|
| `filename` | `merged_lora` | Plain name saves to your ComfyUI loras folder. Absolute path (e.g. `/path/to/my_lora`) saves to that location |
| `save_rank` | 0 (auto) | 0 = use each layer's existing rank from the merge. Non-zero = force this rank for layers that need compression |
| `bake_strength` | enabled | When on, the saved LoRA reproduces your exact merge at strength 1.0. When off, strengths are not baked in |

**Outputs:** `STRING` (file path)

## Installation

### ComfyUI Manager
Search for "LoRA Optimizer" in ComfyUI Manager and install.

### Manual
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ethanfel/ComfyUI-LoRA-Optimizer.git
```
Restart ComfyUI. Both nodes appear under the `loaders/lora` category.

## Compatibility

- **Models:** SD 1.5, SDXL, Flux, Z-Image (Lumina2), Wan 2.1/2.2, LTX Video, Qwen-Image, and other architectures supported by ComfyUI
- **LoRA formats:** Standard LoRA, LoCon, LyCORIS, diffusers/PEFT formats
- **Trainers:** Kohya, AI-Toolkit, LyCORIS, Musubi Tuner, diffusers — auto-normalized when `normalize_keys` is enabled
- **Flux sliced weights:** Handled correctly (linear1_qkv offsets)
- **Z-Image fused QKV:** Split for per-component analysis, re-fused after merge
- **Stack formats:** Native LoRA Stack dicts, plus standard tuples from Efficiency Nodes / Comfyroll

## Credits

- Originally based on [ComfyUI-ZImage-LoRA-Merger](https://github.com/DanrisiUA/ComfyUI-ZImage-LoRA-Merger) by DanrisiUA
- Per-prefix adaptive approach inspired by [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) by shootthesound (per-block LoRA analysis)
- Thanks to Scruffy and Ramonguthrie for suggesting the per-block analysis approach
- TIES-Merging: [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)
- DARE: [Yu et al., ICML 2024](https://arxiv.org/abs/2311.03099) — Drop And REscale for language model merging
- DELLA: [Deep et al., 2024](https://arxiv.org/abs/2406.11617) — magnitude-aware sparsification

## License

MIT License - see [LICENSE](LICENSE).
