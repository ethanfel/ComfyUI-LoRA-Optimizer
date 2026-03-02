<p align="center">
  <img src="assets/banner.svg" alt="LoRA Optimizer" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Custom_Nodes-blue?style=flat-square" alt="ComfyUI">
  <img src="https://img.shields.io/badge/TIES_Merging-NeurIPS_2023-8b5cf6?style=flat-square" alt="TIES">
  <img src="https://img.shields.io/badge/Flux_%7C_SDXL_%7C_SD1.5-Compatible-22c55e?style=flat-square" alt="Compatible">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT">
</p>

---

A ComfyUI node that **automatically analyzes your LoRA stack** and selects the best merge strategy — diff-based merging, TIES conflict resolution, and auto-tuned parameters. Two nodes: **LoRA Stack** (build input) and **LoRA Optimizer** (analyze + merge).

## The Problem

Stacking LoRAs in ComfyUI adds their effects together. On distilled/turbo models (SDXL-Turbo, LCM, Lightning, Flux-schnell), the accumulated effect exceeds what the model can handle, causing **overexposure, color blowout, and artifacts**.

```
model += lora1_effect x strength1
model += lora2_effect x strength2
total effect = strength1 + strength2  -->  easily exceeds 1.0
```

The optimizer solves this by computing full weight diffs, detecting sign conflicts, and merging with the optimal strategy.

<p align="center">
  <img src="assets/comparison.png" alt="Before/After Comparison" width="100%">
</p>

## Nodes

### LoRA Stack

Builds a list of LoRAs for the optimizer. Chain multiple Stack nodes to add any number of LoRAs.

**Inputs:** LoRA selector, strength, optional previous `LORA_STACK`

**Outputs:** `LORA_STACK`

---

### LoRA Optimizer

The auto-optimizer. Takes a `LORA_STACK`, analyzes the LoRAs, and automatically selects the best merge mode and parameters. Outputs the merged result plus a detailed analysis report explaining what it chose and why.

Also accepts standard tuple-format stacks `(lora_name, model_strength, clip_strength)` from Efficiency Nodes, Comfyroll, and similar packs.

Uses a **two-pass streaming architecture** for low memory usage:
- **Pass 1 (Analysis):** Computes weight diffs per prefix, samples conflict and magnitude statistics, then discards the diffs. Only lightweight scalars are kept.
- **Pass 2 (Merge):** Recomputes diffs per prefix and merges with the auto-selected strategy. Each prefix is freed after merging.

Peak memory is ~one prefix at a time (~260MB) regardless of LoRA count or model size. GPU-accelerated on both passes.

<p align="center">
  <img src="assets/optimizer-pipeline.svg" alt="Optimizer Pipeline" width="100%">
</p>

**What it analyzes:**
- Per-LoRA metrics (rank, key count, effective L2 norms)
- Pairwise sign conflict ratios (sampled for efficiency)
- Magnitude distribution across all weight diffs
- Key overlap between LoRAs

**How it decides:**

| Condition | Decision |
|-----------|----------|
| Sign conflict > 25% | TIES mode (resolves conflicts) |
| Sign conflict <= 25% | weighted_average (simple, effective) |
| Magnitude ratio > 2x between LoRAs | `total` sign method (stronger LoRA gets more influence) |
| Magnitude ratio <= 2x | `frequency` sign method (equal votes) |
| TIES mode selected | Auto-density estimated from magnitude distribution |

**Per-prefix adaptive mode** (default): Each weight prefix picks its own strategy based on local conflict data. Non-overlapping prefixes use `weighted_sum` at full strength — preserving the full effect of each LoRA where they don't compete. Only genuinely conflicting prefixes use TIES. Set `optimization_mode` to `global` for the original single-strategy behavior.

**Inputs:** `MODEL`, `CLIP`, `LORA_STACK`, output strength, clip strength multiplier, auto strength, optimization mode, free VRAM between passes.

**Outputs:** `MODEL`, `CLIP`, `STRING` (analysis report)

#### TIES Merging

The optimizer can automatically select TIES-Merging (Trim, Elect Sign, Disjoint Merge — [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)) when sign conflicts are detected between LoRAs.

<p align="center">
  <img src="assets/ties-diagram.svg" alt="TIES Merging Pipeline" width="100%">
</p>

#### Auto-Strength

When `auto_strength` is set to `enabled`, the optimizer automatically reduces per-LoRA strengths before merging to prevent overexposure from stacking. This is especially useful on distilled/turbo models where 2+ LoRAs at full strength cause blown-out results even with optimal merge mode selection.

The algorithm uses **L2-aware energy normalization**: it measures each LoRA's actual weight magnitude (L2 norms collected during analysis) and scales all strengths so the total combined energy matches what the strongest single LoRA would contribute alone.

| Scenario | Result |
|----------|--------|
| 2 equal LoRAs at strength 1.0 | Each reduced to ~0.71 (1/sqrt(2)) |
| 1 strong + 1 weak LoRA | Proportional reduction, preserving ratio |
| Single LoRA | No change (scale = 1.0) |
| `auto_strength` disabled | No adjustment (default) |

Your original strength ratios are always preserved — the algorithm only scales them down uniformly.

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
  style_lora.safetensors: 1.0 -> 0.7071
  detail_lora.safetensors: 0.8 -> 0.5657
  Scale factor: 0.7071
  Method: L2-aware energy normalization

--- Pairwise Analysis ---
  style_lora.safetensors vs detail_lora.safetensors:
    Overlapping positions: 89420
    Sign conflicts: 31297 (35.0%)

--- Collection Statistics ---
  Total LoRAs: 2
  Total unique keys: 196
  Avg sign conflict ratio: 35.0%
  Magnitude ratio (max/min L2): 2.00x

--- Auto-Selected Parameters ---
  Merge mode: ties
  Density: 0.42
  Sign method: frequency

--- Per-Prefix Strategy ---
  weighted_sum (single LoRA):        28 prefixes (14%)
  weighted_average (low conflict):  120 prefixes (61%)
  ties (high conflict):              48 prefixes (24%)
  Total:                            196 prefixes

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

> **Limitation:** The optimizer only analyzes LoRAs in its own stack. It cannot see LoRA patches applied by upstream nodes (Load LoRA, etc.) — those stack additively on top of the optimizer's output. Fully baked merges (safetensors checkpoints) are indistinguishable from base weights and cannot be detected.

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

- **Models:** SD 1.5, SDXL, Flux, and other architectures supported by ComfyUI
- **LoRA formats:** Standard LoRA, LoCon, diffusers formats
- **Flux sliced weights:** Handled correctly (linear1_qkv offsets)
- **Stack formats:** Native LoRA Stack dicts, plus standard tuples from Efficiency Nodes / Comfyroll

## Credits

- Originally based on [ComfyUI-ZImage-LoRA-Merger](https://github.com/DanrisiUA/ComfyUI-ZImage-LoRA-Merger) by DanrisiUA
- TIES-Merging: [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)

## License

MIT License - see [LICENSE](LICENSE).
