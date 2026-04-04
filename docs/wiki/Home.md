# LoRA Optimizer Wiki

A ComfyUI node suite that **automatically analyzes your LoRA stack** and selects heuristic merge strategies per weight group. Instead of blindly stacking LoRAs (which causes oversaturation and sign conflicts), the optimizer examines where LoRAs overlap, how much they conflict, how aligned their subspaces are, and picks a local approach for each region independently.

---

## Quick Start

1. Add a **LoRA Stack** (or **LoRA Stack (Dynamic)**) node
2. Select your LoRAs and set strengths
3. Connect to **LoRA Optimizer** (optionally with Settings nodes for fine-grained control)
4. Connect `MODEL` and optionally `CLIP` from your checkpoint loader
5. Use the optimizer's `MODEL`/`CLIP` outputs for sampling

The optimizer handles everything automatically — conflict analysis, strategy selection, and merge execution. Connect the `STRING` output to a **Show Text** node to see the full analysis report.

---

## Wiki Pages

### Core Concepts

- **[[How It Works]]** — Full pipeline walkthrough from stack building through analysis, strategy selection, merge execution, and post-processing. Includes algorithm interaction diagrams and the composition order of all techniques.

### Reference

- **[[Nodes]]** — Detailed reference for every node: inputs, outputs, parameters, and usage notes.
- **[[Configuration Guide]]** — Every parameter explained with recommendations for common scenarios.

### Practical

- **[[Workflows]]** — Common workflow patterns with ASCII diagrams: basic merging, autotuning, conflict editing, WanVideo, per-prompt hooks, and exporting.
- **[[Tips and Troubleshooting]]** — Best practices, common pitfalls, what NOT to merge, and memory optimization.
- **[[Community Cache]]** — Share and reuse AutoTuner results across the community via Hugging Face. Skip sweeps entirely when others have already found the best config for your LoRAs.

### Background

- **[[Merge Algorithms]]** — Deep dive into each merge algorithm: the math, when each is selected, and how they interact with sparsification and quality enhancements.

---

## At a Glance

| Feature | Description |
|---------|-------------|
| **Per-prefix adaptive merge** | Each weight group gets its own strategy based on local conflict data |
| **5 merge algorithms** | Weighted sum, weighted average, TIES, SLERP, consensus |
| **Sparsification** | DARE and DELLA with standard and conflict-aware variants |
| **3 quality levels** | Standard, enhanced (DO + column-wise + TALL-masks), maximum (+ KnOTS SVD) |
| **Auto-strength** | Interference-aware energy normalization prevents oversaturation |
| **Architecture presets** | Tuned thresholds for UNet, DiT, and LLM architectures |
| **Key normalization** | Mix LoRAs from any trainer (Kohya, AI-Toolkit, LyCORIS, diffusers, etc.) |
| **SVD compression** | Re-compress merged patches to low-rank for ~32x RAM savings |
| **AutoTuner** | Sweep 2,000+ parameter combinations and rank them by internal metrics or an external evaluator |
| **Settings nodes** | Modular configuration: Merge Settings (shared), Optimizer Settings, AutoTuner Settings — optional, sensible defaults without them |
| **Compatibility analyzer** | Planning node that groups merge-safe LoRAs, surfaces conflicts, and optionally auto-creates optimized node setups |
| **Low memory** | Two-pass streaming architecture — peak memory scales with the largest active target group, not the full stack |
| **Community cache** | Share AutoTuner results via Hugging Face — skip sweeps when the best config is already known |
| **8 architectures** | SD 1.5, SDXL, FLUX, Z-Image, Wan, LTX Video, ACE-Step, Qwen-Image |

---

## Installation

### ComfyUI Manager
Search for **"LoRA Optimizer"** in ComfyUI Manager and install.

### Manual
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ethanfel/ComfyUI-LoRA-Optimizer.git
```
Restart ComfyUI. Nodes appear under the **loaders** category.
