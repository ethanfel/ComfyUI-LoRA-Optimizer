# Nodes

Complete reference for every node in the LoRA Optimizer suite.

---

## LoRA Stack

Builds a list of LoRAs for the optimizer. Chain multiple Stack nodes to add any number of LoRAs.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `lora_name` | COMBO | Yes | LoRA file selector (`.safetensors` from your loras folder) |
| `strength` | FLOAT | Yes | How much this LoRA contributes (default 1.0) |
| `conflict_mode` | COMBO | Yes | Where this LoRA's contributions apply: `all`, `low_conflict`, `high_conflict` |
| `key_filter` | COMBO | Yes | Which prefixes this LoRA contributes to: `all`, `shared_only`, `unique_only` |
| `lora_stack` | LORA_STACK | No | Previous stack to append to (for chaining) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `LORA_STACK` | LORA_STACK | The accumulated LoRA list |

---

## LoRA Stack (Dynamic)

Single node with 1–10 adjustable LoRA slots. Replaces chaining multiple Stack nodes.

### Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `lora_count` | 1–10 | Number of LoRA slots to show |
| `settings_visibility` | `simple`, `advanced` | Simple: one strength slider per LoRA. Advanced: separate model/clip strength, conflict mode, key filter |
| `input_mode` | `dropdown`, `text` | Dropdown: standard file picker. Text: type a name or path, auto-matched against installed LoRAs |
| `base_model_filter` | dynamic | Filters LoRA list by base model (requires [ComfyUI-Lora-Manager](https://github.com/hayden-fr/ComfyUI-Lora-Manager)) |

### Per-LoRA Inputs (Simple Mode)

| Input | Type | Description |
|-------|------|-------------|
| `lora_name_i` | COMBO/STRING | LoRA file |
| `strength_i` | FLOAT | Strength (applies to both model and CLIP) |

### Per-LoRA Inputs (Advanced Mode)

| Input | Type | Description |
|-------|------|-------------|
| `lora_name_i` | COMBO/STRING | LoRA file |
| `model_strength_i` | FLOAT | Model patch strength |
| `clip_strength_i` | FLOAT | CLIP patch strength |
| `conflict_mode_i` | COMBO | `all`, `low_conflict`, `high_conflict` |
| `key_filter_i` | COMBO | `all`, `shared_only`, `unique_only` |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `LORA_STACK` | LORA_STACK | The accumulated LoRA list |

Optional `lora_stack` input accepts a previous stack for chaining with other Stack nodes.

---

## LoRA Merge Settings

Shared configuration node for parameters common to both the optimizer and AutoTuner.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `normalize_keys` | COMBO | Yes | `enabled` | `enabled`, `disabled` — architecture-aware key normalization |
| `architecture_preset` | COMBO | Yes | `auto` | `auto`, `sd_unet`, `dit`, `llm` — numeric threshold tuning |
| `auto_strength_floor` | FLOAT | Yes | -1.0 | Minimum auto-strength scale factor for orthogonal LoRAs (`-1` = architecture default) |
| `decision_smoothing` | FLOAT | Yes | 0.25 | Smooth per-prefix decision metrics toward block averages (0 disables) |
| `smooth_slerp_gate` | BOOLEAN | Yes | `false` | Use per-prefix cosine similarity for SLERP gate instead of collection average |
| `vram_budget` | FLOAT | Yes | 0.0 | Fraction of free VRAM for keeping patches on GPU (0.0–1.0) |
| `cache_patches` | COMBO | Yes | `enabled` | `enabled`, `disabled` — keep merge in RAM for re-execution |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `merge_settings` | MERGE_SETTINGS | Shared settings for Optimizer Settings or AutoTuner Settings |

---

## LoRA Optimizer Settings

Optimizer-specific configuration. Accepts optional Merge Settings input.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `auto_strength` | COMBO | Yes | `enabled` | `enabled`, `disabled` — interference-aware energy normalization |
| `optimization_mode` | COMBO | Yes | `per_prefix` | `per_prefix`, `global`, `additive` |
| `merge_refinement` | COMBO | Yes | `none` | `none`, `refine`, `full` |
| `sparsification` | COMBO | Yes | `disabled` | `disabled`, `dare`, `della`, `dare_conflict`, `della_conflict` |
| `sparsification_density` | FLOAT | Yes | 0.7 | Fraction of parameters to keep (0.01–1.0) |
| `dare_dampening` | FLOAT | Yes | 0.0 | DAREx noise reduction (0–1.0) |
| `patch_compression` | COMBO | Yes | `smart` | `smart`, `aggressive`, `disabled` |
| `svd_device` | COMBO | Yes | `gpu` | `gpu`, `cpu` — device for SVD compression |
| `free_vram_between_passes` | COMBO | Yes | `disabled` | `enabled`, `disabled` — release GPU cache between passes |
| `strategy_set` | COMBO | Yes | `full` | `full`, `no_slerp`, `basic` — strategy selection logic |

### Optional Inputs

| Input | Type | Description |
|-------|------|-------------|
| `merge_settings` | MERGE_SETTINGS | Shared settings from LoRA Merge Settings node |
| `merge_strategy_override` | STRING | Force a specific merge strategy (connect from Conflict Editor) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `settings` | OPTIMIZER_SETTINGS | Connect to LoRA Optimizer's `settings` input |

---

## LoRA AutoTuner Settings

AutoTuner-specific configuration. Accepts optional Merge Settings input.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `top_n` | INT | Yes | 3 | Number of candidates to merge and score (1–10) |
| `scoring_svd` | COMBO | Yes | `disabled` | `enabled`, `disabled` — SVD-based effective rank scoring |
| `scoring_device` | COMBO | Yes | `gpu` | `cpu`, `gpu` — device for SVD scoring |
| `scoring_speed` | COMBO | Yes | `turbo` | `full`, `fast`, `turbo`, `turbo+` — subsample scoring |
| `scoring_formula` | COMBO | Yes | `v2` | `v2`, `v1` — scoring formula version |
| `diff_cache_mode` | COMBO | Yes | `auto` | `disabled`, `auto`, `ram`, `disk` |
| `diff_cache_ram_pct` | FLOAT | Yes | 0.5 | RAM fraction for auto diff cache (0.1–0.9) |
| `community_cache` | COMBO | Yes | `disabled` | `disabled`, `download_only`, `upload_and_download` — community HF cache |
| `memory_mode` | COMBO | Yes | `auto` | `disabled`, `auto`, `read_only`, `clear_and_run` — persistent tuning result cache across sessions |
| `selection` | INT | Yes | 1 | Which ranked config to replay from memory on a cache hit (1 = top-ranked) |

### Optional Inputs

| Input | Type | Description |
|-------|------|-------------|
| `merge_settings` | MERGE_SETTINGS | Shared settings from LoRA Merge Settings node |
| `evaluator` | AUTOTUNER_EVALUATOR | External evaluator for prompt/reference scoring |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `settings` | OPTIMIZER_SETTINGS | Connect to LoRA AutoTuner's settings input |

---

## LoRA Optimizer

The simplified auto-optimizer. Sensible defaults, minimal controls. Auto-strength is enabled by default. Accepts optional `settings` and `tuner_data` inputs for advanced control without exposing all parameters.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | — | Base model from checkpoint loader |
| `lora_stack` | LORA_STACK | Yes | — | Stack of LoRAs to merge |
| `output_strength` | FLOAT | Yes | 1.0 | Master strength multiplier (0–10) |
| `clip` | CLIP | No | — | Text encoder (if using CLIP LoRA keys) |
| `clip_strength_multiplier` | FLOAT | No | 1.0 | CLIP strength relative to model strength (0–10) |
| `tuner_data` | TUNER_DATA | No | — | AutoTuner result — applies the winning config's settings |
| `settings` | OPTIMIZER_SETTINGS | No | — | From LoRA Optimizer Settings or LoRA AutoTuner Settings |

When no `settings` node is connected, uses built-in defaults: `auto_strength=enabled`, `optimization_mode=per_prefix`, `merge_refinement=none`, `patch_compression=smart`, `vram_budget=0.0`.

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MODEL` | MODEL | Patched model ready for sampling |
| `CLIP` | CLIP | Patched text encoder |
| `analysis_report` | STRING | Analysis report with block strategy map |
| `tuner_data` | TUNER_DATA | For Merge Selector or Save Tuner Data |
| `LORA_DATA` | LORA_DATA | Merged patches for downstream nodes |

---

## LoRA Optimizer (Legacy)

> **Deprecated:** Superseded by **LoRA Optimizer** + **Settings nodes**. Use the Legacy variant only for the AutoTuner ↔ Optimizer bridge workflow (which requires `settings_source`).

Full-featured optimizer with all parameters on one node.

### Inputs

All inputs from the simple variant, plus:

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_strength` | COMBO | enabled | `enabled` / `disabled` — interference-aware energy normalization |
| `optimization_mode` | COMBO | per_prefix | `per_prefix`, `global`, `additive` |
| `merge_refinement` | COMBO | none | `none`, `refine`, `full` |
| `sparsification` | COMBO | disabled | `disabled`, `dare`, `della`, `dare_conflict`, `della_conflict` |
| `sparsification_density` | FLOAT | 0.7 | Fraction of parameters to keep (0.01–1.0) |
| `dare_dampening` | FLOAT | 0.0 | DAREx noise reduction (0–1.0, only affects DARE modes) |
| `patch_compression` | COMBO | smart | `smart`, `aggressive`, `disabled` |
| `svd_device` | COMBO | gpu | `gpu`, `cpu` — device for SVD compression |
| `cache_patches` | COMBO | enabled | `enabled`, `disabled` — keep merge in RAM for re-execution |
| `free_vram_between_passes` | COMBO | disabled | `enabled`, `disabled` — release GPU cache between passes |
| `normalize_keys` | COMBO | enabled | `enabled`, `disabled` — architecture-aware key normalization |
| `strategy_set` | COMBO | full | `full`, `no_slerp`, `basic` — strategy selection logic |
| `architecture_preset` | COMBO | auto | `auto`, `sd_unet`, `dit`, `llm` — numeric threshold tuning |
| `auto_strength_floor` | FLOAT | -1.0 | Minimum auto-strength scale factor for orthogonal LoRAs (`-1` = architecture default) |
| `decision_smoothing` | FLOAT | 0.25 | Smooth per-prefix decision metrics toward the surrounding block average (0 disables smoothing) |
| `smooth_slerp_gate` | BOOLEAN | false | Use per-prefix cosine similarity for SLERP gate instead of collection average |
| `vram_budget` | FLOAT | 0.0 | Fraction of free VRAM for keeping patches on GPU (0.0–1.0) |

### Optional Inputs

| Input | Type | Description |
|-------|------|-------------|
| `merge_strategy_override` | STRING | Force a specific merge strategy (connect from Conflict Editor) |
| `tuner_data` | TUNER_DATA | Optional AutoTuner result used when `settings_source=from_autotuner` or `from_tuner_data` |
| `settings_source` | COMBO | `manual`, `from_autotuner`, or `from_tuner_data`; controls whether widgets or AutoTuner config drive the node |

### Outputs

Same as the simple variant: `MODEL`, `CLIP`, `report` (STRING), `tuner_data` (TUNER_DATA), `LORA_DATA`.

---

## LoRA AutoTuner

Automated parameter sweep that ranks merge configurations.

### Inputs

All inputs from the Legacy optimizer, plus:

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `top_n` | INT | 3 | Number of candidates to actually merge and score (1–10) |
| `scoring_svd` | COMBO | disabled | `enabled`, `disabled` — SVD-based effective rank scoring |
| `scoring_device` | COMBO | gpu | `cpu`, `gpu` — device for SVD scoring (GPU is 10–50x faster) |
| `scoring_speed` | COMBO | turbo | `full`, `fast`, `turbo`, `turbo+` — subsample scoring for faster sweeps |
| `output_mode` | COMBO | merge | `merge`, `tuning_only` — output the top-ranked merged model, or pass the base model through so a downstream optimizer node applies the AutoTuner's settings |
| `auto_strength_floor` | FLOAT | -1.0 | Minimum auto-strength scale factor for orthogonal LoRAs |
| `decision_smoothing` | FLOAT | 0.25 | Same smoothing control as the optimizer; affects both ranking and final merge |
| `evaluator` | AUTOTUNER_EVALUATOR | — | Optional external evaluator hook for prompt/reference scoring |
| `community_cache` | COMBO | disabled | `disabled`, `download_only`, `upload_and_download` — community HF cache (no names shared, content hashes only) |
| `memory_mode` | COMBO | auto | `disabled`, `auto`, `read_only`, `clear_and_run` — persistent tuning result cache across sessions |
| `selection` | INT | 1 | Which ranked config to replay from memory on a cache hit (1 = top-ranked) |
| `cache_patches` | COMBO | enabled | Cache the final AutoTuner result in RAM for fast re-execution |
| `diff_cache_mode` | COMBO | auto | Diff cache mode across candidates: `disabled`, `auto`, `ram`, `disk` |
| `diff_cache_ram_pct` | FLOAT | 0.5 | RAM fraction used before `auto` diff cache spills to disk |
| `vram_budget` | FLOAT | 0.0 | Fraction of free VRAM for patch placement |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MODEL` | MODEL | Patched model from the top-ranked config |
| `CLIP` | CLIP | Patched text encoder |
| `report` | STRING | Ranked results with heuristic, internal, and optional external scores |
| `analysis_report` | STRING | Full optimizer analysis report for the top-ranked config |
| `TUNER_DATA` | TUNER_DATA | All ranked configs for Merge Selector |
| `LORA_DATA` | LORA_DATA | Merged patches from the top-ranked config |

Marked as `OUTPUT_NODE` for ComfyUI's re-execution optimization.

See [[How It Works#appendix-autotuner]] for details on the two-phase architecture and scoring.

---

## Merge Selector

Applies a specific ranked configuration from AutoTuner results.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | MODEL | Yes | — | Base model |
| `tuner_data` | TUNER_DATA | Yes | — | Output from LoRA AutoTuner |
| `selection` | INT | Yes | 1 | Which ranked config to apply (1 = top-ranked, 2 = next-ranked, etc.) |
| `clip` | CLIP | No | — | Text encoder |
| `clip_strength_multiplier` | FLOAT | No | 1.0 | CLIP strength multiplier when replaying the selected config |
| `auto_strength_floor` | FLOAT | No | -1.0 | Manual override for orthogonal auto-strength floor; `-1` reuses tuner setting |
| `decision_smoothing` | FLOAT | No | 0.25 | Smoothing used when replaying the selected config |

Validates that the LoRA stack hasn't changed since the AutoTuner ran (via hash comparison).

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `MODEL` | MODEL | Patched model from the selected config |
| `CLIP` | CLIP | Patched text encoder |
| `report` | STRING | Analysis report for the selected config |
| `LORA_DATA` | LORA_DATA | Merged patches from the selected config |

Marked as `OUTPUT_NODE`.

---

## LoRA Conflict Editor

Interactive conflict analysis with per-LoRA override controls.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | MODEL | Yes | Base model |
| `lora_stack` | LORA_STACK | Yes | Stack to analyze |
| `merge_strategy` | COMBO | Yes | `auto`, `ties`, `consensus`, `slerp`, `weighted_average`, `weighted_sum` |

Per-LoRA conflict mode overrides appear dynamically based on the stack.

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `LORA_STACK` | LORA_STACK | Enriched stack with conflict mode settings |
| `analysis_report` | STRING | Pairwise conflict analysis |
| `merge_strategy` | STRING | Strategy override string (connect to optimizer's `merge_strategy_override`) |

### How It Works

1. Loads all LoRAs, computes full-rank diffs per prefix
2. Measures pairwise sign conflict ratios
3. Auto-suggests per-LoRA conflict modes:
   - < 15% average conflict: `all`
   - 15–40%: `low_conflict`
   - \> 40%: `high_conflict`
4. Allows manual override of each LoRA's conflict mode and the merge strategy
5. Outputs feed directly into the optimizer

---

## Save Merged LoRA

Exports merged patches as a standalone `.safetensors` file usable with any standard LoRA loader.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `lora_data` | LORA_DATA | Yes | — | From Optimizer, AutoTuner, or Merge Selector |
| `save_folder` | COMBO | Yes | first configured LoRA folder | Which configured ComfyUI LoRA directory to save into |
| `filename` | STRING | Yes | `merged_lora` | File name relative to `save_folder`. Subdirectories allowed |
| `save_rank` | INT | Yes | 0 | 0 = use existing layer ranks. Non-zero = force this rank via SVD |
| `bake_strength` | COMBO | Yes | enabled | Bake `output_strength` so the saved LoRA works at strength 1.0 |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `filepath` | STRING | Full path to the saved `.safetensors` file |

---

## Build AutoTuner Python Evaluator

Builds an `AUTOTUNER_EVALUATOR` object from a Python module path + callable name.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `module_path` | STRING | Yes | — | Python file path or importable module |
| `callable_name` | STRING | Yes | `evaluate_candidate` | Callable that returns a float score or `{score, details}` |
| `combine_mode` | COMBO | No | blend | `blend`, `external_only`, `multiply` |
| `weight` | FLOAT | No | 0.5 | Blend weight when `combine_mode=blend` |
| `context_json` | STRING | No | `{}` | JSON passed through as `context` to the evaluator |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `evaluator` | AUTOTUNER_EVALUATOR | External evaluator spec for LoRA AutoTuner |

---

## Save / Load Tuner Data

Persist AutoTuner rankings for reuse across optimizer runs.

### Save Tuner Data Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tuner_data` | TUNER_DATA | Yes | — | Ranked AutoTuner results |
| `save_folder` | COMBO | Yes | first configured tuner_data folder | Which configured `tuner_data` directory to save into |
| `filename` | STRING | Yes | `tuner_data` | File name under `save_folder`. Subdirectories allowed; `.tuner` is added automatically unless `.json`/`.tuner` is supplied |
| `overwrite` | BOOLEAN | Yes | `true` | Overwrite an existing file or append `_001`, `_002`, … to avoid clobbering |

### Load Tuner Data Outputs

| Output | Type | Description |
|--------|------|-------------|
| `tuner_data` | TUNER_DATA | Loaded AutoTuner results ready for Merge Selector |

---

## LoRA Compatibility Analyzer

Pre-merge planning node that analyzes overlap, cosine similarity, and conflicts without applying a merge.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | BOOLEAN | Yes | `false` | Run the analyzer only when explicitly enabled |
| `create_nodes` | BOOLEAN | Yes | `true` | When enabled, auto-creates LoraLoader nodes for solo LoRAs and LoRA Stack (Dynamic) nodes for merge groups based on the compatibility analysis |
| `model` | MODEL | Yes | — | Base model used for target grouping and key mapping |
| `lora_stack` | LORA_STACK | Yes | — | Stack to analyze |
| `clip` | CLIP | No | — | Optional CLIP model for text-encoder LoRA keys (also used for CLIP-aware loader selection) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `report` | STRING | Human-readable compatibility report with suggested groups |
| `compatibility_map` | IMAGE | Heatmap of pairwise compatibility values |

---

## Merged LoRA to Hook

Wraps merged patches as conditioning hooks for per-prompt LoRA application.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `lora_data` | LORA_DATA | Yes | From Optimizer, AutoTuner, or Merge Selector |
| `prev_hooks` | HOOKS | No | Chain with existing hooks |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `HOOKS` | HOOKS | Conditioning hooks for use with Cond Set Props |

### Use Cases

- **Per-prompt LoRA** — different merged LoRAs on positive vs negative conditioning
- **Scheduled application** — combine with hook keyframes for step-specific LoRA
- **Regional conditioning** — apply LoRA to specific image regions
- **Preserve base model** — keep MODEL unpatched, apply LoRA only through conditioning

---

## WanVideo LoRA Optimizer (WIP)

Optimizer variant for WanVideo models via [kijai's WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper).

### Differences from Standard Optimizer

| Aspect | Standard | WanVideo |
|--------|----------|----------|
| Model input | `MODEL` | `WANVIDEOMODEL` |
| CLIP | Supported | Not used |
| `normalize_keys` default | enabled | **enabled** |
| `cache_patches` default | enabled | **disabled** |
| `architecture_preset` default | auto | **dit** |

All merge algorithms work identically — TIES, DARE/DELLA, SVD compression, auto-strength, quality enhancements, key normalization.

### Inputs/Outputs

Same as the Legacy optimizer, but with `WANVIDEOMODEL` replacing `MODEL` and no CLIP inputs/outputs. Outputs `LORA_DATA` for Save Merged LoRA.

---

## Merged LoRA → WanVideo (WIP)

Bridges `LORA_DATA` to a WanVideo wrapper model.

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `wan_model` | WANVIDEOMODEL | Yes | WanVideo model to patch |
| `lora_data` | LORA_DATA | No | Merged patches to apply |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `WANVIDEOMODEL` | WANVIDEOMODEL | Patched WanVideo model |

Handles the `_orig_mod.` key prefix mismatch from `torch.compile` automatically.
