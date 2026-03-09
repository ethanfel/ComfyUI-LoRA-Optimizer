# Workflows

Common workflow patterns for the LoRA Optimizer. Each example shows the node connections and explains when to use that pattern.

---

## Basic 2-LoRA Merge

The simplest workflow. Stack two LoRAs, let the optimizer handle everything.

```
Load Checkpoint ──► MODEL ──► LoRA Optimizer ──► MODEL ──► KSampler
                    CLIP ──►                 ──► CLIP  ──► CLIP Text Encode
                                 ▲
LoRA Stack (LoRA A) ─────────────┤
         ↓                       │
LoRA Stack (LoRA B) ─────────────┘
```

Or with the Dynamic variant (no chaining needed):

```
Load Checkpoint ──► MODEL ──► LoRA Optimizer ──► MODEL ──► KSampler
                    CLIP ──►                 ──► CLIP  ──► CLIP Text Encode
                                 ▲
LoRA Stack (Dynamic) ────────────┘
  [LoRA A, strength=1.0]
  [LoRA B, strength=0.8]
```

**When to use:** Most common case. Two style/character LoRAs that you want to blend cleanly.

---

## With Settings Nodes

Use Settings nodes to configure the optimizer without cluttering the main node.

```
LoRA Merge Settings ──► MERGE_SETTINGS ──► LoRA Optimizer Settings ──► OPTIMIZER_SETTINGS
                                                                              ↓
Load Checkpoint ──► MODEL ──► LoRA Optimizer ◄────────────────────────────────┘
                    CLIP ──►             (settings input)
                                 ▲
LoRA Stack ──────────────────────┘
```

Or chain all three tiers:

```
LoRA Merge Settings ──► MERGE_SETTINGS ──► LoRA AutoTuner Settings ──► OPTIMIZER_SETTINGS
                                                                              ↓
Load Checkpoint ──► MODEL ──► LoRA AutoTuner ◄────────────────────────────────┘
                    CLIP ──►
                                 ▲
LoRA Stack ──────────────────────┘
```

**When to use:** When you want fine-grained control (sparsification, compression, smoothing) but prefer a clean main node. Also useful when sharing the same merge settings between optimizer and tuner workflows.

---

## Compatibility Analyzer

Analyze your LoRA stack for compatibility before merging. Optionally auto-creates optimized node setups.

```
Load Checkpoint ──► MODEL ──► LoRA Compatibility Analyzer ──► report ──► Show Text
                    CLIP ──►                               ──► compatibility_map ──► Preview Image
                                 ▲
LoRA Stack (Dynamic) ────────────┘
  [LoRA A, LoRA B, LoRA C, LoRA D]
```

When `create_nodes=true` (default), the analyzer auto-creates nodes based on its grouping:
- **Solo LoRAs** (no meaningful overlap with others) → individual **LoraLoader** nodes
- **Merge groups** (compatible LoRAs) → **LoRA Stack (Dynamic)** nodes feeding an optimizer
- CLIP-aware loader selection when a CLIP model is connected

**When to use:** When you have many LoRAs and want to know which ones should be merged together vs loaded individually. Run the analyzer first, review the report and created nodes, then build your workflow from the suggestions.

---

## With Conflict Control

Manually analyze conflicts and override merge settings before optimizing.

```
LoRA Stack (LoRA A)
         ↓
LoRA Stack (LoRA B)
         ↓
LoRA Stack (LoRA C)
         ↓
LoRA Conflict Editor ──► LORA_STACK ──► LoRA Optimizer (Legacy) ──► MODEL
         │                                      ▲                 ──► CLIP
         │                                      │
         └── merge_strategy (STRING) ────────────┘
                                          (strategy override)
```

**When to use:** When you have 3+ LoRAs and want to understand which ones conflict, or when you want to force specific LoRAs to only contribute in low-conflict or high-conflict regions.

**Workflow:**
1. Connect your stack to the Conflict Editor
2. Read the analysis report — it shows pairwise conflict ratios
3. Adjust per-LoRA conflict modes if the auto-suggestions don't suit your needs
4. Optionally override the merge strategy (e.g., force `slerp` or `consensus`)
5. Connect both outputs to the optimizer

---

## AutoTuner (Rank Configs)

Let the AutoTuner sweep all parameter combinations and rank the strongest merges.

```
Load Checkpoint ──► MODEL ──► LoRA AutoTuner ──► MODEL ──► KSampler
                    CLIP ──►                 ──► CLIP
                                 ▲           ──► report ──► Show Text
LoRA Stack ──────────────────────┘           ──► TUNER_DATA
                                             ──► LORA_DATA
```

**When to use:** When you're not sure which settings are strongest for your stack, or when you want to compare different configurations systematically.

### Trying Alternatives with Merge Selector

```
LoRA AutoTuner ──► TUNER_DATA ──► Merge Selector (selection=2) ──► MODEL
                                                               ──► CLIP
                                                               ──► LORA_DATA
```

The AutoTuner ranks all configs by its composite score. Selection 1 = top-ranked, 2 = next-ranked, etc. Use the report to see what each config does, then switch between them with the Merge Selector.

---

## AutoTuner → Optimizer Bridge

Use the AutoTuner to rank configs, then hand the winning settings to a downstream Legacy Optimizer for manual tweaking without rewiring the graph. This workflow requires the **LoRA Optimizer (Legacy)** node.

```
Load Checkpoint ──► MODEL ──► LoRA AutoTuner ──► MODEL ──► LoRA Optimizer (Legacy) ──► MODEL ──► KSampler
                    CLIP ──►                 ──► CLIP  ──►                          ──► CLIP
                                 ▲                                    ▲
LoRA Stack ──────────────────────┘                                    │
                     TUNER_DATA ───────────────────────────────────────┘
```

**Switch behavior** (via the Legacy optimizer's `settings_source`):
- `settings_source = from_autotuner`
  - AutoTuner applies the top-ranked merge.
  - Legacy Optimizer becomes a passthrough and mirrors the winning settings in its widgets.
- `settings_source = manual`
  - AutoTuner passes the base model through.
  - Legacy Optimizer takes over using its own widget settings, starting from the AutoTuner recommendation.
- `settings_source = from_tuner_data`
  - Legacy Optimizer reads settings from the connected `tuner_data` input.

> **Note:** For new workflows, prefer using the **LoRA Optimizer** with a `tuner_data` connection from the AutoTuner instead of the bridge pattern.

**When to use:** When you want the AutoTuner to narrow the search space first, then manually tweak merge quality, sparsification, or smoothing from a strong starting point.

---

## Export Merged LoRA

Save the merge result as a standalone `.safetensors` file.

```
LoRA Optimizer ──► LORA_DATA ──► Save Merged LoRA ──► filepath (STRING)
```

The saved file works with any standard LoRA loader — no optimizer needed to use it. Set `bake_strength=enabled` so the saved LoRA reproduces your exact merge at strength 1.0. Choose the destination with `save_folder`; `filename` is relative to that folder and may include subdirectories.

**When to use:**
- Share your merge with others
- Use the merged LoRA in other tools
- Speed up workflows by pre-merging (no analysis overhead at generation time)

### Export from AutoTuner

```
LoRA AutoTuner ──► LORA_DATA ──► Save Merged LoRA
```

Or from a specific alternative:

```
LoRA AutoTuner ──► TUNER_DATA ──► Merge Selector (selection=3) ──► LORA_DATA ──► Save Merged LoRA
```

---

## Per-Prompt LoRA (Conditioning Hooks)

Apply the merged LoRA only to specific conditioning entries instead of globally to the model.

```
Load Checkpoint ──► MODEL ──────────────────────────────────────► KSampler
                    CLIP ──► CLIP Text Encode (positive) ──►     ▲
                                      ↓                          │
                              Cond Set Props ◄── HOOKS           │
                                      │              ↑           │
                                      ▼              │           │
                                 (positive in) ──────┤           │
                                                     │           │
LoRA Optimizer ──► LORA_DATA ──► Merged LoRA to Hook ┘           │
                                                                 │
                    CLIP ──► CLIP Text Encode (negative) ────────┘
```

**When to use:**
- Different merged LoRAs on positive vs negative conditioning
- Apply the LoRA only during certain sampling steps (with hook keyframes)
- Regional conditioning — LoRA on specific image regions
- Keep the base MODEL unpatched while still using the merge

The `prev_hooks` input allows chaining multiple hook sources:

```
Merged LoRA to Hook (merge A) ──► HOOKS ──► Merged LoRA to Hook (merge B) ──► HOOKS ──► Cond Set Props
```

---

## WanVideo Merge

Merging LoRAs for WanVideo models.

### Direct Optimization

```
WanVideoModelLoader ──► WANVIDEOMODEL ──► WanVideo LoRA Optimizer ──► WANVIDEOMODEL ──► WanVideoSampler
                                                    ▲
                             LoRA Stack ────────────┘
```

### Chaining with Individual LoRAs

Individual (non-merged) LoRAs go through the standard WanVideo path. The optimizer applies merged LoRAs on top.

```
WanVideoLoraSelect ──► WanVideoModelLoader ──► WANVIDEOMODEL ──► WanVideo LoRA Optimizer ──► Sampler
  (individual LoRA)                                                        ▲
                                                    LoRA Stack ────────────┘
                                                   (LoRAs to merge)
```

### Via LORA_DATA Bridge

If you prefer to use the standard LoRA Optimizer and then bridge the result:

```
LoRA Optimizer ──► LORA_DATA ──► Merged LoRA → WanVideo ──► WANVIDEOMODEL ──► Sampler
                                       ▲
WanVideoModelLoader ──► WANVIDEOMODEL ──┘
```

**Defaults for WanVideo:**
- `normalize_keys=enabled` — WanVideo LoRAs come from many trainers
- `cache_patches=disabled` — video models are large
- `architecture_preset=dit` — DiT-tuned thresholds

---

## Upstream LoRAs + Optimizer

Apply edit/distillation LoRAs separately, merge only style LoRAs with the optimizer.

```
Load Checkpoint ──► MODEL ──► Load LoRA (Turbo/LCM/DPO) ──► MODEL ──► LoRA Optimizer ──► MODEL
                    CLIP ──► Load LoRA                   ──► CLIP ──►               ──► CLIP
                                                                           ▲
                                                    LoRA Stack ────────────┘
                                                   (style/character LoRAs only)
```

**Important:** Never put distillation LoRAs (LCM, Lightning, Turbo, Hyper), DPO LoRAs, or edit model LoRAs (Qwen edit, Klein edit, instruction-editing LoRAs) in the optimizer stack. They modify the model's fundamental behavior and their weights are precisely calibrated. Apply them via standard Load LoRA nodes upstream.

---

## I2V / T2V Key Filtering

Use key filters to handle variant-specific keys when merging across video LoRA types.

### Make an I2V LoRA T2V-Compatible

```
LoRA Stack (T2V LoRA, key_filter=all)
         ↓
LoRA Stack (I2V LoRA, key_filter=shared_only)  ← strips I2V-only keys
         ↓
LoRA Optimizer ──► MODEL (T2V-compatible result)
```

The I2V LoRA's unique keys (`k_img`, `v_img`, `img_emb`, etc.) are skipped because `shared_only` only keeps keys present in 2+ LoRAs.

### Extract Lightweight I2V Adapter

```
LoRA Stack (T2V LoRA, key_filter=all)
         ↓
LoRA Stack (I2V LoRA, key_filter=unique_only)  ← keeps only I2V-specific keys
         ↓
LoRA Optimizer ──► LORA_DATA ──► Save Merged LoRA (small I2V adapter)
```

---

## Full Pipeline (Everything)

Combining multiple features in one workflow:

```
Load Checkpoint ──► MODEL ──► Load LoRA (Turbo) ──► MODEL
                    CLIP ──►                    ──► CLIP
                                                      │
LoRA Stack (Dynamic)                                  │
  [Style LoRA A]                                      │
  [Character LoRA B]                                  │
  [Detail LoRA C]                                     │
         ↓                                            │
LoRA Conflict Editor                                  │
         ↓                                            │
LoRA AutoTuner ◄──────────────────────────────────────┘
         │
         ├──► MODEL ──► KSampler ──► image
         ├──► CLIP
         ├──► report ──► Show Text
         ├──► TUNER_DATA ──► Merge Selector (try alternatives)
         └──► LORA_DATA ──► Save Merged LoRA (export best)
```
