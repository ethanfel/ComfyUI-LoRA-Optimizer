# ACE-Step 1.5 LoRA Merge Optimization

Technical reference for all architecture-specific choices made in ZImage LoRA Merger
for ACE-Step 1.5 music generation model LoRAs.

## ACE-Step 1.5 Architecture Overview

ACE-Step 1.5 is a 2B-parameter Diffusion Transformer (DiT) for music generation.
It uses a Qwen3-based architecture with 24 transformer blocks, each containing
self-attention, cross-attention, and MLP (SwiGLU) sub-modules.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| num_hidden_layers | 24 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 (GQA) |
| head_dim | 128 |
| intermediate_size | 6144 |
| sliding_window | 128 |
| text_hidden_dim | 1024 |
| timbre_hidden_dim | 64 |
| dtype | bfloat16 |

Source: [ACE-Step/Ace-Step1.5 config.json](https://huggingface.co/ACE-Step/Ace-Step1.5/blob/main/config.json)

### Hybrid Attention Pattern

Self-attention alternates between two modes across the 24 layers:

- **Odd layers (0, 2, 4, ..., 22)**: Sliding Window Attention (window=128).
  Captures local acoustic nuances — timbre texture, transients, articulation.
- **Even layers (1, 3, 5, ..., 23)**: Global Group Query Attention (GQA).
  Maintains long-range rhythmic and melodic consistency.

Cross-attention **always uses full/global attention** regardless of layer index.
This ensures voice identity from the timbre encoder has unrestricted access to
every position in the audio sequence.

Source: [ACE-Step 1.5 paper (arXiv:2602.00744)](https://arxiv.org/html/2602.00744v3),
[Architecture analysis (zenn.dev)](https://zenn.dev/asap/articles/6a717d7a68ec02?locale=en)

### Conditioning Stack

Three conditioning pathways are concatenated along the token axis and injected
via cross-attention into every DiT layer:

1. **Qwen3-0.6B caption embeddings** — text descriptions of genre, mood, style
2. **AceStepLyricEncoder** (8-layer bidirectional transformer) — lyric content
3. **AceStepTimbreEncoder** (4-layer bidirectional transformer) — voice/instrument
   timbre from reference audio (30s context window, 25Hz latent via 1D VAE)

The timbre encoder processes reference audio into a 2048-dimensional embedding.
When no reference audio is provided, 30 seconds of silence is used as input.
Non-padded tokens from all three sources are packed left before concatenation
to prevent padding artifacts in attention.

Source: [ACE-Step 1.5 model code](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/models/xl_turbo/modeling_acestep_v15_xl_turbo.py),
[ACE-Step 1.5 Explained (Substack)](https://artintech.substack.com/p/ace-step-15-explained)

### DiT Layer Structure (AceStepDiTLayer)

Each of the 24 layers contains:

```
AceStepDiTLayer:
  self_attn_norm  (Qwen3RMSNorm)
  self_attn       (AceStepAttention — sliding or global per layer_types)
  cross_attn_norm (Qwen3RMSNorm)
  cross_attn      (AceStepAttention — always global)
  mlp_norm        (Qwen3RMSNorm)
  mlp             (Qwen3MLP — SwiGLU: gate_proj, up_proj, down_proj)
  scale_shift_table (nn.Parameter, shape [1, 6, hidden_size] — AdaLN modulation from timestep)
```

The scale_shift_table provides 6 modulation values per layer:
`(shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa)`
used for adaptive layer norm on self-attention and MLP outputs.

Source: [ACE-Step 1.5 model code (xl_turbo)](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/models/xl_turbo/modeling_acestep_v15_xl_turbo.py)

### LoRA Key Structure (v1.5)

v1.5 uses PEFT-format LoRA adapters. Canonical key patterns after normalization:

| Module | Key Pattern | Shape (lora_A) | Shape (lora_B) |
|--------|-------------|-----------------|-----------------|
| Self-attention | `diffusion_model.layers.{0-23}.self_attn.{q,k,v,o}_proj` | [r, 2048] | q/o: [2048, r], k/v: [1024, r] |
| Cross-attention | `diffusion_model.layers.{0-23}.cross_attn.{q,k,v,o}_proj` | [r, 2048] | q/o: [2048, r], k/v: [1024, r] |
| MLP | `diffusion_model.layers.{0-23}.mlp.{gate,up,down}_proj` | gate/up: [r, 2048], down: [r, 6144] | gate/up: [6144, r], down: [2048, r] |

Note: k/v projections output 1024 (8 KV heads x 128 head_dim) due to GQA,
while q/o projections output 2048 (16 heads x 128 head_dim).

Community LoRAs commonly target all three module types (self_attn, cross_attn, mlp)
and may use DoRA with per-module rank patterns via Side-Step's Fisher Information
analysis.

Source: [HuggingFace community LoRAs](https://huggingface.co/6san/symphonic_metal_lora_for_ace-step_v15),
[Side-Step toolkit](https://github.com/koda-dernet/Side-Step)

## Voice-Critical Layer Analysis

When merging a vocal LoRA with a music/style LoRA, voice identity is at risk
of being diluted. Understanding which layers carry voice information guides
the merge strategy.

### Priority ranking for voice preservation

| Priority | Module type | Reasoning |
|----------|------------|-----------|
| **Highest** | `cross_attn` (all 24 layers) | Primary voice-identity pathway. Attends to timbre encoder output (voice characteristics) concatenated with text/lyric conditioning. Always uses full global attention — never sliding window. |
| **High** | `self_attn` in sliding window layers (odd: 0, 2, 4, ...) | Captures local acoustic details — timbre texture, vocal transients, articulation within 128-token windows. |
| **Medium** | `self_attn` in global layers (even: 1, 3, 5, ...) | Long-range structure — melody contour, rhythmic patterns, song form. More style than voice. |
| **Lower** | `mlp` (all layers) | General audio feature transformation. Less specific to voice identity. |

### Why cross-attention is critical

In ACE-Step 1.5, voice identity enters the model exclusively through
cross-attention. The timbre encoder produces a representation of the reference
voice, which is concatenated with text and lyric embeddings and attended to
by the DiT's cross-attention layers.

When a vocal LoRA is trained, its cross-attention weights learn to respond
strongly to the voice's timbre representation. When a music LoRA is trained
on instrumental data, its cross-attention weights learn to respond to
genre/style conditioning. These are typically orthogonal — they encode
different aspects of the conditioning signal.

Naive averaging or TIES trimming on cross-attention layers destroys the
vocal LoRA's learned response to timbre conditioning.

Source: [ACE-Step v1.0 paper (arXiv:2506.00045)](https://arxiv.org/abs/2506.00045),
[Voice cloning discussion (Issue #259)](https://github.com/ace-step/ACE-Step/issues/259),
[Architecture analysis (zenn.dev)](https://zenn.dev/asap/articles/6a717d7a68ec02?locale=en)

## Merge Optimization Decisions

### 1. Dedicated `acestep_dit` Architecture Preset

**Decision**: Create a separate preset instead of sharing the generic `dit` preset.

**Justification**: The generic `dit` preset is tuned for image/video diffusion
transformers (Flux, WAN, Z-Image, LTX) where all attention layers serve similar
functions. In ACE-Step, cross-attention has a fundamentally different role
(voice conditioning) that requires more conservative merge behavior.

**Preset values and their rationale**:

#### `ties_conflict_threshold`: 0.25 → 0.35

The TIES conflict threshold determines when the merger switches from
weighted_average/SLERP to TIES (trim-elect-merge) mode. TIES aggressively
trims low-magnitude values and resolves sign conflicts by majority vote.

For vocal + music LoRA merging, TIES trimming is destructive to voice:
- Cross-attention layers encode orthogonal concepts (voice timbre vs genre)
- TIES trims the minority-magnitude signal, which could be the voice
- Sign conflicts in cross-attn are expected (different conditioning responses)
  but don't represent real semantic conflict

Raising the threshold from 0.25 to 0.35 means TIES only activates for
genuinely high conflict (>35% excess), causing more prefixes to use
SLERP (magnitude-preserving interpolation) instead.

#### `orthogonal_cos_sim_max`: 0.25 → 0.30

This threshold determines when two LoRAs are classified as "orthogonal"
(independent signals). Orthogonal LoRAs get routed to weighted_average → SLERP
instead of TIES, since ~50% sign conflict is the statistical baseline for
independent vectors and doesn't represent real conflict.

Vocal and music LoRAs are typically orthogonal in most layers (cosine
similarity near zero). Widening the orthogonal band from 0.25 to 0.30
catches more of these cases, ensuring SLERP is used to preserve both
voice and music signals at full magnitude.

#### `orthogonal_conflict_max`: 0.60 → 0.65

The maximum conflict ratio allowed for the orthogonal classification.
Slightly raised to accommodate the naturally higher base-rate conflict
seen in audio LoRA pairs.

#### `dare_ideal_density`: 0.8 → 0.85

When DARE sparsification is used, this controls the target density
(fraction of parameters kept). Higher density preserves more signal,
which is important for voice quality — dropping 15% of voice-encoding
parameters is less destructive than dropping 20%.

#### `auto_strength_orthogonal_floor`: 0.85 → 1.0

When vocal and music LoRAs are orthogonal, auto-strength normally
attenuates the merge to prevent magnitude explosion. For ACE-Step,
full magnitude preservation (floor = 1.0) is critical because:
- Audio LoRAs encode energy/volume information in weight magnitude
- Attenuation reduces vocal presence in the generated output
- Orthogonal signals don't interfere, so no magnitude reduction is needed

This matches the behavior already applied to video architectures (WAN, LTX)
via `_VIDEO_ARCH_ORTHOGONAL_FLOOR`, where motion energy similarly requires
magnitude preservation.

Source for video precedent: existing `_VIDEO_ARCH_ORTHOGONAL_FLOOR` in codebase

### 2. Orthogonal Floor Override

**Decision**: Add ACE-Step to `_VIDEO_ARCH_ORTHOGONAL_FLOOR` with value 1.0.

**Justification**: Same reasoning as the preset floor. This is a secondary
enforcement point that catches edge cases where the preset floor might be
overridden by other logic. Ensures orthogonal vocal + music LoRA merges
always preserve full magnitude.

### 3. v1.0 Format Detection and Normalization

**Decision**: Support ACE-Step v1.0 LoRA format alongside v1.5.

**Justification**: v1.0 LoRAs use a completely different key structure
(`transformer_blocks.N.attn.to_q` vs v1.5's `layers.N.self_attn.q_proj`)
and include unique voice-critical keys (`speaker_embedder`, `lyric_encoder`)
that are absent in v1.5.

Without v1.0 support, these LoRAs were misdetected as WAN (because
`transformer_blocks.0.cross_attn` matches the WAN detector's `blocks.` check)
and would be incorrectly normalized.

#### v1.0 → v1.5 Key Mapping

| v1.0 pattern | v1.5 canonical form |
|-------------|-------------------|
| `transformer_blocks.N` | `layers.N` |
| `.attn.` (bare) | `.self_attn.` |
| `.to_q.` / `.to_k.` / `.to_v.` | `.q_proj.` / `.k_proj.` / `.v_proj.` |
| `.to_out.0.` | `.o_proj.` |
| `speaker_embedder` | `diffusion_model.speaker_embedder` |
| `lyric_encoder.encoders.N.self_attn.linear_q` | `diffusion_model.lyric_encoder.encoders.N.self_attn.q_proj` |

Source: [ACE-Step v1.0 repo](https://github.com/ace-step/ACE-Step),
[Chinese RAP LoRA](https://huggingface.co/ACE-Step/ACE-Step-v1-chinese-rap-LoRA)

#### v1.0 Voice-Critical Keys

- **`speaker_embedder`**: Directly encodes voice timbre as a 512-dim embedding
  projected to 2560-dim for cross-attention. Trained with 50% dropout in the
  base model, so the remaining signal is concentrated and critical.
  During merge, if only the vocal LoRA targets this prefix, it becomes a
  single-LoRA case → `weighted_sum` → full preservation.

- **`lyric_encoder`**: 6-block conformer encoding lyric content. Less
  voice-specific but encodes pronunciation/phonetic patterns that contribute
  to vocal character.

Source: [ACE-Step v1.0 paper (arXiv:2506.00045)](https://arxiv.org/html/2506.00045v1)

### 4. Bug Fix: `to_out` → `o_proj` Mapping

**Decision**: Fix the regex that mapped `to_out` → `out_proj` (wrong) to
correctly produce `o_proj`, and handle v1.0's `to_out.0.` suffix.

**Before (broken)**:
```python
re.sub(r"\.to_(q|k|v|out)\.", lambda m: f".{m.group(1)}_proj.", new_k)
# to_out → out_proj (WRONG — v1.5 uses o_proj)
# to_out.0 → not matched (the .0. after to_out breaks the pattern)
```

**After (fixed)**:
```python
re.sub(r"\.to_(q|k|v)\.", lambda m: f".{m.group(1)}_proj.", new_k)
re.sub(r"\.to_out\.0\.", ".o_proj.", new_k)
re.sub(r"\.to_out\.", ".o_proj.", new_k)
```

This bug would cause v1.0 LoRA output projection keys to not match v1.5
keys, breaking cross-LoRA prefix grouping when merging a v1.0 LoRA with
a v1.5 LoRA.

### 5. WAN Detection Refinement

**Decision**: Exclude `transformer_blocks` from the WAN architecture detector.

**Justification**: WAN uses bare `blocks.N` while ACE-Step v1.0 uses
`transformer_blocks.N`. The substring `blocks.` appears in both, causing
v1.0 ACE-Step LoRAs to be misdetected as WAN. The fix checks for
`transformer_blocks` and skips the WAN path.

## How the Existing Merge System Benefits ACE-Step

The per-prefix adaptive merge strategy already provides significant advantages
over the community's standard merge tool (`MERGE-LORA.py`), which performs
blind weighted-sum across all layers.

### Per-prefix strategy selection

For each LoRA key prefix (e.g., `layers.5.cross_attn.q_proj`), the merger:

1. Computes pairwise conflict metrics (sign conflict, cosine similarity,
   excess conflict, subspace overlap, magnitude ratio)
2. Classifies the relationship (orthogonal, aligned, conflicting)
3. Selects the optimal strategy:
   - **SLERP** for orthogonal signals (preserves both at full magnitude)
   - **Consensus** for aligned signals (Fisher-proxy importance weighting)
   - **TIES** for genuine conflicts (trim + elect sign + disjoint merge)
   - **Weighted sum** for single-LoRA prefixes

### Expected behavior for vocal + music LoRA merge

| Prefix type | Expected relationship | Expected strategy | Voice outcome |
|-------------|----------------------|-------------------|---------------|
| `cross_attn` | Orthogonal (voice vs genre conditioning) | SLERP | Both voice and genre preserved |
| `self_attn` (odd/sliding) | Low conflict (different local patterns) | SLERP or weighted_average | Timbre details preserved |
| `self_attn` (even/global) | Variable | Auto-selected | Structure blended |
| `mlp` | Low-medium conflict | SLERP or weighted_average | General features blended |
| `speaker_embedder` (v1.0) | Single-LoRA (only vocal targets it) | Weighted sum | Full voice preservation |

### Comparison with community MERGE-LORA.py

| Feature | MERGE-LORA.py | ZImage LoRA Merger |
|---------|---------------|-------------------|
| Merge method | Weighted sum only | TIES, SLERP, consensus, weighted avg, auto-select |
| Conflict handling | None (opposing signals cancel) | Per-prefix conflict analysis |
| Architecture awareness | None | ACE-Step detected, `acestep_dit` preset applied |
| Per-layer control | None (same weights everywhere) | Per-prefix adaptive strategy |
| Voice preservation | Manual strength tuning only | Automatic via orthogonal detection + SLERP |
| Magnitude preservation | Optional RMS clamp | Auto-strength with orthogonal floor = 1.0 |
| Rank handling | SVD truncation to min(r1, r2) | Configurable SVD compression |
| Multi-LoRA | 2 at a time (chain for 3+) | N-way native |

Source for MERGE-LORA.py: [DisturbingTheField HuggingFace repos](https://huggingface.co/DisturbingTheField/ACE-Step-v1.5-raspy-vocal-and-instrumental-5-LoRAs)

## Practical Recommendations for ACE-Step Users

### Merging ACE-Step LoRAs

1. Load LoRAs into a **LoRA Stack** node and connect to a **LoRA Optimizer** node
   (or use **LoRA AutoTuner** for automatic quality optimization)
2. Architecture preset will auto-detect as `acestep_dit`
3. Use `per_prefix` optimization mode (default)
4. Keep `normalize_keys` enabled (handles v1.0/v1.5/Kohya/PEFT format differences)
5. Set LoRA scale to 0.3–0.5 at inference for best results

This applies to all ACE-Step LoRA merges — vocal + music, genre + genre,
instrument + style, etc. The `acestep_dit` preset is tuned for preserving
distinct signals (voice, genre, instrument character) that would otherwise
be diluted by naive averaging.

### If voice quality degrades (vocal + music merges)

- Try `merge_strategy_override: slerp` to force SLERP on all prefixes
- Reduce the music LoRA's strength relative to the vocal LoRA
- Check the merge report — cross_attn prefixes should show SLERP, not TIES

### Training tips for better merge compatibility

These are from the ACE-Step community, not from our merge code:

- Don't isolate vocals before training — keep instrumentals in training data
- Use trigger words to tag the singer consistently
- Reduce `ssl_coeff` (mHuBERT loss weight) to 0.1 or 0 for stronger voice capture
- Use Side-Step's Fisher Information analysis for adaptive per-layer rank allocation

Source: [Voice cloning discussion (Issue #259)](https://github.com/ace-step/ACE-Step/issues/259),
[Side-Step toolkit](https://github.com/koda-dernet/Side-Step)

## Test Coverage

25 tests cover ACE-Step-specific functionality:

- **7 detection tests**: v1.0 (transformer_blocks, speaker_embedder, lyric_encoder)
  and v1.5 (PEFT, bare layers, cross_attn, MLP-only negative case)
- **12 normalization tests**: v1.5 PEFT prefix stripping, bare layers prefix,
  Kohya underscore conversion, MLP keys, v1.0→v1.5 mapping (transformer_blocks→layers,
  attn→self_attn, to_q→q_proj, to_out.0→o_proj, cross_attn preserved),
  speaker_embedder, lyric_encoder (linear_q/v→q_proj/v_proj), no double-prefixing
- **6 preset tests**: dedicated preset mapping, wider orthogonal band, higher
  TIES threshold, full magnitude preservation, auto-resolve, manual override

## References

- [ACE-Step 1.5 GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [ACE-Step 1.5 Paper (arXiv:2602.00744)](https://arxiv.org/html/2602.00744v3)
- [ACE-Step v1.0 Paper (arXiv:2506.00045)](https://arxiv.org/abs/2506.00045)
- [ACE-Step 1.5 config.json](https://huggingface.co/ACE-Step/Ace-Step1.5/blob/main/config.json)
- [ACE-Step 1.5 Model Code](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/models/xl_turbo/modeling_acestep_v15_xl_turbo.py)
- [ACE-Step 1.5 Explained (Substack)](https://artintech.substack.com/p/ace-step-15-explained)
- [ACE-Step Architecture Analysis (zenn.dev)](https://zenn.dev/asap/articles/6a717d7a68ec02?locale=en)
- [Voice Cloning Issue #259](https://github.com/ace-step/ACE-Step/issues/259)
- [Side-Step Training Toolkit](https://github.com/koda-dernet/Side-Step)
- [Community Merged LoRAs (DisturbingTheField)](https://huggingface.co/DisturbingTheField/ACE-Step-v1.5-raspy-vocal-and-instrumental-5-LoRAs)
- [Community Merged LoRAs (acoustic guitar)](https://huggingface.co/DisturbingTheField/ACE-Step-v1.5-acoustic-guitar-and-a-merge-LoRA)
- [LoRA Training Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md)
- [Musician's Guide Discussion](https://github.com/ace-step/ACE-Step-1.5/discussions/235)
- [LoRA Discussion/Library](https://github.com/ace-step/ACE-Step-1.5/discussions/338)
