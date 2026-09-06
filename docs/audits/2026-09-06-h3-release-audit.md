# Node audit and MiniMax H3 research — 2026-09-06

Historical baseline audit of `a5e80a9`. The subsequent fixes and current validation limits are recorded in [Phase 1 validation](2026-09-06-phase1-validation.md). The original diagnostic script targets that baseline; use the regression suite for the corrected code. Probe 9 used an unsupported filter spelling; the valid `audio_only` regression independently confirms and covers the single-adapter defect.

## Verdict

**Hold the version bump.** H3 support is not yet reliable across the advertised formats. There are reproducible weight-mapping and conversion defects, including a regression affecting non-H3 models. The next release should prioritize correctness and explicit compatibility checks over additional merge algorithms.

Audited repository: `a5e80a9` (`1.8.3`). Local ComfyUI reference: `e6a51d7d`, dated September 4, 2026. Current public upstream sources were also checked. This audit does not modify production code, bump the version, commit, or push.

## Validation and limits

- Existing suite: `python -m pytest -q` — **582 passed, 3 skipped, 7 subtests passed**.
- Added a diagnostic harness: [h3_release_probes.py](h3_release_probes.py). Run from the repository root:

  ```bash
  python docs/audits/h3_release_probes.py /media/p5/Comfyui
  ```

- Probes use small CPU tensors and the repository's ComfyUI test stubs. The mapping probe executes the actual generic-key loop extracted from the installed ComfyUI source. Other probes call the repository's real conversion, merge-dispatch, cache, extraction, and export functions. Export calls capture tensors instead of writing model files.
- Inspected public safetensors **headers only**, not full weights, for newer FastH3 and converted PDD adapters. No large model download or render was performed.
- Thus, the findings establish code/format failures, not measured video-quality differences. Quantized runtime application, full save/reload integration, and H3 audiovisual generation still require integration testing.

## Confirmed findings

### 1. [P1] Normalized H3 attention never resolves against native ComfyUI targets

Locations: `lora_optimizer.py:2517`, `:2412`, `:9423`; tests around `tests/test_lora_optimizer.py:3497`.

ComfyUI maps a module prefix such as `diffusion_model.blocks.0.attn.qkv_proj` to the parameter **`diffusion_model.blocks.0.attn.qkv_proj.weight`**. All three new H3 regular expressions assume the target ends at `qkv_proj`. Consequently, the split-QKV alias helper adds no aliases, normalized `to_q/to_k/to_v` entries are dropped, and the re-fusion/reverse-map code has the same suffix defect. The existing H3 tests use invented targets without `.weight`, masking this integration error. See [ComfyUI's mapper](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/lora.py).

Probe: three normalized attention prefixes, **zero resolved groups**; three valid `.weight` slice patches remain three slices instead of becoming a native fused patch.

Affected: normalized H3 file stacks, including Simple's default `normalize_keys="enabled"`. This does **not** mean that an already-native file with normalization disabled necessarily fails.

Fix direction: distinguish module aliases from parameter targets throughout mapping, re-fusion, and export. Test actual `.weight` state-dict targets; also test Q-only/Q+V adapters and filtered components. Partial slices need zero-filled native export or explicit rejection, not unsupported pseudo-key output.

### 2. [P1] H3 detection misclassifies ordinary SDXL/UNet feed-forward keys

Location: `lora_optimizer.py:1156`.

The unanchored `transformer_blocks.*.ff.net.0.proj` / `ff.net.2` check is not H3-specific. It matches ordinary nested UNet attention blocks before SDXL or other architecture checks run.

Probe: an SDXL-style dictionary containing both `unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj` and a `text_encoder_2` key is classified as `minimax_h3`. Normalization then changes the UNet feed-forward module to `...mlp.fc1`, removes its wrapper, and swaps the B-factor halves. This is the wrong target/layout for that model.

Fix direction: exclude nested UNet blocks, use architecture-specific roots and dimensions, and cross-check the supplied model class. Add negative fixtures for SD1.5, SDXL, LTX, ACE-Step, Qwen, and partial adapters. An architecture *preset* override is not a substitute for fixing key-format detection.

### 3. [P1] PEFT's `default` name is no longer a valid H3 QKV-layout discriminator

Location: `lora_optimizer.py:2117` and `:2251`.

Any native `qkv_proj.lora_A.default.weight` causes an interleaved-to-contiguous permutation. However, DiffSynth now trains directly against ComfyUI-compatible H3 models, whose attention reads contiguous `[Q; K; V]`; its training module still uses PEFT's default adapter naming. Sources: [Comfy-layout attention](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/minimax_h3_dit_comfy.py), [pruned training recipe](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.sh), [training/export implementation](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/diffusion/training_module.py).

Probe: identical contiguous factors produce different deltas after merely adding `.default` to their names. Q rows expected to be `128..131` become `384..387`.

Fix direction: explicit source-layout profiles, supported producer metadata, and an override for ambiguous files. Tensor shape and `.default` alone cannot identify the permutation. Preserve this profile when caching or exporting.

### 4. [P1] File-based merging loses dense weight and bias patches used by newer H3 adapters

Locations: `lora_optimizer.py:2810`, `:3185`, `:14781`.

The prefix collector and file parser accept LoRA/LoHa/LoKr factors but do not ingest `.diff` and `.diff_b` payloads. A dictionary consisting of those keys produces zero prefixes. If an adapter also contains ordinary factors, those factors can merge while the dense updates disappear. Export similarly lacks a lossless dense-vector/bias path.

This is a current compatibility problem, not a hypothetical format. The [FastH3 extraction card](https://huggingface.co/drozbay/MiniMax-H3-FastH3-Preview-LoRA) describes dense norms, biases, and pruned AdaLN updates. Header inspection of its pruned rank-64 file at revision `4f95050e9f28a761ed8408654de1d808f715eb07` found **547 tensors: 426 factor tensors, 65 `.diff`, and 56 `.diff_b`**. One AdaLN weight diff is `[96768, 8]` and its bias diff is `[96768]`.

Fix direction: carry dense matrices and vectors as first-class additive patches through collection, analysis, merging, filtering, save, and reload. Until then, detect these files and reject incomplete merges explicitly. Do not report full FastH3/pruned-adapter support.

### 5. [P1] File-mode DoRA loses its magnitude component

Locations: `lora_optimizer.py:3337`, `:3524`; compare inline safeguards at `:2560` and `:4004`.

The file parser returns only up/down/alpha/mid and ignores `dora_scale`. The ordinary merge path consequently treats DoRA as additive `BA`. Inline capture already recognizes that DoRA is base-dependent and refuses this conversion, so behavior differs by entry path.

Probe: with identity base weights and zero DoRA magnitudes, the correct final weight is zero, hence delta `-I`. The file parser instead computes an all-ones delta from the factors. This concerns the diff-based file merge path; the single-file stock-loader shortcut can handle DoRA separately. Compare [ComfyUI's weight decomposition](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/weight_adapter/base.py).

Fix direction: reject unsupported base-dependent file payloads, preserve their original application semantics, or explicitly materialize them against a verified base and ordered patch chain. Do not silently discard magnitude tensors.

### 6. [P2] LoCon export omits the middle convolution tensor

Location: `lora_optimizer.py:14801` and `:14849`.

`SaveMergedLoRA` reads `mid` from a LoRAAdapter, deliberately skips its compression, then writes only up/down/alpha. It never writes `.lora_mid.weight`.

Probe: a valid LoCon with a 3×3 middle kernel has a flattened delta of shape `[2,18]`; the exported factors reconstruct only `[2,2]`.

Fix direction: serialize the middle tensor, or fold it into a supported representation before export. Require a stock-loader round-trip test for convolution adapters.

### 7. [P2] Float32 export can create infinities from a finite adapter

Locations: `lora_optimizer.py:14754`, `:14849`, `:14873`.

When all patches are float32/float64, export defaults to float16. A valid factorization can have large factors but a small product. The validation checks NaNs, not all non-finite values.

Probe: float32 `B=100000`, `A=0.00001` yields a finite delta near one. Export casts B to float16 infinity and still hands the result to the writer.

Fix direction: expose output dtype or choose it conservatively; check `isfinite` after every conversion and abort unsafe saves. Optional factor balancing may improve representability, but must preserve the product and alpha.

### 8. [P2] Merge-cache identity ignores changed in-memory tensors

Locations: `lora_optimizer.py:6342`, `:8369`; extractor naming at `:16600`.

The instance cache hashes names, strengths, flags, and settings, not the tensor payload. Extracted adapters always use the same synthetic name. Re-extracting different weights into an otherwise identical stack can therefore replay the previous merge on the same model object.

Probe: replacing a tensor leaves the cache key unchanged; an `optimize_merge` call with the changed payload retrieves a seeded stale cached result. This is distinct from the documented name-based persistent-memory policy.

Fix direction: give in-memory outputs content/revision identities and include them in both execution and instance-cache keys. Reuse a memoized identity instead of hashing gigabytes on every candidate. Add a re-extraction regression test.

### 9. [P2] Single-LoRA shortcut ignores filters and produces no export data

Location: `lora_optimizer.py:8339`.

For non-H3/non-ZImage file entries, the shortcut forwards the entire dictionary to the stock loader without evaluating `key_filter` or other layer transformations. It returns `lora_data=None`.

Probe: a Wan-style LoRA requested with `attention_only` still forwards its FFN factors, and produces no data for Save/Hook consumers. Settings can therefore change behavior when a second LoRA is removed or its strength becomes zero.

Fix direction: use the shortcut only when no filtering/transformation/output-data requirements exist, or construct filtered patches and consistent LORA_DATA even for one adapter.

### 10. [P2] Extraction's auto rank ignores its documented maximum

Locations: `lora_optimizer.py:16405`, `:6113`.

The node describes `rank` as an upper bound in auto mode. `_extract_lora_svd` caps the selected rank only by the available singular values, not the requested limit.

Probe: identity delta `[8,8]`, `rank=1`, `rank_mode="auto"`, energy `0.99` returns **rank 8**. Large H3 full-finetune deltas could unexpectedly allocate/export near-full-rank factors.

Fix direction: either honor the maximum and report achieved reconstruction energy, or change the control/contract explicitly. Report bias/norm omissions separately from zero deltas; the current extractor also skips 1-D differences.

## Additional source-audit concerns

These are grounded in inspected code, but were not exercised through complete runtime workflows in this audit.

- **AutoTuner settings propagation:** `star_eta`, `tame_layers`, and `tame_threshold` enter `auto_tune`, but are absent from its ordinary candidate call (`:13598`), final selected-config merge (`:14025`), and instance-cache key (`:12720`). Some early replay/single paths do forward them. Audit fresh sweep versus cached replay consistency and include these settings in all dependent cache identities.
- **Pruned/full and partition compatibility:** a full-width AdaLN delta is skipped against an 8-wide pruned target; the current mismatch report records only the output dimension, which can be identical in this case. Record complete shapes. The `transformer`/`transformer_ref` prefix check is not a fingerprint of the loaded base, and prefix stripping loses provenance. Carry partition, layout, and basis identity explicitly.
- **Cache schema versioning:** H3 math changes bumped `ANALYSIS_CACHE_VERSION`, while stored tuner rankings use separate `AUTOTUNER_ALGO_VERSION`. Invalidate affected rankings as well as raw metrics, and account for model/basis and normalization profiles where relevant.
- **Coverage reporting:** unmapped prefixes are logged but not returned as structured coverage data; unsupported tensor types may never reach that warning. Report expected/applied/skipped counts per adapter and tensor kind. Add strict mode, especially for distillation adapters.
- **Release automation:** `.github/workflows/publish.yml` publishes on a `pyproject.toml` change without running tests in that workflow. Add a required test job, including real-Comfy mapping/save-reload fixtures, before publication.

## What changed upstream, and how H3 should be merged

### Full-to-pruned AdaLN conversion is now practical

Pruning replaces the full timestep embedding with an 8-dimensional curve basis. Renaming keys or reshaping factors cannot bridge those coordinate systems. Current implementations preserve both the projected weight update and a constant bias update. [The pruned-model implementation change](https://huggingface.co/multimodalart/MiniMax-H3-Pruned/commit/d3ff0e3a7c2e394e54b1055f1a2bb28ea9c8287f) supplies affine maps per partition; [the PDD implementation](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc#pruned-checkpoints) fits/checks the corresponding curve basis.

Writing the full input as `x(t) ≈ c + V z(t)`, an ordinary adapter contributes:

```text
s B A x(t) ≈ (s B A V) z(t) + s B A c
```

Therefore both the projected factors and the constant offset are required. Use a matching partition/basis, preserve alpha/rank scaling, and measure fitting residual over the timestep grid. This does not automatically handle time-embedder modifications or arbitrary DoRA; those need separate treatment. A rank-8 shape match alone is insufficient because different bases can use different coordinates.

### PDD adapters are not standalone LoRAs

[Alibaba PAI's release](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) uses Parallel Decoding Distillation. Its [ComfyUI integration](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc) applies a trunk adapter plus interval-dependent output heads on a trained schedule. An ordinary LoRA merge cannot preserve the latter.

Header inspection of the [converted FL2VA file](https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI/blob/main/minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors), revision `acd4775bfc614a38c764729399632de2f423e1e7`, found 258 LoRA modules plus four head-bank tensors: `proj_out.weight/bias` and `audio_proj_out.weight/bias`, each with 32 intervals. The current collector ignores those heads.

Recommendation: detect the PDD metadata/tensors and reject ordinary merge/export, or preserve them via an explicitly supported companion runtime. Do not promise that `preserve=True` makes such a bundle a complete standalone LoRA.

### Keep exact additive merging as the fidelity baseline

For ordinary adapters on a verified common base, the reference is:

```text
delta = sum_i strength_i * (alpha_i / rank_i) * B_i A_i
```

Use exact factor concatenation when possible; do not independently average A and B because their product introduces cross-terms. The repository already has an exact-linear path worth retaining. [PEFT's merging explanation](https://huggingface.co/blog/peft_merging) describes the distinction.

For ordinary Turbo/distillation LoRAs, validate additive application first, with automatic strength, sparsification, spectral cleaning, and lossy compression disabled. `preserve` protects an overlay from conflict merging, but does not supply missing dense payloads, fix a wrong base/layout, or carry external runtime heads. For style/character combinations, compare conflict-aware modes against this baseline rather than treating a weight-space score as a measured video-quality result.

## Newer merging research worth considering

| Method | Evidence and requirements | Recommendation for these nodes |
|---|---|---|
| **SSR-Merge**, June 2026, ICML | Constructs a joint low-rank space and decorrelates/routes signals. The released code needs a prompt per LoRA and a calibration pass, then exports a conventional adapter. Its listed backbones are Flux, Qwen, Z-Image, HiDream, and Flux2—not H3. [Paper](https://arxiv.org/abs/2606.10617), [implementation](https://github.com/nagara214/SSR-Merge). | My strongest candidate for a later experimental calibration node. It is not a drop-in data-free merge mode, and H3/video/audio evaluation is still needed. |
| **CT-Merging**, July 2026 | Consensus subspace directions with task-level RMS scaling; reported results concern CLIP adapter benchmarks. [Paper](https://arxiv.org/abs/2607.20561). | Worth a bounded weight-space prototype after correctness fixes. No demonstrated H3 advantage in the cited evidence. |
| **TARA-Merging**, CVPR 2026 | Preserves task subspaces while reweighting directions through a preference-weighted pseudo-loss; evaluated on vision and NLI tasks. [Paper](https://arxiv.org/abs/2603.26299). | Research candidate, not an H3 default. Requires a meaningful preference/calibration objective for generative audiovisual output. |

There is no basis in these sources to claim a universally superior H3 merge algorithm. The directly actionable upstream advance is **basis-aware pruned AdaLN conversion with bias preservation**, not replacing addition with a newer acronym.

## Recommended release order

1. Fix H3 `.weight` mapping and architecture false positives. Replace synthetic-target tests with native mapping fixtures.
2. Introduce layout/base/basis compatibility profiles; support dense patches or reject affected files. Detect PDD bundles. Make partial application visible and optionally fatal.
3. Fix DoRA file handling, LoCon/dtype export, stale in-memory cache identity, single-item behavior, extraction rank limits, and AutoTuner option propagation. Version all affected caches.
4. Gate release on load → normalize → merge → save → stock reload equivalence, including mixed ranks/alphas, native versus Diffusers/PEFT layouts, missing Q/K/V branches, dense biases, and quantized/pruned bases.
5. Run fixed-seed H3 A/B renders on matching FL2VA and Ref2VA bases: native baseline versus merged/reloaded adapter; evaluate motion, identity, prompt adherence, audio content, and synchronization. Then make a small version bump with a precise supported-format matrix.

Later performance work: compress exact concatenated factors using QR plus a small-core SVD rather than first materializing large dense H3 matrices; expose measured reconstruction error and explicit dtype/rank budgets. This is an implementation improvement proposal, not a benchmarked speed claim from this audit.
