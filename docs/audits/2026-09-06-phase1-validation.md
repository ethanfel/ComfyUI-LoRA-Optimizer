# Phase 1 correctness validation — 2026-09-06

Baseline: `a5e80a9`, package version **1.8.3**, left unchanged. Implementation is a working-tree change, not a release or authorization to publish. Phase 2 algorithm/performance investigations have not started.

## Audit disposition

| Audit finding | Disposition |
|---|---|
| 1: H3 QKV alias/re-fusion/export naming | Fixed using real `.weight` targets; partial slices zero-fill on native export. |
| 2: SDXL misdetected as H3 | Fixed: nested UNet keys and second text encoder checked before H3/ACE heuristics. |
| 3: PEFT `.default` inferred as interleaved | Fixed: explicit `auto` / `comfy` / `diffsynth` layout, ambiguous fused PEFT rejected. |
| 4: Dense `.diff` / `.diff_b` lost | Fixed: target grouping, shape checks, merge and export retain dense matrices/vectors. |
| 5: DoRA ignored | Safely rejected in ordinary merge/export; existing Inline barriers retain base-dependent original patches. |
| 6: LoCon middle tensor lost on export | Fixed and verified through the real stock loader. |
| 7: Finite fp32 factors overflow on fp16 export | Fixed: preserve factor dtype, validate finiteness, atomic replacement. |
| 8: Same-name in-memory payload reuses old merge | Fixed: tensor identity/mutation revisions, revision-aware content hashing, model/CLIP patch identity, layout/context-aware caches. |
| 9: Single-adapter filtering/export shortcut | Fixed: use the common pipeline and return LORA_DATA. The original probe used unsupported `attention_only`; the valid `audio_only` filter independently reproduces the defect and is the regression test. |
| 10: Auto-extraction ignores maximum rank | Fixed: enforce rank cap, log achieved retained energy, distinguish changed vectors/non-floating/shape/SVD omissions. |

Additional safety work: explicit H3 partition/basis conflicts, hard missing-target/full-pruned shape errors, duplicate normalized/export keys, PDD runtime rejection, STAR/taming forwarding through tuning/replays/settings, analysis/ranking cache schema invalidation, legacy dynamic-widget migration, and a test gate before registry publication.

Source architecture is checked per adapter; source layout is not inferred from the target model. Wrapper names are preserved as provenance, not promoted to verified partition identities. An optional target `lora_optimizer_h3_profile` with `partition`/`basis` fields can assert known compatibility; without it, base compatibility remains unknown. These labels are not cryptographic weight fingerprints.

## Reproducible checks

Local result: **612 Python tests passed, 3 skipped, 18 subtests passed**; **3 JavaScript migration tests passed**; isolated real-ComfyUI round trip passed. The GitHub workflow is configured, not remotely executed as part of this local change.

Run unit tests in the repository's Python environment:

```sh
python -m pytest -q
node tests/js/dynamic_migration.test.cjs
python tests/integration/comfy_roundtrip.py /path/to/ComfyUI
```

The integration script runs in a separate process, imports the installed ComfyUI and real safetensors, and does not inherit unit-test stubs. It uses small CPU modules with native H3 parameter names, actual ComfyUI key mapping, ModelPatcher registration, stock loading and weight application. Tested against ComfyUI commit `e6a51d7db7673e083e5f01f3740bd5515a8132a2`; CI pins that commit.

Round-trip checks cover partial QKV, signed strength and alpha/rank, dense AdaLN weight/bias and norm updates, signed CLIP weight/bias exports, Hook payload wiring, LoCon middle tensors, finite fp32 factors above fp16 range, and failed-save preservation of an existing file. Matrix comparisons use float32 absolute tolerances up to `3e-6` and relative tolerances up to `3e-5`. Unit regressions additionally exercise explicit interleaved conversion, cached/uncached merges, cleaning settings and Selector replay, architecture negatives and rank limits. JavaScript tests cover legacy 7/8/9-widget slots and new layout values.

Payload ownership: ordinary tensor replacement and in-place PyTorch operations invalidate caches. External mutation via `.data`, NumPy views, or another process is not a supported revision signal; callers must replace tensors or emit a new captured payload after such writes. File-reference loads are revision-checked before reuse.

## Validation limits and next checkpoint

Both `minimax_h3_fl2va_pruned_int8_convrot.safetensors` and `minimax_h3_ref2va_pruned_int8_convrot.safetensors` are present locally. The tested Python environment has PyTorch `2.11.0+cu130` but `torch.cuda.is_available()` is false; NVIDIA driver access also failed. No full-size checkpoint was loaded, model downloaded, service restarted, or render submitted. Full fixed-seed FL2VA/Ref2VA video/audio comparisons, particularly quantized/pruned perceptual equivalence, remain **unverified**.

H3 “support” therefore means the documented payload/layout checks and CPU patch round trips, not every training method, checkpoint combination, or audiovisual feature. Full-to-pruned conversion and PDD runtime interoperability remain Phase 2 decisions. Review this fix checkpoint before a version bump, commit/push, or improvement implementation.
