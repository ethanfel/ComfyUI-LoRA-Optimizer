# H3 and node correctness plan, followed by improvement review

Status: Phase 1 implemented and CPU-validated; full-model render checks remain pending a usable GPU. Phase 2 has not started.

Results and limitations: [Phase 1 validation record](../audits/2026-09-06-phase1-validation.md). Package version remains 1.8.3; no commit, push, or publication in this phase.

Baseline: repository `a5e80a9`, version `1.8.3`. Evidence: [September 6 audit](../audits/2026-09-06-h3-release-audit.md) and [diagnostic probes](../audits/h3_release_probes.py).

## Sequence and scope

1. Fix and verify the correctness defects.
2. Review the improvements individually, with designs, costs, evidence, and acceptance criteria.
3. Choose which improvements to implement separately from the corrective release.

No new merge algorithm, automatic full-to-pruned conversion, PDD runtime integration, or performance rewrite belongs in the initial fix phase. The correctness requirement is: **apply a supported adapter completely and faithfully, or reject the operation with an actionable explanation.** Never label a partial merge successful merely because no exception occurred.

This document is a plan, not authorization to publish. Leave the version unchanged during implementation; commit/push and release publication are separate handoff actions. Preserve unrelated existing work and plan files.

## Phase 1 — Correctness fixes

### 1A. Establish regression tests before changing behavior

- [x] Convert the audit probes into ordinary tests asserting the intended behavior, grouped by defect.
- [x] Keep tiny synthetic tensors for deterministic mathematical checks, but use real `.weight` parameter targets and representative trainer key names.
- [x] Add a separate integration-test process using a pinned ComfyUI version and real safetensors serialization. It must not inherit the existing tests' mocked ComfyUI/safetensors modules.
- [x] Store small schema/provenance fixtures for inspected H3 formats in the regression/integration test sources; no full model weights vendored or downloaded.
- [x] Establish expected delta reconstruction with `sum(strength * alpha/rank * B @ A)` for ordinary additive adapters. Set explicit dtype-appropriate tolerances rather than promising bitwise equality after quantization.

Done when each of the ten confirmed audit bugs has a failing regression against the baseline, and additional source-audit concerns have either a reproduction or a documented disposition. The existing passing suite remains the regression baseline, not the sole release criterion.

### 1B. Repair architecture detection and H3 QKV conversion

Audit findings: 1, 2, 3.

- [x] Make H3 detection specific enough to exclude nested UNet feed-forward blocks. Check SD1.5, SDXL, LTX, ACE-Step, Qwen, Z-Image, and partial adapters.
- [x] Distinguish model architecture from source adapter layout. Detect/check each adapter individually; do not normalize a mixed stack using only its first recognizable entry.
- [x] Cross-check model information when available, but do not assume the target model reveals the source adapter's QKV row order.
- [x] Correct alias creation, target lookup, re-fusion, and reverse export mapping to use parameter names ending in `.weight` and module prefixes without it.
- [x] Introduce explicit source-layout selection for ambiguous H3 inputs. Prefer verified metadata where available; `.default`, filename hints, or tensor shape alone must not trigger an irreversible permutation.
- [x] Preserve layout and partition provenance through normalization and caching. Preserve compatibility with existing stack tuple formats and workflows when adding optional controls.
- [x] Handle Q-only, Q+V, and filtered QKV components. Use correctly zero-filled native fused export without inventing contributions in absent slices.

Done when native contiguous, raw interleaved, and Diffusers split fixtures reconstruct the expected delta; every expected attention target resolves; normalization is idempotent; ambiguous inputs stop clearly; and non-H3 fixtures retain their correct architecture/layout. Include shared and independent A factors, mixed ranks/alphas, signed strengths, and token-refiner blocks.

### 1C. Preserve supported payloads and reject unsupported semantics

Audit findings: 4, 5; coverage and base-compatibility concerns.

- [x] Carry ordinary `.diff` matrices and `.diff_b` vectors as first-class additive patches, including bias-only and norm-only entries.
- [x] Verify native ComfyUI key conventions for each tensor kind. Retain tensor kind and shape through target grouping, filtering, analysis, merge, and export.
- [x] Check every merge strategy that consumes dense vectors. If a mode requires a matrix/subspace representation it cannot handle, reject that combination explicitly rather than skipping the tensor.
- [x] Add a consumed/applied/skipped accounting result per adapter. Separate intentional user filtering, missing targets, shape mismatches, and unsupported tensor types.
- [x] Detect DoRA and other base-dependent payloads before additive file conversion. Initially reject unsupported merge/transform/export operations; retain a stock-loader passthrough only where its semantics are actually preserved. Full DoRA merging is not part of this corrective phase.
- [x] Detect PDD metadata/head tensors and reject ordinary standalone merge/export with a companion-runtime explanation.
- [x] Detect full-versus-pruned AdaLN incompatibility and report complete shapes. Reject known partition/basis conflicts; expose unknown compatibility as unknown instead of inferring safety from shape equality.
- [x] Make incomplete H3/distillation application a hard error. Intentional partial filtering must remain explicit and visible in the report.

Done when a mixed factor/dense adapter loses no unfiltered payload in an additive round trip, unsupported bundles fail before returning a partially patched model, and the report reconciles all input payloads. Compatible pruned-native deltas are in scope; automatically converting full-width deltas to a pruned basis is deferred.

### 1D. Make export preserve the accepted patch representation

Audit findings: 6, 7; export portions of 1 and 4.

- [x] Write LoCon middle tensors, or perform a validated equivalent conversion without dropping convolution structure.
- [x] Serialize dense matrices, biases, and norm vectors without forcing them through low-rank compression.
- [x] Preserve float32 factors when necessary; make any lossy dtype choice explicit and safe for mixed-dtype stacks.
- [x] Validate all output tensors and alpha values with `isfinite` after conversion. Reject non-finite output before replacing a destination file.
- [x] Detect duplicate exported keys and unresolved slice mappings instead of silently overwriting contributions.
- [x] Keep bake-strength behavior equivalent across model and CLIP patches, including dense biases.
- [x] Use temporary-file/atomic replacement where practical so an interrupted save does not leave a corrupt output.

Done when save → real stock reload reconstructs the same supported patches within declared tolerances. Cover LoCon, partial QKV, dense vectors, mixed alpha/rank, negative strengths, and the finite-float32/overflowing-float16 counterexample. Failed validation must not overwrite an existing valid file.

### 1E. Correct cache identity and configuration propagation

Audit finding: 8; AutoTuner and cache-schema concerns.

- [x] Include in-memory payload identity in both execution-change detection and instance-cache keys. Re-extraction under the same display name must invalidate the cache.
- [x] Define immutable/revisioned payload ownership before memoizing hashes. Tensor replacement and supported in-place edits must not reuse an obsolete content hash.
- [x] Include relevant source layout, normalization policy, model/CLIP identity, and basis/partition compatibility identity in dependent cache keys. Preserve adapter order wherever semantics are order-dependent.
- [x] Reproduce and fix propagation of STAR/taming controls through fresh AutoTuner analysis, candidate generation, final application, cached replay, and selector/settings paths.
- [x] Add every merge-affecting setting to the appropriate cache identity. Invalidate dependent analysis when a preprocessing option changes.
- [x] Version both analysis data and stored tuner rankings after the semantic changes. Do not treat a package-version bump as cache invalidation.

Done when changing weights/layout/settings recomputes affected results; unchanged inputs still reuse valid work; and fresh versus cached application produces equivalent patches and effective settings. Cover a changed extracted adapter with the same name on the same model.

### 1F. Fix single-item behavior and extraction limits

Audit findings: 9, 10.

- [x] Make single-LoRA processing honor filters and supported transformations.
- [x] Return consistent LORA_DATA for Save/Hook consumers where supported. An unsupported output representation must be explained explicitly, not returned as an unexplained `None`.
- [x] Test the transition from two active adapters to one when the second strength becomes zero.
- [x] Enforce the documented auto-extraction rank maximum. Report actual rank and achieved reconstruction energy when the maximum prevents meeting the requested energy threshold.
- [x] Distinguish omitted bias/norm/non-floating tensors from genuinely zero deltas in extraction diagnostics; do not claim full-finetune equivalence when components are omitted.

Done when the supported `audio_only` single-adapter filter excludes non-audio factors (the original audit's `attention_only` was not a supported filter), downstream output behavior is consistent, and auto extraction with maximum rank 1 never returns rank 8. Preserve unrelated existing node interfaces.

### 1G. Integration, release checks, and fix-phase review

- [x] Run the full existing suite plus all new regressions.
- [x] Run actual ComfyUI mapping/application and safetensors round trips in isolation from unit-test stubs.
- [x] Check Optimizer, Simple, Inline, AutoTuner, Selector, Save, Extract, and Hook paths where each fix applies; confirm unrelated WanVideo behavior is unchanged.
- [x] Test cached/uncached operation, a fresh process, and failure cases—not only successful conversion.
- [x] Add automated test jobs and make the publish workflow depend on their success. Use lightweight fixtures for CI; keep full H3 render checks outside ordinary CI.
- [x] Inspect locally available hardware/models before arranging a fixed-seed H3 smoke test. Do not automatically download large assets, restart services, or consume external paid compute.
- [ ] **Pending usable GPU:** When assets are available, compare direct loading with merged/reloaded adapters on matching FL2VA and Ref2VA bases, and pruned/quantized variants. Check video motion/identity and audio/synchronization; separate mathematical equivalence from perceptual quality.
- [x] Publish a precise supported-format matrix in the documentation, including unsupported/ambiguous inputs and the reason they are rejected.
- [x] Review every audit item as fixed, safely rejected, or explicitly deferred. No high-priority issue may remain a silent-failure path.

Fix-phase exit: all correctness gates pass and any missing full-model validation is stated explicitly. Review the results with the user before proceeding into improvement design or claiming full H3 compatibility. Keep any eventual corrective version bump separate from experimental improvements; no automatic publication at this checkpoint.

Suggested reviewable implementation groups: regression infrastructure → detection/layout/QKV → payload safety → export fidelity → cache/settings → single-item/extraction → integration/docs/CI. Some tests accompany each corresponding fix rather than all tests landing in one permanently failing commit.

## Phase 2 — Detailed improvement review, after fixes

The following are investigation items, not preapproved implementations. For each, produce a short design covering upstream evidence at that time, intended users, exact semantics, compatibility, UI changes, runtime/memory cost, licensing/dependencies, failure policy, tests, and a go/no-go recommendation.

### 2A. Automatic H3 full-to-pruned AdaLN conversion

First priority for the detailed review. Evaluate how to identify/retrieve the matching affine basis; preserve both projected weight and constant bias updates; verify residuals across timesteps; and distinguish known bases, finetunes, and hybrids. Decide whether conversion should be explicit preprocessing or integrated into loading. Treat time-embedder changes and DoRA as separate cases. Depends on phase 1's dense payload and provenance support.

### 2B. Low-rank compression performance and error controls

Compare the existing dense approach with QR plus a small-core SVD on concatenated factors. Measure peak RAM/VRAM, wall time, numerical error, and output rank on representative H3 layers. Specify when it is valid: ordinary low-rank additive combinations, not an automatic replacement for nonlinear/masked merges. Review explicit dtype, rank/size budgets, and reconstruction-error reporting.

### 2C. H3-oriented tuning evaluation

Design an opt-in evaluation workflow using fixed prompts/seeds and an additive baseline. Separate weight-space heuristics from measured audiovisual behavior. Evaluate prompt/identity retention, motion stability, sound content, synchronization, and sensitivity across seeds. Define a bounded render budget before running it. Decide which checks can inform AutoTuner and which belong in a manual benchmark workflow.

### 2D. New merge algorithms

Recheck current implementations and evidence, then compare SSR-Merge, CT-Merging, and TARA-Merging with the repaired additive/conflict-aware baseline. Assess calibration requirements and whether a meaningful H3 audiovisual objective exists. Start with one justified prototype, not three new default modes. Require demonstrated benefit, acceptable resource use, and clear opt-in semantics before integration.

### 2E. PDD interoperability

Choose between documenting a supported external-runtime workflow and carrying PDD heads/schedule metadata through a dedicated integration. Define ownership and patch ordering, and verify rejection of incompatible exports or schedules. Do not expand an ordinary LORA_DATA payload into a runtime-dependent format implicitly.

Phase-2 exit: an evidence-backed shortlist with implementation order and acceptance tests, reviewed before coding. Improvements can ship independently after the corrective release is ready; they must not weaken the fix phase's fidelity and compatibility guarantees.
