# Phase 2 experimental merge validation — 2026-09-06

Implementation baseline: `5bc2970`. Phase 2 validation is complete and commit/push was authorized at handoff; package version is still **1.8.3**. No release, checkpoint download, service restart, or full-model render was performed.

## Implemented scope

- Separate LoRA Experimental Options node with independent NP-LoRA / CT-Merging switches, appended optional sockets on AutoTuner and AutoTuner Settings, and Simple/Inline propagation.
- Stable grid and disabled/disconnected execution/cache identity preserved. Enabled runs retain stable trials and add bounded experimental trials plus an additive baseline. Full-target scoring checks every layer before ranking.
- Independent paper-based implementations with documented numerical safeguards, additive side-payload/preserve policy, exact-factor QR/small-core SVD when eligible, and dense fallback.
- Experimental parameters carried through fresh application, in-memory/persistent replay, Selector, legacy/Simple bridges, and safetensors metadata. Community rankings, strength-ignoring memory, cross-candidate patch reuse, and dataset recording are isolated from experiments.

Usage, parameter defaults, math sources, and deliberate numerical extensions: [experimental merging](../experimental-merging.md).

## Checks

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 645 passed, 3 skipped, 30 subtests passed |
| `node tests/js/dynamic_migration.test.cjs` | 3 passed |
| `python tests/integration/comfy_roundtrip.py /media/p5/Comfyui` | Passed with real ComfyUI / safetensors, CPU only |
| `git diff --check` | Passed |

New tests cover independent NP closed-form and CT projector references; signed strengths, mixed ranks and alpha; zero/identical/cancelling/tied subspaces; role reversal; factor/dense agreement; additive bias/norm/unique targets; preserve overlays; candidate switches/budgets; actual shared-convolution rejection under fast scoring settings; cache separation; dataset isolation; and replay through every supported entry point. Inline Chain tests force each experimental method to win through a test evaluator and verify that the input patcher remains untouched.

The isolated integration test now exports/reloads both methods on miniature H3-shaped modules: partial QKV zero filling, AdaLN dense updates/bias, norm vectors, signed model/CLIP contributions, and experimental metadata. It checks actual stock-loaded weight application, not merely file serialization. These are tensor-semantics tests, not generation-quality tests.

## Numerical and timing observations

On the fixed mixed-rank float32 test fixture, NP dense/factor results are within `1e-7` relative Frobenius error of a float64 reference; CT is within `7e-6`. CT polar alignment is more sensitive near rank deficiency: tests use an explicit `2e-5` relative Frobenius bound plus elementwise tolerances, not a bitwise-equality claim.

A small CPU-only NP timing probe used two random rank-8 updates of shape 512×256, generator seed 99, one Torch CPU thread, one warmup and three timed merges. Dense SVD averaged **12.40 ms**, factor SVD **1.82 ms**, with relative output difference **4.52e-8**. This measures the merge function on that synthetic case only; it is not an H3 throughput or memory benchmark. Dense patches still consume memory.

## Remaining limits

PyTorch is `2.11.0+cu130`, but `torch.cuda.is_available()` is false. Full H3 FL2VA/Ref2VA renders, audio/synchronization, pruned/quantized perceptual quality, and full-size GPU performance remain unverified. No claim of superiority over additive or stable merging is made.

Shared spatial convolutions and experimental formula trees remain unsupported. Turbo/distillation adapters must be explicitly preserved or applied separately; arbitrary files cannot be reliably auto-classified. Automatic full-to-pruned affine conversion, PDD runtime integration, SSR calibration, and CASA transfer are not implemented by this change.
