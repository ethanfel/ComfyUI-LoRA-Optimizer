# QKV reassembly memory regression

The Windows reports on 1.8.6 failed after Pass 2, first during FP32 expansion
of a native QKV diff and then inside dense Q/K/V concatenation. Reducing patch
storage from 50% to 35% did not prevent the second failure. The native access
violation has not been reproduced; these tests establish the allocation bug,
not a Windows driver/runtime diagnosis.

## Fix

- Check remaining patch-storage allowance and live workspace headroom before
  allocating fused outputs, casts, padding, or native/sliced accumulation.
- Assemble mixed-device groups on CPU; do not migrate every component to the
  first component's GPU. Dense groups use one destination with sequential
  copies instead of device copies followed by another full concatenation.
- Release consumed entries and their stale scoring references group by group.
  Read-only/export callers keep their input dictionaries and tensors intact.
- Retry a recoverable CUDA OOM on CPU before consuming the group's inputs.
  Other runtime errors still propagate. This cannot catch a fatal native crash.
- Keep GPU scoring enabled. GPU-fused dense groups are scored and offloaded
  immediately; CPU-fused groups are scored on GPU individually afterward.
- Export reassembly uses RAM. No node connections, scoring formula, merge
  settings, or public defaults are changed.

## Bounded comparison

Run each command in a separate process with CUDA available:

```bash
python tests/integration/qkv_memory_stress.py baseline
python tests/integration/qkv_memory_stress.py fixed
```

The baseline reads reassembly methods from release commit `27482bb`, which
must be available in local Git history. No model weights or ComfyUI server are
used. Each process has a 768 MiB PyTorch allocator cap. The synthetic stack
has 96 native-plus-sliced dense QKV groups, starting at 432 MiB of CUDA tensors.

Observed on an RTX 5090 with Linux, September 10, 2026:

| Case | Peak PyTorch allocated | Outcome | Groups scored on GPU |
| --- | ---: | --- | ---: |
| 1.8.6 reassembly | 645.75 MiB | CUDA OOM before completion | 0 |
| Fixed reassembly | 432 MiB | Completed; all output values checked | 96 |

The allocator cap also includes reserved blocks; peak allocated memory is
not the entire device footprint. This is a scaled memory regression, not an
H3 render-quality or throughput benchmark. Do not compare the failed run's
elapsed time with the successful run's time as a speed measurement.

Unit tests cover dtype promotion, noncontiguous inputs, partial H3 padding,
native/sliced accumulation, low-rank and exotic adapters, input preservation,
allocation planning, stale-reference release, OOM retry, and unchanged GPU
scoring. Real ComfyUI CPU integration tests also check application and export
round trips. The original Windows workflow still requires validation.

Validation of an isolated 1.8.6 tree plus this fix (unrelated experimental
worktree changes excluded): 1,092 CPU tests and 69 subtests passed, with 11
CUDA-only tests skipped in that run. A separate CUDA run passed all 46 selected
QKV/scoring regressions, including the eight new CUDA cases and existing
Z-Image deferral tests. Both real ComfyUI CPU integration programs passed,
including all ten inline-loader/file-stack/export parity cases. No running
ComfyUI session was started, stopped, updated, or used for these checks.
