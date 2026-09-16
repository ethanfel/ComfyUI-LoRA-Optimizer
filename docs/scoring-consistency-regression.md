# No-op sparsification / representation scoring regression

2026-09-16 — local correction following RAM fix `2334459`.

## Observed report

A two-adapter, rank-16 H3 inline run reported:

- Unsparsified full-strategy candidate: score 0.45, sparsity 15.5%, energy 1.51×.
- DARE-conflict candidate: score 0.49, sparsity 18.6%, energy 1.51×, **zero groups
  sparsified and 208 guard skips**.
- Changing DARE dampening also occupied another slot with the same result.

Skipped preprocessing cannot justify a quality improvement. This does not prove
anything about the visual quality of either configuration.

## Reproduction and cause

A generic rank-16 synthetic pair reproduced the mechanism without user weights:
with the diff cache enabled, weighted-average and weighted-sum no-op DARE outputs
remained dense while their disabled counterparts used native factors. Relative
delta differences were around 1e-7 (FP32 arithmetic), but the old measurements
were 3.78418% versus 4.05579% sparsity. Without the cache, both kept factors and
their measurements matched.

The disabled path used a 64-column sample and its own maximum to measure factor
sparsity. The dense path used every element and the full-matrix maximum. That
systematically made storage format affect the sparsity threshold. A separate
guard required `_diff_cache is None` to restore plain factors after skipped
DARE/DELLA. Captured inline adapters do not even read that cache, but its presence
still blocked their restoration. These differences predate the streaming RAM fix.

## Correction

- Dense and ordinary factored patches now use the same 64-column sample, or all
  columns when narrower. Both use FP32 absolute values and a threshold of 1% of
  the **sample** maximum. Counts use the same helper.
- The final target key seeds a private CPU generator for indices, avoiding the
  different CPU/CUDA permutations previously produced from an identical seed.
  Only the tiny index tensor transfers; scoring stays on the requested device.
  QKV groups are measured after fusion, and Z-Image output-projection renaming
  uses the same canonical identity before/after inline scoring.
- Verified conflict-guard skips can restore eligible plain linear factors with
  the cache enabled as well as disabled. The cache contains raw FP32 expansions;
  unsupported dense/exotic contributors still fall back, and cleaning, masks,
  preserve overlays and spatial targets retain their exclusions.
- Cache hits now retain source FP16/BF16 dtype metadata for ordinary file LoRAs
  and dense payloads. Previously a warm cache could leave those outputs FP32
  when recomputation would use the source dtype.
- Reports say **Sampled sparsity**. Scoring revision `1.13.2` forces automatic
  retuning instead of replaying old rankings. Explicit older `TUNER_DATA` keeps
  the existing warning and should be regenerated.

The merge equations, statistical score weights, optional SVD rank formulas,
and search/dedup policy are unchanged. This is not a general proof that optional
dense-rank and factor-rank proxies are interchangeable. No-op tests with full
SVD tie because equivalent candidates now retain the same representation.
Numerically identical configurations can still appear as ties in the report;
this fix does not expand or deduplicate the candidate search.

## Validation

`tests/test_scoring_representation.py` covers dense/factor sparsity agreement,
signed and zero scales, FP32/FP16/BF16 factors, narrow/wide matrices, a deliberately
unsampled outlier, deterministic indices without consuming global RNG, inline
and tensor-free score records, and output-key renaming.

Cache tests cover disabled/auto/RAM/disk modes, both cold and warm caches, plain
file and captured-inline inputs, sum/average/SLERP, and DARE/DELLA guard skips.
The unchanged candidate deltas and all measured scores match their baseline,
including optional full-SVD scoring. Complete synthetic AutoTuner sweeps also
check that the disabled, skipped-DARE, dampened-skipped-DARE and skipped-DELLA
trials tie. CUDA checks verify device placement and bounded sparsity scratch
space instead of a full-sized absolute-value/mask allocation.

The user's actual H3 run still needs a fresh sweep after restart/update. Sampled
sparsity can miss unsampled features and is a weight-space statistic, not a
validated visual-quality metric. Exact historic scores are not expected to
remain unchanged after correcting the inconsistent measurement.
