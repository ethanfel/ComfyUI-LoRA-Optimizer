# Pair/LoRA Cache Design

**Date:** 2026-04-03  
**Branch:** feat/analysis-cache  
**Status:** Approved

## Problem

The run-level analysis cache (`{names_hash}.analysis.json`) is keyed by the full set of LoRAs. Adding or swapping any LoRA causes a full cache miss — every prefix is re-analyzed from scratch, even though pairwise metrics between unchanged LoRAs are mathematically identical across runs.

Example: running [A, B] then [A, B, C] recomputes pair (A,B) in full, wasting significant time.

## Solution

Two new cache layers keyed by individual LoRA identity:

- **Per-LoRA cache** (`{lora_hash}.lora.json`): per-prefix stats for one LoRA (norm_sq, rank, magnitude_samples, strength_sign, target_key, is_clip, skip_count, raw_n)
- **Per-pair cache** (`{hash_a}_{hash_b}.pair.json`): per-prefix pairwise metrics for one (LoRA_A, LoRA_B) combination (overlap, conflict, dot, norm_a_sq, norm_b_sq, weighted_total, weighted_conflict, expected_conflict, excess_conflict, subspace_overlap, subspace_weight)

When the run-level cache misses, the pair/lora caches allow full or partial prefix reconstruction without reloading diff tensors.

## Hash Keys

`_lora_identity_hash(lora_item)` — 16-char hex from `(name, mtime, size)`, same data `_compute_names_only_hash` already collects, applied per-LoRA.

Pair file key: `{min_hash}_{max_hash}` (lexicographic sort) — order-independent regardless of LoRA index in any given run.

## File Contents

### `{lora_hash}.lora.json`

```json
{
  "algo_version": "...",
  "per_prefix": {
    "lora_unet_down_blocks_0_...": {
      "norm_sq": 1.5,
      "rank": 16,
      "magnitude_samples_unscaled": [0.1, 0.2],
      "strength_sign": 1,
      "target_key": "model.layers.0.weight",
      "is_clip": false,
      "skip_count": 0,
      "raw_n": 1
    }
  }
}
```

### `{hash_a}_{hash_b}.pair.json`

`hash_a < hash_b` always. `norm_a_sq` corresponds to the LoRA with the lexicographically smaller hash.

```json
{
  "algo_version": "...",
  "per_prefix": {
    "lora_unet_down_blocks_0_...": {
      "overlap": 100, "conflict": 30, "dot": 0.5,
      "norm_a_sq": 1.0, "norm_b_sq": 1.0,
      "weighted_total": 0.8, "weighted_conflict": 0.2,
      "expected_conflict": 0.15, "excess_conflict": 0.05,
      "subspace_overlap": 0.3, "subspace_weight": 1.0
    }
  }
}
```

Both files: atomic writes (tmp + replace), invalidated by `algo_version` change. Changed LoRA files get a different hash → different filename → stale files are never referenced (no explicit expiry).

Note: subspace basis (SVD vectors) is NOT cached — at max_rank=8 it would be ~256KB per prefix per LoRA, too large for JSON. Basis is always recomputed when needed.

## Integration with Analysis Pipeline

### Load phase (before prefix loop, in `auto_tune`)

```python
lora_hashes = {i: _lora_identity_hash(lora) for i, lora in enumerate(active_loras)}
lora_caches = {i: _lora_cache_load(h) for i, h in lora_hashes.items()}
pair_caches = {
    (i, j): _pair_cache_load(lora_hashes[i], lora_hashes[j])
    for i, j in pairs
}
```

Pass `lora_caches` and `pair_caches` to `_run_group_analysis`.

### Per-prefix decision (inside `_run_group_analysis`)

1. **Full hit** — all LoRAs have lora cache entries for this prefix AND all pairs have pair cache entries → call `_reconstruct_from_pair_lora_cache`, skip `_analyze_target_group` entirely
2. **Partial or full miss** → call `_analyze_target_group` as normal; freshly computed results update `new_lora_entries` and `new_pair_entries`

Sign-flip check: if any LoRA's current strength sign differs from cached `strength_sign`, treat as miss for that prefix and fall back to `_analyze_target_group`.

### Save phase (after successful run, end-of-run)

For each LoRA and each pair: merge newly computed per-prefix entries into the existing cache file (load → merge → atomic write).

No incremental per-prefix writes — crash recovery is handled by the existing `.partial` file (run-level). The pair/lora cache is a secondary optimization layer; losing it on crash is acceptable.

## New Methods on `LoRAAutoTuner`

| Method | Purpose |
|--------|---------|
| `_lora_identity_hash(lora_item)` | 16-char hash from name+mtime+size |
| `_lora_cache_path(lora_hash)` | `{hash}.lora.json` |
| `_lora_cache_load(lora_hash)` | per_prefix dict or None |
| `_lora_cache_save(lora_hash, per_prefix)` | atomic write |
| `_pair_cache_path(hash_a, hash_b)` | `{min}_{max}.pair.json` |
| `_pair_cache_load(hash_a, hash_b)` | per_prefix dict or None |
| `_pair_cache_save(hash_a, hash_b, per_prefix)` | atomic write |
| `_extract_for_lora_cache(result, lora_idx, active_loras)` | single-LoRA entry dict |
| `_extract_for_pair_cache(result, i, j, hash_a, hash_b)` | single-pair entry dict |
| `_reconstruct_from_pair_lora_cache(prefix, lora_entries, pair_entries, active_loras, lora_hashes)` | 8-tuple or None |

**Wiring changes to `_run_group_analysis`:**
- Add optional `lora_caches=None, pair_caches=None` params
- On full hit: call `_reconstruct_from_pair_lora_cache` instead of `_analyze_target_group`
- Return `new_lora_entries` and `new_pair_entries` alongside existing `new_analysis_entries`

## Non-Goals

- No subspace basis caching (too large for JSON)
- No incremental per-prefix writes for pair/lora caches (run-level `.partial` handles crash recovery)
- No cache size limits or eviction (files are small per-LoRA; disk space grows slowly)
- No changes to the existing run-level cache (remains the fast path for repeated identical runs)
