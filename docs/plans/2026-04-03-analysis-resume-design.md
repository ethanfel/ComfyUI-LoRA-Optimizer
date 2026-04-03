# Analysis Resume Design

**Date:** 2026-04-03  
**Branch:** feat/analysis-cache  
**Status:** Approved

## Problem

If `_run_group_analysis` crashes mid-run (OOM, power loss, etc.), no analysis progress is saved. The next run starts from scratch even though many prefixes were already fully computed.

## Solution

Checkpoint per-prefix analysis results to a `.partial` file after each prefix completes. On resume, load the partial file as `cached_analysis` — already-done prefixes are skipped, the run continues from where it crashed.

## Files

Two files per `names_only_hash`:

| File | Purpose |
|------|---------|
| `{hash}.analysis.json` | Complete cache — written only on full success |
| `{hash}.analysis.partial.json` | In-progress checkpoint — written after each prefix |

Both share the same JSON structure so load logic is reusable:

```json
{
  "analysis_version": 1,
  "algo_version": "...",
  "created_at": "...",
  "source_loras": [...],
  "per_prefix": { ... }
}
```

## New Methods on `LoRAAutoTuner`

```python
_analysis_partial_path(hash)       # → {hash}.analysis.partial.json
_analysis_partial_load(hash)       # same logic as _analysis_cache_load, reads .partial
_analysis_partial_save(hash, per_prefix, source_loras)  # atomic write (tmp + os.replace)
_analysis_partial_delete(hash)     # silent os.unlink on success
```

## Resume Flow in `auto_tune`

Load priority: full cache → partial → nothing:

```python
cached_analysis = self._analysis_cache_load(names_only_hash)
if cached_analysis is None:
    cached_analysis = self._analysis_partial_load(names_only_hash)
```

Define callback with accumulator seeded from whatever was already cached:

```python
partial_accumulated = dict(cached_analysis or {})

def on_prefix_done(prefix, entry):
    partial_accumulated[prefix] = entry
    self._analysis_partial_save(names_only_hash, partial_accumulated, source_loras)
```

Pass callback to `_run_group_analysis`. On successful return, promote and clean up:

```python
self._analysis_cache_save(names_only_hash, merged, source_loras)
self._analysis_partial_delete(names_only_hash)
```

## Changes to `_run_group_analysis`

One new optional parameter:

```python
def _run_group_analysis(self, ..., on_prefix_done=None):
```

After each freshly-computed prefix is extracted, fire the callback:

```python
entry = self._extract_for_analysis_cache(result, active_loras)
new_analysis_entries[prefix] = entry
if on_prefix_done is not None:
    on_prefix_done(prefix, entry)
```

Already-cached prefixes skip the callback — they're already in `partial_accumulated` from initialization.

Both non-`auto_tune` call sites pass `on_prefix_done=None` implicitly — no behavior change.

## Non-Goals

- No checkpointing inside a single prefix (mid-tensor-compute crash is unrecoverable anyway)
- No configurable flush interval (per-prefix is simple and correct)
- No changes to the optimizer call site
