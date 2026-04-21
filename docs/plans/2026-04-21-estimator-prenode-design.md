# LoRA Merge Estimator Pre-Node — Design

**Date:** 2026-04-21
**Status:** Design approved, implementation pending
**Scope:** New ComfyUI node that predicts optimal merge configs from collected HF dataset, skipping the AutoTuner's Phase 2 grid sweep.

## Motivation

The AutoTuner has two expensive phases:
- **Phase 1** (analysis): per-prefix conflict math, deterministic, cacheable. 5–30s typical; ~1s when all pairs are in the HF community cache.
- **Phase 2** (grid scoring): merges + scores ~200 candidate configs. Dominant cost; not cacheable because it depends on the full combo.

The HF community cache dataset (`ethanfel/lora-optimizer-community-cache`) now holds 400 config entries + 327 per-LoRA caches + 695 per-pair caches — enough data to predict merge outcomes by similarity lookup instead of re-running the grid. Phase 1 output forms a natural "feature fingerprint" for each combo; past merges with similar fingerprints are strong predictors of what will win on a new combo.

The goal is a single new node that takes a new LoRA combo, runs Phase 1 (fact), retrieves similar past merges from the dataset, and emits a `tuner_data` struct the existing optimizer can replay — no grid sweep needed.

## Core concept

Case-based retrieval (k-NN), not parametric regression. Rationale: 400 samples is starved for a neural net or gradient-boosted regressor but perfect for instance-based methods. k-NN also generalizes gracefully to partially-seen LoRAs as long as their Phase 1 stats resemble past combos.

## Workflow topology

```
[LoRA Stack] ─┬──→ [Estimator] ──tuner_data──→ [Optimizer (settings_source=from_tuner_data)] → merged model
              │                                     ↑
              └─────────────────────────────────────┘
```

LoRA stack wires to both nodes. The optimizer's existing `settings_source=from_tuner_data` plug consumes the estimator's output transparently — no optimizer changes.

## Data flow

1. **Phase 1** (reused from AutoTuner): detect architecture, compute per-prefix conflict/magnitude/subspace stats. Cache-hit prefixes download from HF; cache-miss prefixes run tensor math locally.
2. **Featurize**: aggregate stats into a fixed-dim combo fingerprint (see below).
3. **Retrieve**: cosine k-NN against a prebuilt index, with hard filter on `base_model_family` + `combo_size`.
4. **Aggregate candidates**: sum inverse-distance-weighted scores per unique config across retrieved neighbors' `candidates` arrays. Return top-N.
5. **Emit**: `tuner_data` (optimizer-compatible) + `estimator_report` (neighbors, distances, confidence).

## Feature representation

~120-dim vector per combo, combining:

- **Aggregated pair stats**: mean / p90 / max / std across all prefixes, across all pairs in the combo, for each of 11 pair stats (`overlap`, `conflict`, `dot`, `norm_a_sq`, `norm_b_sq`, `weighted_total`, `weighted_conflict`, `expected_conflict`, `excess_conflict`, `subspace_overlap`, `subspace_weight`). 11 × 4 = 44 dims.
- **Aggregated lora stats**: same four aggregates across 7 lora stats. 7 × 4 = 28 dims.
- **Worst-pair summary**: stats of the pair with the highest `excess_conflict` — preserves tail behavior (distinguishes "one ugly pair in a triple" from "all three mild"). ~20 dims.
- **Combo size one-hot**: pair / triple / quad / larger. 4 dims.
- **Base model family one-hot**: zimage / wan / flux / qwen_image / acestep / ltx / sdxl / sd15 / unknown. 9 dims.

Normalization: z-score per feature against the training set. Stored in `meta.json` so inference applies the same transform.

## Retrieval algorithm

- **Index**: `sklearn.NearestNeighbors(metric="cosine")` over the z-scored feature matrix.
- **k**: 5 (configurable via node input).
- **Hard filter**: before distance ranking, filter the index to entries matching `base_model_family` and `combo_size`. Prevents "almost matches" across regimes.
- **Aggregation**: each neighbor contributes up to 10 candidates (top_n from its grid sweep). For each unique `config` tuple, sum `score_final × (1 / (1 + distance))` across contributing neighbors. Return top-N by aggregated score.
- **Confidence signal**: mean distance of retrieved neighbors exposed in the report; low = strong match, high = sparse coverage in this region.

### Follow-up (v2): local re-rank
Once the MVP ships, add an optional re-rank step that recomputes the AutoTuner's heuristic score for each top retrieved candidate against the *new* combo's actual Phase 1 stats. Retrieval picks the candidate set; heuristic provides local-evidence tiebreaking.

## Per-prefix prediction

The current HF payload lacks per-prefix strategy decisions (only global configs). Two changes land before the estimator:

1. **AutoTuner instrumentation** (separate commit, independent of estimator): during grid evaluation, record which merge strategy wins for each prefix for each top-N candidate. Add `per_prefix_decisions: {prefix: strategy_name}` to each candidate in the upload payload.
2. **Rerun the 400 combos**: user reruns via the existing `LoRACombinationGenerator`. Uploads overwrite the old keys (`config/{joined_hashes}_{arch}.config.json`), so the dataset is enriched in-place — no schema forking.

Estimator MVP then ships with both global retrieval AND per-prefix retrieval on day one. The 2 legacy `sd_unet` configs are dropped from scope as non-Zimage noise.

## Artifacts and freshness

- **Location**: `models/estimator/index.pkl` + `models/estimator/meta.json`
- **Build**: one-shot `scripts/build_estimator_index.py`:
    - `snapshot_download` configs/pair.json/lora.json from HF (pinned to current HEAD SHA)
    - Parse → feature matrix + label arrays
    - Fit `NearestNeighbors`, pickle index
    - Write `meta.json` with: `hf_commit_sha`, `ESTIMATOR_INDEX_VERSION`, z-score params, feature schema
- **First-run**: build takes 30–60s (local); cached thereafter
- **Freshness**: on each estimator invocation, ~50ms HTTP request to fetch current HF HEAD SHA. If different from cached `hf_commit_sha`, rebuild silently. Falls back to cached index on no-internet with a warning.
- **Node input**: `rebuild_index: auto | force | skip` for manual control.
- **Versioning**: `ESTIMATOR_INDEX_VERSION` (separate from `ANALYSIS_CACHE_VERSION`) — bump when feature schema changes, triggers unconditional rebuild.

## Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Feature extractor | `lora_optimizer.py` (new class `EstimatorFeatureExtractor`) | Build 120-dim vector from Phase 1 stats |
| Index builder | `scripts/build_estimator_index.py` | Offline pipeline: pull HF → featurize → fit → pickle |
| Estimator runtime | `lora_optimizer.py` (new class `LoRAEstimator`) | Load index, run retrieval, emit `tuner_data` |
| Estimator node | `lora_optimizer.py` (NODE_CLASS_MAPPINGS) | ComfyUI surface |
| Per-prefix capture | `lora_optimizer.py` (AutoTuner instrumentation) | Record winning per-prefix strategies during grid evaluation |
| Tests | `tests/test_estimator.py` | Feature shape, retrieval determinism, tuner_data schema, freshness logic |

## Dependencies

- `scikit-learn` (new — for `NearestNeighbors`). Add to `pyproject.toml`.
- `numpy` (already transitive).
- `huggingface_hub` (already used by community cache).

## Out of scope

- **Fast / cold-LoRA mode** (no Phase 1). Metadata-only signal (rank, arch, filename) is too weak for useful retrieval — would poison user trust. Revisit if a concrete use case appears.
- **Parametric regression / neural estimator**. 400 samples insufficient; k-NN is the right tool at this scale.
- **Uploading prebuilt indices to HF**. Local-build is simpler, always current, no maintenance step.
- **Cross-architecture pooling**. Hard filter on `base_model_family` means WAN data won't inform Zimage predictions. Pooling could be a later study if the dataset grows enough.

## Open questions (for implementation)

- Exact worst-pair summary dims — which stats are most informative?
- Default `k` — 5 is a guess; tune against the 400-sample set during index build.
- How to expose the `estimator_report` in ComfyUI (string output vs. structured panel).
- Whether to cache retrieval results per `(combo fingerprint hash)` — probably yes, tiny cost.
