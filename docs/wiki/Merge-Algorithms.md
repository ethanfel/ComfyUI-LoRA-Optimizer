# Merge Algorithms

Deep dive into each merge algorithm: the math, when each is selected, and how they interact with sparsification and quality enhancements.

For the overall pipeline and composition order, see [[How It Works]].

---

## Overview

The optimizer has five merge algorithms. Each prefix in the model can use a different one based on local conflict data.

| Algorithm | Best For | Properties |
|-----------|----------|------------|
| [Weighted Sum](#weighted-sum) | Single-LoRA prefixes | Lossless, fully SVD-compressible |
| [Weighted Average](#weighted-average) | Low-conflict overlapping prefixes | Lossless, fully SVD-compressible |
| [TIES](#ties-merging) | High-conflict prefixes (>25% sign conflict) | Nonlinear, lossy SVD |
| [SLERP](#slerp) | Low-conflict, similar-magnitude LoRAs | Magnitude-preserving |
| [Consensus](#consensus-merging) | Highly similar LoRAs (high cosine similarity) | Importance-weighted, spectral cleanup |

---

## Weighted Sum

The simplest merge: plain scaled addition.

### Formula

```
output[i] = Σ_k (strength_k × diff_k[i])
```

Each LoRA's contribution is scaled by its strength and summed. No normalization, no conflict resolution.

### When Selected

- Automatically used for prefixes where **only one LoRA** has weights (no overlap, no conflict)
- Forced everywhere when `optimization_mode=additive`

### Properties

- Preserves every contribution exactly
- No information loss — output is the exact linear combination
- Fully compressible by SVD (rank = sum of input ranks)
- Can oversaturate when LoRAs reinforce each other at high strength

---

## Weighted Average

Normalized sum that prevents magnitude inflation.

### Formula

```
total_weight = Σ_k |strength_k|

output[i] = Σ_k (strength_k / total_weight × diff_k[i])
```

Same as weighted sum, but divided by the sum of strengths. This ensures the output magnitude stays proportional to individual LoRAs rather than growing with the number of LoRAs.

### When Selected

- Per-prefix: 2+ LoRAs present AND sign conflict ratio ≤ 25%
- Global: aggregate conflict ≤ 25% AND cosine similarity between 0.3–0.7
- Can be upgraded to SLERP in `v1.2` behavior profile when LoRAs are orthogonal

### Properties

- Fair blending — each LoRA contributes proportionally
- Prevents magnitude inflation from stacking
- Fully compressible by SVD (linear operation)
- Doesn't handle sign conflicts (opposing contributions cancel partially)

---

## TIES Merging

**T**rim, **I**nterfere, **E**lect **S**ign — specifically designed to resolve sign conflicts.

*Reference: [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708)*

### Algorithm

TIES has three steps applied at each position independently:

#### Step 1 — Trim (Sparsification)

Keep only the top-k% of weights by absolute magnitude. The rest are zeroed out.

```
for each LoRA k:
    threshold_k = quantile(|diff_k|, 1 - density)
    trimmed_k[i] = diff_k[i] if |diff_k[i]| ≥ threshold_k else 0
```

The `density` parameter controls how aggressive the trim is.

**Interaction with DARE/DELLA:** If sparsification is enabled, it **replaces** this trim step entirely. Both achieve the same goal (zeroing out low-value weights), so running both would be redundant.

#### Step 2 — Elect Sign

For each position, vote on the majority sign direction across all LoRAs.

**Frequency method** (equal votes):
```
elected_sign[i] = sign(Σ_k sign(trimmed_k[i]))
```
Each LoRA gets one vote per position.

**Total method** (magnitude-weighted votes):
```
elected_sign[i] = sign(Σ_k trimmed_k[i])
```
Larger magnitudes carry more weight in the vote.

The optimizer picks the method based on magnitude ratio:
- Ratio > 2× → `total` (stronger LoRA dominates)
- Ratio ≤ 2× → `frequency` (equal votes)

#### Step 3 — Disjoint Merge

Average only the LoRAs that agree with the elected sign. Contributors opposing the majority are excluded.

```
agreeing_k = {k : sign(trimmed_k[i]) == elected_sign[i]}

output[i] = Σ_{k ∈ agreeing_k} (strength_k × trimmed_k[i]) / Σ_{k ∈ agreeing_k} |strength_k|
```

### When Selected

- Per-prefix: 2+ LoRAs present AND sign conflict ratio > 25%
- Global: aggregate conflict ratio > 25%

### Properties

- Resolves sign conflicts without cancellation — opposing LoRAs don't interfere
- Nonlinear operations (trim + sign election) produce full-rank output
- SVD compression is lossy on TIES prefixes
- Best for genuinely conflicting LoRAs

---

## SLERP

**S**pherical **L**inear int**ERP**olation — magnitude-preserving directional blend along the geodesic on a hypersphere.

### Formula (2 LoRAs)

```
Ω = arccos(cos_sim(diff_A, diff_B))

output = sin((1-t)Ω)/sin(Ω) × diff_A + sin(tΩ)/sin(Ω) × diff_B
```

Where `t` is the interpolation weight derived from LoRA strengths.

For near-parallel vectors (Ω ≈ 0), falls back to linear interpolation to avoid numerical instability.

### Formula (3+ LoRAs)

Iterative pairwise SLERP, sorted by descending weight:

```
1. Sort diffs by strength (descending)
2. result = diff_strongest
3. for each remaining diff_k (by decreasing strength):
       t_k = adjusted interpolation weight
       result = slerp(result, diff_k, t_k)
4. Correct final magnitude to match weighted average of input norms
```

The strongest LoRA anchors the direction; subsequent LoRAs blend in.

### When Selected

- v1.2 behavior profile: auto-selected when conflict ≤ 25% AND LoRAs are detected as orthogonal (low cosine similarity)
- Not auto-selected in `no_slerp` or `classic` profiles
- Can be forced via `merge_strategy_override`

### Properties

- Preserves weight magnitude (no inflation or deflation)
- Smooth directional interpolation on the unit sphere
- Magnitude correction ensures output norm matches expected value
- Best for low-conflict LoRAs of similar magnitude

---

## Consensus Merging

Three-stage algorithm designed for highly similar LoRAs where importance-weighting and spectral cleanup produce cleaner results than simple averaging.

### Stage 1 — Fisher-Proxy Importance Weighting

Weight each parameter by `|diff|²` as a proxy for Fisher information (parameter importance):

```
numerator[i] = Σ_k (diff_k[i] × strength_k × |diff_k[i]|²)

denominator[i] = Σ_k (|strength_k| × |diff_k[i]|²)

output[i] = numerator[i] / denominator[i]
```

Parameters where LoRAs make large changes get more influence. Positions with near-zero changes contribute little.

### Stage 2 — MAGIC Calibration

Rescale the merged result so its L2 norm matches the weighted average of input norms:

```
target_norm = Σ_k (|strength_k| × ||diff_k||₂) / Σ_k |strength_k|

output = output × (target_norm / ||output||₂)
```

This prevents magnitude drift from the importance weighting.

### Stage 3 — MonoSoup Spectral Cleanup

SVD-based entropy filtering on 2D+ tensors:

```
1. Compute SVD: output = U × S × Vᵀ
2. Compute entropy of singular values: H = -Σ (p_i × log(p_i))
   where p_i = s_i / Σ s_j
3. Effective rank = exp(H)
4. Keep only top effective_rank singular values
5. Reconstruct: output = U[:,:r] × S[:r] × Vᵀ[:r,:]
```

This removes noise — singular values with negligible contribution are filtered out, producing a cleaner merge.

### When Selected

- v1.2 behavior profile: auto-selected when conflict ≤ 25% AND cosine similarity > 0.7 (highly similar LoRAs)
- Not available in `classic` profile
- Can be forced via `merge_strategy_override`

### Properties

- Importance-weighted — parameters that change most get the most influence
- Magnitude-calibrated — no drift from weighting
- Spectral cleanup — removes merge noise via SVD filtering
- Best for similar LoRAs (same concept, different epochs/seeds)

---

## Sparsification Methods

Sparsification is a preprocessing step that reduces parameter interference by zeroing out weights before merging. Two algorithms with two variants each.

### DARE (Drop And REscale)

*Reference: [Yu et al., ICML 2024](https://arxiv.org/abs/2311.03099)*

```
mask[i] ~ Bernoulli(density)              # each position kept with probability = density
sparse_diff[i] = diff[i] × mask[i]        # zero out dropped positions
sparse_diff[i] = sparse_diff[i] / density  # rescale survivors to preserve expected value
```

**DAREx enhancement** (ICLR 2025): The `dare_dampening` parameter `λ` interpolates rescaling:
```
rescale_factor = 1 / (density^(1-λ))
```
At λ=0: standard DARE. Higher λ: less noise amplification at low density.

### DELLA (magnitude-aware dropout)

*Reference: [Deep et al., 2024](https://arxiv.org/abs/2406.11617)*

```
for each row of diff:
    rank positions by |value| (ascending)
    drop_probability[i] = (1 - density) × (rank[i] / num_positions)
    mask[i] ~ Bernoulli(1 - drop_probability[i])
    sparse_diff[i] = diff[i] × mask[i] / (1 - drop_probability[i])
```

Low-magnitude weights have higher drop probability. High-magnitude weights are preserved. More surgical than DARE.

### Conflict-Aware Variants

Standard sparsification operates everywhere. Conflict-aware variants (`dare_conflict`, `della_conflict`) only sparsify positions where LoRAs actually interfere:

```
conflict_mask[i] = (count of LoRAs with opposing signs at position i) ≥ 2

if conflict_mask[i]:
    apply DARE or DELLA sparsification
else:
    keep full weight (no sparsification)
```

This preserves unique contributions and same-sign reinforcements while still reducing interference at contested positions.

### Interaction with Merge Strategies

| Merge Strategy | Sparsification Role |
|---------------|-------------------|
| TIES | **Replaces** the TIES trim step (Step 1) |
| Weighted Sum | Preprocessing before sum |
| Weighted Average | Preprocessing before average |
| SLERP | Preprocessing before interpolation |
| Consensus | Preprocessing before importance weighting |

---

## Quality Enhancements

Quality enhancements compose with all merge strategies. They modify the diffs or the merge process to improve results. See [[How It Works#quality-enhancements]] for the pipeline position.

### DO-Merging (enhanced+)

*Reference: [arXiv 2505.15875](https://arxiv.org/abs/2505.15875)*

Orthogonalizes LoRA direction vectors via Modified Gram-Schmidt:

```
1. Flatten each diff to a 1D vector
2. Apply Gram-Schmidt orthogonalization (order: by descending norm)
3. Record original magnitudes
4. Reconstruct: new_diff = orthogonal_direction × original_magnitude
```

After orthogonalization, LoRAs that were partially aligned become truly independent. This reduces directional interference while preserving each LoRA's strength.

### Column-Wise Conflict Resolution (enhanced+)

*Inspired by [ZipLoRA, Shah et al., 2025](https://arxiv.org/abs/2311.13600)*

Instead of each element voting on sign direction independently, entire output neurons (rows of the weight matrix) vote as a unit:

```
for each row j of the weight matrix:
    row_sign_votes = Σ_k sign(diff_k[j, :]) × |diff_k[j, :]|
    elected_row_sign = sign(Σ row_sign_votes)
    apply elected_row_sign to entire row
```

Preserves structural coherence — a neuron's input weights work together as a functional unit. Falls back to element-wise for 1D tensors (biases, layer norms).

### TALL-Masks (enhanced+)

*Reference: [Wang et al., 2024](https://arxiv.org/abs/2406.12832)*

Identifies "selfish" weights — positions where one LoRA dominates:

```
for each position i:
    contributions = [|strength_k × diff_k[i]| for each LoRA k]
    dominant = argmax(contributions)
    if contributions[dominant] ≥ sum(others) × lambda:
        mark as selfish (belongs to LoRA dominant)

1. Separate selfish weights from consensus pool
2. Run merge algorithm on consensus weights only
3. Add selfish weights back to the result
```

Threshold `lambda=1.0` — a LoRA's contribution must exceed all others combined to be considered selfish. This protects unique features from being averaged away during consensus merging.

### KnOTS SVD Alignment (maximum)

*Reference: [Ramé et al., 2024](https://arxiv.org/abs/2407.09095)*

Projects all diffs into a shared singular value basis for better comparability:

```
1. Concatenate diffs column-wise: M = [diff_1 | diff_2 | ... | diff_N]
2. Compute truncated SVD: M ≈ U × S × Vᵀ  (rank ≤ 256)
3. For each diff k:
       proj_k = Uᵀ × diff_k         # project into shared basis
       aligned_k = U × proj_k        # reconstruct in aligned space
4. Use aligned diffs for all subsequent operations
```

This makes diffs more directly comparable by representing them in a shared coordinate system. Falls back to CPU on GPU OOM; skips gracefully if both fail.
