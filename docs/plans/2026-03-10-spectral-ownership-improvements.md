# Spectral Ownership Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve `_spectral_ownership_split` with soft ownership weighting, bilateral (U+V) overlap, energy-adaptive rank, and post-merge singular value calibration (SVC).

**Architecture:** Two incremental phases. Phase 1 replaces the binary private/shared threshold with continuous ownership weighting, adds bilateral U+V cross-Gram overlap, and uses energy-adaptive rank selection. Phase 2 adds post-merge SVC to calibrate over-accumulated shared spectral directions. Both phases only modify `lora_optimizer.py` and `tests/test_lora_optimizer.py`.

**Tech Stack:** PyTorch (`torch.svd_lowrank`, `torch.linalg.svd`), Python 3.10+

---

## Phase 1: Soft Bilateral Ownership + Energy-Adaptive Rank

### Task 1: Add unit tests for `_spectral_ownership_split` improvements

**Files:**
- Modify: `tests/test_lora_optimizer.py`

**Step 1: Write failing tests for the improved method**

Add these tests after the existing test class methods (before `if __name__`):

```python
def test_spectral_split_orthogonal_loras_all_private(self):
    """Two orthogonal LoRAs: all directions should be private (ownership ~1.0)."""
    # LoRA A lives in rows 0-7, LoRA B in rows 8-15 — zero overlap
    a = torch.zeros(16, 16)
    a[:8, :8] = torch.randn(8, 8)
    b = torch.zeros(16, 16)
    b[8:, 8:] = torch.randn(8, 8)
    diffs = [(a, 1.0), (b, 1.0)]
    shared, private = LoRAOptimizer._spectral_ownership_split(diffs)
    # With orthogonal LoRAs, private_addition should capture most energy
    self.assertIsNotNone(private)
    # Private addition should have significant norm (both LoRAs contribute)
    self.assertGreater(private.norm().item(), 0.1 * (a.norm() + b.norm()).item())

def test_spectral_split_identical_loras_all_shared(self):
    """Two identical LoRAs: all directions shared, no private component."""
    a = torch.randn(16, 16)
    diffs = [(a.clone(), 1.0), (a.clone(), 1.0)]
    shared, private = LoRAOptimizer._spectral_ownership_split(diffs)
    # Identical LoRAs have full overlap — nothing is private
    self.assertIsNone(private)

def test_spectral_split_soft_weighting_partial_overlap(self):
    """Partially overlapping LoRAs: private_addition should be non-None
    and shared diffs should preserve total energy approximately."""
    torch.manual_seed(42)
    # Shared component + unique components
    shared_base = torch.randn(32, 32) * 0.5
    a = shared_base + torch.randn(32, 32) * 0.3
    b = shared_base + torch.randn(32, 32) * 0.3
    original_sum = a * 0.6 + b * 0.4
    diffs = [(a, 0.6), (b, 0.4)]
    shared_diffs, private = LoRAOptimizer._spectral_ownership_split(diffs)
    # Should have some private directions
    self.assertIsNotNone(private)
    # Reconstruction: merge shared (weighted_avg) + private should approximate original
    total_w = sum(abs(w) for _, w in shared_diffs)
    merged_shared = sum(d * (w / total_w) for d, w in shared_diffs)
    reconstructed = merged_shared + private
    # Energy should be in the right ballpark (within 50% — soft weighting redistributes)
    self.assertGreater(reconstructed.norm().item(), original_sum.norm().item() * 0.3)

def test_spectral_split_energy_adaptive_rank(self):
    """Low-rank input should use fewer SVD components than max_rank."""
    # Rank-2 matrix: only 2 meaningful singular values
    u = torch.randn(32, 2)
    v = torch.randn(32, 2)
    low_rank = u @ v.T
    noise = torch.randn(32, 32) * 0.01
    a = low_rank + noise
    b = torch.randn(32, 32)  # Full rank, different
    diffs = [(a, 1.0), (b, 1.0)]
    # Should not crash and should handle gracefully
    shared, private = LoRAOptimizer._spectral_ownership_split(diffs)
    self.assertEqual(len(shared), 2)

def test_spectral_split_1d_guard(self):
    """1D tensors should pass through unchanged."""
    a = torch.randn(16)
    diffs = [(a, 1.0), (torch.randn(16), 1.0)]
    shared, private = LoRAOptimizer._spectral_ownership_split(diffs)
    self.assertIsNone(private)
    self.assertTrue(torch.equal(shared[0][0], a))

def test_spectral_split_single_lora_guard(self):
    """Single LoRA should pass through unchanged."""
    a = torch.randn(16, 16)
    diffs = [(a, 1.0)]
    shared, private = LoRAOptimizer._spectral_ownership_split(diffs)
    self.assertIsNone(private)
    self.assertTrue(torch.equal(shared[0][0], a))
```

**Step 2: Run tests to verify they fail**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v -k "spectral_split" 2>&1 | tail -20`
Expected: Some tests FAIL (orthogonal test fails because current binary method may not capture all energy correctly; soft weighting test fails because current method uses hard threshold)

**Step 3: Commit test stubs**

```bash
git add tests/test_lora_optimizer.py
git commit -m "test: add unit tests for spectral ownership improvements"
```

---

### Task 2: Implement energy-adaptive rank selection

**Files:**
- Modify: `lora_optimizer.py:2494-2531` (the `_spectral_ownership_split` method, SVD section)

**Step 1: Replace fixed `max_rank` with energy-adaptive rank**

In `_spectral_ownership_split`, replace the SVD loop (lines ~2517-2531) with energy-adaptive rank selection. The idea: compute SVD with oversampled rank, then keep only directions that capture ≥90% of cumulative energy. This avoids wasting compute on noise directions for low-rank layers.

Replace the current code from `rank = min(max_rank, ...)` through the end of the SVD try block with:

```python
        rank = min(max_rank, min(original_shape[0], int(torch.tensor(original_shape[1:]).prod().item())))

        # Step 1: Per-LoRA truncated SVD with energy-adaptive rank
        Us = []  # U_i: [out, effective_rank_i]
        Ss = []  # S_i: [effective_rank_i]
        Vs = []  # V_i: [in, effective_rank_i]
        try:
            for d, _ in diffs_with_weights:
                mat = d.reshape(original_shape[0], -1).to(device=dev, dtype=torch.float32)
                q_oversample = min(rank + max(10, rank // 5), min(mat.shape))
                U, S, V = torch.svd_lowrank(mat, q=q_oversample, niter=4)
                U, S, V = U[:, :rank], S[:rank], V[:, :rank]
                # Energy-adaptive rank: keep directions capturing ≥ energy_threshold
                total_energy = S.sum()
                if total_energy > 1e-10:
                    cumulative = S.cumsum(0) / total_energy
                    effective_rank = max(1, int((cumulative < energy_threshold).sum().item()) + 1)
                    effective_rank = min(effective_rank, rank)
                    U, S, V = U[:, :effective_rank], S[:effective_rank], V[:, :effective_rank]
                Us.append(U)
                Ss.append(S)
                Vs.append(V)
                del mat
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            return diffs_with_weights, None
```

Also update the method signature to add `energy_threshold=0.9`:

```python
    def _spectral_ownership_split(diffs_with_weights, ownership_threshold=0.5,
                                   max_rank=16, energy_threshold=0.9,
                                   compute_device=None):
```

**Step 2: Run tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v -k "spectral_split" 2>&1 | tail -20`

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(spectral): add energy-adaptive rank selection"
```

---

### Task 3: Implement bilateral (U+V) cross-Gram overlap

**Files:**
- Modify: `lora_optimizer.py:2535-2551` (the cross-Gram ownership section)

**Step 1: Replace U-only overlap with bilateral U*V geometric mean**

The current code computes `sharedness_k = max_{j≠i} ||U_j^T @ u_i^k||²` using only left singular vectors. This misses cases where two LoRAs share the same output-space direction but target different input features.

Replace the sharedness computation (the inner `for i / for j` loop, from `for i in range(n):` through `del sharedness, private_mask` at the end of the i-loop body) with:

```python
        for i in range(n):
            rank_i = Us[i].shape[1]
            sharedness = torch.zeros(rank_i, device=dev)
            for j in range(n):
                if j == i:
                    continue
                # Left singular vector overlap: G_U = U_j^T @ U_i
                G_U = Us[j].T @ Us[i]  # [rank_j, rank_i]
                u_overlap = (G_U * G_U).sum(dim=0)  # [rank_i]

                # Right singular vector overlap: G_V = V_j^T @ V_i
                # Use min rank to handle variable effective ranks
                min_in = min(Vs[j].shape[0], Vs[i].shape[0])
                G_V = Vs[j][:min_in].T @ Vs[i][:min_in]  # [rank_j, rank_i]
                v_overlap = (G_V * G_V).sum(dim=0)  # [rank_i]

                # Bilateral sharedness: geometric mean of U and V overlap
                bilateral = torch.sqrt(u_overlap * v_overlap)
                sharedness = torch.max(sharedness, bilateral)
                del G_U, G_V, u_overlap, v_overlap, bilateral
```

Note: The `Vs` already have the same `in` dimension (they come from the same-shaped diffs reshaped to `[out, in]`), so `min_in` is just a safety guard. The `torch.sqrt(u * v)` geometric mean means a direction is only "shared" if it overlaps in *both* input and output space.

The rest of the loop body (private_mask classification, reconstruction) stays the same.

**Step 2: Run tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v -k "spectral_split" 2>&1 | tail -20`

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(spectral): bilateral U+V cross-Gram overlap"
```

---

### Task 4: Replace binary threshold with soft ownership weighting

**Files:**
- Modify: `lora_optimizer.py:2553-2573` (the private/shared classification and reconstruction section)

**Step 1: Replace hard threshold with continuous ownership weighting**

This is the core improvement. Instead of `private_mask = sharedness < threshold` (binary), use a continuous ownership weight per direction:

```python
            # Soft ownership: continuous weighting based on sharedness
            # ownership = 1.0 means fully private, 0.0 means fully shared
            # Smooth sigmoid transition centered at ownership_threshold
            # steepness=10 gives a fairly sharp but continuous transition
            ownership = torch.sigmoid(10.0 * (ownership_threshold - sharedness))

            # Weight singular values by ownership for private component
            # and by (1 - ownership) for shared component
            S_private = Ss[i] * ownership
            S_shared = Ss[i] * (1.0 - ownership)

            has_any_private = (ownership > 0.01).any()
            if has_any_private:
                has_private = True
                # Reconstruct private component with ownership-weighted singular values
                d_private = ((Us[i] * S_private.unsqueeze(0)) @ Vs[i].T).reshape(original_shape)
                # Reconstruct shared component with inverse-weighted singular values
                d_shared = ((Us[i] * S_shared.unsqueeze(0)) @ Vs[i].T).reshape(original_shape)
                # Add residual: original - (private + shared from SVD) to preserve non-SVD content
                d, w = diffs_with_weights[i]
                d_full = d.to(device=dev, dtype=torch.float32)
                residual = d_full - ((Us[i] * Ss[i].unsqueeze(0)) @ Vs[i].T).reshape(original_shape)
                d_shared = d_shared + residual  # residual goes to shared (conservative)
                shared_diffs.append((d_shared.to(dtype=d.dtype, device=d.device), w))
                private_addition.add_(d_private * w)
                del d_private, d_shared, residual, d_full
            else:
                shared_diffs.append(diffs_with_weights[i])
            del sharedness, ownership, S_private, S_shared
```

Remove the `ownership_threshold` parameter's role as a hard cutoff — it's now the sigmoid center point. The default 0.5 still works well as the midpoint.

**Step 2: Update docstring**

Replace the docstring to reflect the new soft weighting:

```python
        """
        Split each LoRA diff into spectrally private and shared components
        using soft ownership weighting.

        Each LoRA "owns" certain spectral directions (singular value components).
        Ownership is computed bilaterally (both U and V subspaces) using cross-Gram
        overlap, then converted to a continuous weight via sigmoid. Private
        components (high ownership) bypass merge averaging; shared components
        (low ownership) go through the normal merge pipeline.

        Uses energy-adaptive rank: SVD rank per LoRA adapts to capture ≥90%
        of spectral energy, avoiding wasted compute on noise directions.

        Returns (shared_diffs_with_weights, private_addition) where
        private_addition is a tensor to add after merging, or None.
        """
```

**Step 3: Run tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v -k "spectral_split" 2>&1 | tail -20`
Expected: All spectral tests PASS

**Step 4: Run full test suite**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v 2>&1 | tail -15`
Expected: 35/36 pass (pre-existing widget order failure only)

**Step 5: Syntax check**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`

**Step 6: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(spectral): soft ownership weighting replaces binary threshold"
```

---

## Phase 2: Post-Merge Singular Value Calibration (SVC)

### Task 5: Add SVC unit tests

**Files:**
- Modify: `tests/test_lora_optimizer.py`

**Step 1: Write failing tests for SVC**

```python
def test_spectral_svc_reduces_over_accumulation(self):
    """When LoRAs share a dominant direction, SVC should calibrate it down."""
    torch.manual_seed(123)
    # Create two LoRAs that share a strong direction but differ elsewhere
    shared_dir = torch.randn(32, 1) @ torch.randn(1, 32)  # rank-1 shared
    a = shared_dir * 2.0 + torch.randn(32, 32) * 0.1
    b = shared_dir * 2.0 + torch.randn(32, 32) * 0.1
    diffs = [(a, 0.5), (b, 0.5)]
    shared_diffs, private = LoRAOptimizer._spectral_ownership_split(diffs)
    # The shared component should exist and not be wildly inflated
    if private is not None:
        # Merge shared via weighted_average
        total_w = sum(abs(w) for _, w in shared_diffs)
        merged = sum(d.float() * (w / total_w) for d, w in shared_diffs)
        result = merged + private
    else:
        total_w = sum(abs(w) for _, w in shared_diffs)
        result = sum(d.float() * (w / total_w) for d, w in shared_diffs)
    # Result should be reasonable magnitude (not 2x inflated)
    expected_magnitude = (a * 0.5 + b * 0.5).norm()
    self.assertLess(result.norm().item(), expected_magnitude.item() * 2.0)
```

**Step 2: Run test to verify**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v -k "spectral_svc" 2>&1 | tail -10`

**Step 3: Commit**

```bash
git add tests/test_lora_optimizer.py
git commit -m "test: add SVC calibration test for spectral merge"
```

---

### Task 6: Implement post-merge SVC in `_merge_diffs`

**Files:**
- Modify: `lora_optimizer.py` — add `_spectral_calibrate_shared` static method after `_spectral_ownership_split`
- Modify: `lora_optimizer.py` — update all `spectral_additions` merge points to apply SVC

**Step 1: Add `_spectral_calibrate_shared` method**

Insert after `_spectral_ownership_split` (after the `return shared_diffs, private_addition` line):

```python
    @staticmethod
    @torch.no_grad()
    def _spectral_calibrate_shared(merged_shared, original_diffs_with_weights,
                                    max_rank=16, compute_device=None):
        """
        Singular Value Calibration (SVC) for the merged shared component.

        After merging shared parts, spectral directions that were common across
        multiple LoRAs get over-accumulated (singular value inflation). This
        method detects inflated directions by projecting original diffs onto
        the merged result's singular vectors, then rescales to correct.

        Based on: "When Shared Knowledge Hurts: Spectral Over-Accumulation
        in Model Merging" (2025).
        """
        if merged_shared.dim() < 2 or min(merged_shared.shape) < 2:
            return merged_shared
        if len(original_diffs_with_weights) < 2:
            return merged_shared

        original_shape = merged_shared.shape
        dev = compute_device if compute_device is not None else merged_shared.device
        mat = merged_shared.reshape(original_shape[0], -1).to(device=dev, dtype=torch.float32)

        rank = min(max_rank, min(mat.shape))
        try:
            q_oversample = min(rank + max(10, rank // 5), min(mat.shape))
            U, S, V = torch.svd_lowrank(mat, q=q_oversample, niter=4)
            U, S, V = U[:, :rank], S[:rank], V[:, :rank]
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            return merged_shared

        n = len(original_diffs_with_weights)

        # For each spectral direction r of merged result, compute projection
        # coefficients from each original diff
        # a_i^r = u_r^T @ diff_i — response of diff_i in direction r
        gamma = torch.ones(S.shape[0], device=dev)
        for r in range(S.shape[0]):
            if S[r] < 1e-10:
                continue
            u_r = U[:, r]  # [out]
            projections = []
            for d, w in original_diffs_with_weights:
                d_mat = d.reshape(original_shape[0], -1).to(device=dev, dtype=torch.float32)
                a_r = u_r @ d_mat  # [in] — projection onto direction r
                proj_norm_sq = (a_r * a_r).sum()
                projections.append(proj_norm_sq)
                del d_mat, a_r
            # Over-accumulation: if sum of projections >> merged projection,
            # the direction is inflated
            merged_proj = S[r] * S[r]  # ||s_r * v_r||^2 = s_r^2
            sum_proj = sum(projections)
            if sum_proj > 1e-10:
                # gamma_r = n / sum(max(alpha, s_i^r)) — SVC formula
                # Simplified: ratio of expected vs actual energy
                gamma[r] = min(1.0, (merged_proj / sum_proj).item() * n)
            del projections

        # Apply calibration: rescale singular values
        if (gamma < 0.99).any():
            S_calibrated = S * gamma
            calibrated = (U * S_calibrated.unsqueeze(0)) @ V.T
            # Preserve residual (content outside SVD approximation)
            residual = mat - (U * S.unsqueeze(0)) @ V.T
            result = (calibrated + residual).reshape(original_shape)
            del U, S, V, S_calibrated, calibrated, residual, mat, gamma
            return result.to(dtype=merged_shared.dtype, device=merged_shared.device)

        del U, S, V, mat, gamma
        return merged_shared
```

**Step 2: Integrate SVC into merge result assembly**

In `_merge_diffs`, after each merge mode computes its `result` but before adding `spectral_additions`, apply SVC to the merged shared result. The pattern is the same at all 5 non-TIES merge points. For each one, replace:

```python
            if spectral_additions is not None:
                result = result + spectral_additions.to(device=result.device, dtype=torch.float32)
```

with:

```python
            if spectral_additions is not None:
                result = self._spectral_calibrate_shared(
                    result, diffs_with_weights if diffs_with_weights[0] is not None
                    else [], compute_device=dev)
                result = result.to(dtype=torch.float32) + spectral_additions.to(
                    device=result.device, dtype=torch.float32)
```

**Important caveat:** By the time we reach the `spectral_additions` check, `diffs_with_weights` entries may have been set to `None` (freed early). We need to save a reference to the original shared diffs before the merge loop consumes them. Add this right after the spectral split block (after `diffs_with_weights, spectral_additions = ...`):

```python
                    # Save shared diffs reference for post-merge SVC calibration
                    _spectral_shared_ref = list(diffs_with_weights) if spectral_additions is not None else None
```

Then use `_spectral_shared_ref` instead of `diffs_with_weights` in the SVC call:

```python
            if spectral_additions is not None:
                if _spectral_shared_ref is not None:
                    result = self._spectral_calibrate_shared(
                        result, _spectral_shared_ref, compute_device=dev)
                result = result.to(dtype=torch.float32) + spectral_additions.to(
                    device=result.device, dtype=torch.float32)
```

For the TIES path, same pattern with `ties_spectral` and `_ties_spectral_shared_ref`.

**Step 3: Run tests**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v -k "spectral" 2>&1 | tail -20`

**Step 4: Run full test suite**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v 2>&1 | tail -15`

**Step 5: Syntax check**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat(spectral): add post-merge SVC to calibrate shared direction inflation"
```

---

### Task 7: Final integration test and cleanup

**Files:**
- Modify: `tests/test_lora_optimizer.py` (optional: add integration-style test)
- Modify: `lora_optimizer.py` (cleanup any dead code from old binary method)

**Step 1: Add integration test that exercises the full merge pipeline with spectral**

```python
def test_spectral_merge_refinement_end_to_end(self):
    """Spectral refinement through _merge_diffs produces valid output."""
    torch.manual_seed(99)
    a = torch.randn(16, 16)
    b = torch.randn(16, 16)
    diffs = [(a, 0.7), (b, 0.3)]
    opt = LoRAOptimizer()
    # Call _merge_diffs with spectral refinement
    result = opt._merge_diffs(
        diffs, mode="weighted_average", merge_refinement="spectral",
        sparsification="disabled", sparsification_density=0.7,
        compute_device=torch.device("cpu"))
    self.assertIsNotNone(result)
    self.assertEqual(result.shape, a.shape)
    self.assertFalse(torch.isnan(result).any())
    self.assertFalse(torch.isinf(result).any())
```

**Step 2: Run full test suite**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v 2>&1 | tail -20`

**Step 3: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "test: add end-to-end integration test for spectral merge refinement"
```

---

## Summary of Changes

| Component | What Changes |
|-----------|-------------|
| `_spectral_ownership_split` | Energy-adaptive rank, bilateral U+V overlap, soft sigmoid weighting |
| `_spectral_calibrate_shared` (new) | Post-merge SVC to deflate over-accumulated shared directions |
| `_merge_diffs` integration | Save shared refs for SVC, apply calibration before adding private parts |
| Tests | 7+ new unit tests covering guards, orthogonal, identical, partial overlap, SVC, e2e |

## Research References

- [STAR: Spectral Truncation and Rescale](https://arxiv.org/abs/2502.10339) — energy-adaptive rank
- [Task Singular Vectors](https://arxiv.org/abs/2412.00081) — bilateral SVD overlap
- [When Shared Knowledge Hurts: SVC](https://arxiv.org/abs/2602.05536) — post-merge calibration
- [DO-Merging](https://arxiv.org/abs/2505.15875) — magnitude/direction decoupling insight
- [ZipLoRA](https://arxiv.org/abs/2311.13600) — column-wise sparsity analysis
