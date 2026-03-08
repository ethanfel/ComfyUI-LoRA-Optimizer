# Reviewer Summary

This note is only about the changes in this PR and why they exist.

## What this PR is trying to fix

The requested changes, reduced to what actually matters in code review, were:

1. **Analyze and merge by resolved target weight, not raw LoRA prefix.**
   Different trainer formats can produce different prefixes that still map to the same model weight. Late accumulation avoids overwrite loss, but Pass 1 / AutoTuner still misreads true overlap if analysis stays prefix-based.

2. **Keep linear merges exact in low-rank form when possible.**
   `weighted_sum`, `weighted_average`, and `normalize` do not need dense expansion + recompression if the result can still be represented exactly by concatenating low-rank factors.

3. **Use exact streamed energy for auto-strength.**
   Auto-strength should be based on accumulated Frobenius norms and pairwise dots, not mean-per-key proxy norms.

4. **Fix concrete implementation bugs.**
   - `LoRAConflictEditor` dropped `key_filter`
   - alias/canonical save naming was not deterministic
   - save logic needed better rank selection and safer prefix handling

5. **Keep current upstream behavior intact while making those changes.**
   The branch must stay compatible with current upstream nodes/workflows: bridge inputs, full-rank behavior, compatibility analyzer, current saver behavior.

---

## What changed in code

### 1. Target-key-aware analysis and merge

**Why**

This is the main correctness change. It fixes the case where two different aliases map to the same real target weight but were previously analyzed as separate prefixes.

**What changed**

- aliases are collected and grouped by resolved `(is_clip, target_key)`
- Pass 1 metrics are computed per resolved group
- Pass 2 merges per resolved group
- late collision accumulation stays in place as a fallback, not as the primary correctness path

**Where to review**

- `lora_optimizer.py`
  - `_collect_lora_prefixes`
  - `_resolve_target_key`
  - `_build_target_groups`
  - `_run_group_analysis`
  - `optimize_merge`

---

### 2. Exact low-rank path for linear merges

**Why**

For linear merge modes, dense expansion + SVD is unnecessary when the merged result is still exactly representable in low-rank form.

**What changed**

- added an exact linear path that builds merged patches by concatenating low-rank factors
- falls back to the dense path when exact composition is not valid for the patch form

**Where to review**

- `lora_optimizer.py`
  - `_build_exact_linear_patch`
  - `_merge_one_group` inside `LoRAOptimizer.optimize_merge`

---

### 3. Exact streamed auto-strength

**Why**

The old path used norm proxies. This change makes auto-strength use streamed quantities that actually correspond to the merged branch energy.

**What changed**

- accumulate per-LoRA norm squares and pairwise dot products
- compute model and CLIP scaling from those accumulated values
- keep separate model/CLIP handling

**Where to review**

- `lora_optimizer.py`
  - `_compute_branch_auto_scale`
  - `_compute_auto_strengths`

---

### 4. Decision metrics and smoothing

**Why**

Raw sign conflict alone is noisy. This PR keeps the refactor’s improved decision inputs and smoothing in the upstream-based branch.

**What changed**

- carries forward weighted conflict, expected conflict baseline, excess conflict, and subspace overlap
- keeps optional decision smoothing and optional activation-aware importance inputs
- uses those values in per-group decision metrics

**Where to review**

- `lora_optimizer.py`
  - `_compute_pair_metrics`
  - `_smooth_group_decisions`
  - `_auto_select_params`
  - `LoRAOptimizer.optimize_merge`

---

### 5. Saver and small bug fixes

**Why**

These are concrete defects or cleanup items that fall directly out of the requested review.

**What changed**

- `LoRAConflictEditor` now preserves `key_filter`
- `SaveMergedLoRA` uses deterministic canonical prefixes
- adaptive save-rank estimation is kept
- save-path handling remains safe
- canonical prefix lookup handles alias-collapsed targets correctly

**Where to review**

- `lora_optimizer.py`
  - `LoRAConflictEditor`
  - `SaveMergedLoRA`
  - `SaveTunerData`

---

### 6. Upstream behavior deliberately kept

These are not “new ideas” in this PR. They are things this branch intentionally keeps compatible with current upstream while applying the correctness changes above.

- `LoRAOptimizer` `tuner_data` / `settings_source`
- `LoRAAutoTuner` `output_mode`
- current bridge JS behavior
- current full-rank handling
- `LoRACompatibilityAnalyzer`
- current folder-aware tuner-data saving flow

**Where to review**

- `lora_optimizer.py`
- `js/lora_optimizer_bridge.js`

---

## Docs changes

Docs were updated only to match the actual code in this branch:

- describe target-group behavior instead of pure prefix language
- describe AutoTuner as ranked/proxy-based, not objectively best
- describe saver behavior and current node surfaces accurately

Primary files:

- `README.md`
- `docs/wiki/Nodes.md`
- `docs/wiki/Home.md`
- `docs/wiki/How-It-Works.md`
- `docs/wiki/Tips-and-Troubleshooting.md`
- `docs/wiki/Workflows.md`

---

## Tests added or updated

The tests are targeted at the behavior this PR changes.

Covered areas:

- alias-group analysis/merge correctness
- exact linear merge reconstruction
- exact auto-strength math
- excess-conflict / subspace metrics
- `key_filter` preservation
- canonical save prefixes
- safe save-path handling
- bridge/workflow compatibility
- optimizer `TUNER_DATA` output exposure
- compatibility analyzer registration

**Where to review**

- `tests/test_lora_optimizer.py`

---

## Review order

1. `lora_optimizer.py`
2. `tests/test_lora_optimizer.py`
3. `js/lora_optimizer_bridge.js`
4. README/wiki updates

If reviewing for substance only, the key question is:

> Does this PR make Pass 1 / AutoTuner / Pass 2 operate on the same resolved target weight when aliases collide?

That is the main behavior change.
