# Upstream Sync + Merge Correctness Notes

This branch is the local correctness refactor rebased onto upstream `main` after the post-v1.3 changes that landed through `dfd3920`.

## Why this exists

The original local work fixed real correctness issues that were not limited to the old overwrite bug:

- analysis and merge were still prefix-centric instead of target-key-centric,
- linear merges were being materialized in dense space and recompressed unnecessarily,
- auto-strength used proxy norms instead of exact streamed branch energy,
- `LoRAConflictEditor` dropped `key_filter`,
- saved alias selection was not deterministic.

Upstream moved after that work and added new runtime behavior that had to be preserved:

- `LoRAOptimizer` / `LoRAAutoTuner` bridge contract (`tuner_data`, `settings_source`, `output_mode`),
- full-rank aware merge safeguards,
- `LoRACompatibilityAnalyzer`,
- Z-Image fused-QKV save fixes,
- folder-aware tuner-data saving.

The point of this branch is to keep both sets of changes at once instead of choosing one and regressing the other.

## What changed

### Merge correctness

- analysis and merge now group aliases by resolved `(is_clip, target_key)` before Pass 1 and Pass 2,
- linear `weighted_sum` / `weighted_average` / `normalize` paths stay exact in low-rank form when possible,
- auto-strength uses streamed Frobenius norms and pairwise dots instead of mean-per-key proxies,
- mixed-trainer collisions remain accumulated defensively at collection time, but that is now a guard rail rather than the primary correctness path,
- `LoRAConflictEditor` keeps `key_filter`,
- `SaveMergedLoRA` uses canonical prefixes and adaptive rank estimation.

### Upstream compatibility kept

- `LoRAOptimizer` still exposes `tuner_data` and `settings_source`,
- `LoRAAutoTuner` still exposes `output_mode`,
- the bridge JS remains compatible with the upstream workflow surface,
- full-rank thresholds and SLERP/additive gates are preserved,
- `LoRACompatibilityAnalyzer` is available again,
- `SaveTunerData` matches the newer folder-aware `.tuner` flow,
- Z-Image fused-QKV paths keep dtype preservation during re-fusion/saving.

### Documentation

- README and wiki text were kept aligned with the actual implementation,
- overstated claims were tightened,
- the newer upstream bridge / saver / analyzer behavior is documented again.

## Validation

Minimum validation expected for this branch:

- `python3 -m py_compile lora_optimizer.py tests/test_lora_optimizer.py`
- `python -m unittest discover -s tests -v`
- `git diff --check`

Targeted regression coverage was added for:

- alias-group correctness,
- exact low-rank linear merges,
- exact auto-strength math,
- bridge/widget compatibility,
- deterministic save prefixes,
- safe save-path handling,
- optimizer `TUNER_DATA` output exposure,
- compatibility-analyzer node registration.

## Review focus

Review these areas first:

1. `lora_optimizer.py`
   - target-group analysis / merge path
   - exact linear merge path
   - full-rank gating
   - saver behavior
   - bridge return contract
2. `tests/test_lora_optimizer.py`
3. `js/lora_optimizer_bridge.js`
4. `README.md` and `docs/wiki/Nodes.md`
