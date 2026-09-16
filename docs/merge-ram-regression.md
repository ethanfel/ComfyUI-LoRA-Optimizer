# Merge RAM regression investigation

Status: local implementation and synthetic validation, 2026-09-16. Not a new release yet.

## Report and confirmed causes

The reported 128 GB machine used roughly 69% RAM with two ordinary H3 LoRA
loaders versus 98% after the optimized merge, including sampling. Five-candidate
AutoTuner sweeps with full SVD could exhaust RAM. The percentages alone do not
identify which objects account for the roughly 37 GB difference.

Confirmed in the code and controlled reproductions:

- A cache miss on the same base model retained the previous cached merge until
  the replacement finished. Weak-reference tests reproduced it across all five
  candidate merges and the final merge. This increases **retuning peak** memory;
  it is not by itself an explanation of all post-return RAM.
- Discarded candidates still retained full dense patch sets after inline scoring.
  Cross-candidate caches could also retain these tensors, although the later
  scoring pass only needed the measured statistics.
- Pairwise SLERP could emit dense output despite remaining in the linear span
  of the original two low-rank updates. Native-plus-sliced QKV collisions also
  expanded two factored additive contributions to dense output.
- An automatic diff-cache admission check happened after making the CPU copy.

## Changes

1. Release obsolete node-owned cached outputs before new analysis/candidate work
   and before cached-selection replay. Valid complete cache hits still work.
   A tuning-only cache hit no longer pins the cached patched output. Downstream
   ComfyUI caches and input patchers are not mutated.
2. Discarded, uncached, non-evaluator candidates may retain tensor-free dense
   score records. Measure the original native-dtype delta on the requested
   device, using the existing scoring helper. Keep native/sliced QKV inputs
   until all contributors to that projection arrive, then assemble and score
   it. Do not cache those dense QKV inputs across candidates. Factor scoring
   and its existing column sampling are unchanged.
3. Rebuild the final output after clearing sweep caches, including a one-candidate
   sweep. Eligible unmodified two-LoRA SLERP groups use the actual signed,
   sorted, norm-corrected coefficients to concatenate the original factors.
   Accept only smaller storage and full-matrix parity within 2e-5 relative
   Frobenius error, checked in bounded row chunks. No truncated SVD is introduced.
   Candidate scoring deliberately retains its former representation choice.
4. Keep native-plus-sliced additive QKV factor collisions factored when smaller
   on final output. The existing conservative refusion VRAM planner remains.
5. Reject over-budget automatic diff-cache entries before allocating a CPU copy.
   Explicit disk/RAM cache choices are not changed. Completed CPU futures no
   longer retain all previously collected result objects until pool exit.
6. Log resident patch tensor bytes. Correct the old message suggesting that
   disabling the patch cache also freed the patches required for sampling.

Score-only records are private to candidates that cannot be cached or applied.
External evaluators, returned MODEL/CLIP outputs, exports, and replay outputs
continue to receive actual patches. GPU scoring is not moved to CPU.

## Controlled comparison with 1.8.7

`tests/integration/merge_ram_stress.py` compares the released core classes at
`a844802` with the working implementation in separate processes. No ComfyUI
server, downloads, user weights, or generation jobs are involved.

RTX 5090, 2 GiB per-process CUDA allocation cap; 24 synthetic 1536×1536 targets,
two correlated rank-8 adapters, five distinct candidates, `scoring_svd="full"`.
Correlated factors ensure merge-quality SVD is actually enabled. No lossy patch
compression or diff cache. Base weights use meta tensors, so these numbers
exclude a real H3 model's loading/offloading costs.

| Measurement | 1.8.7 | Fixed |
|---|---:|---:|
| Retained dense candidate tensors (SLERP/TIES/consensus) | 216 MiB each | 0 MiB; scalar score records |
| Retained linear candidate factors | 4.5 MiB each | 4.5 MiB each |
| Separate eligible final SLERP output | 216 MiB | 4.5 MiB |
| Whole-test peak process RSS | 2029.5 MiB | 1962.6 MiB |
| Initial process RSS | 756.2 MiB | 685.0 MiB |

All five measured scores matched exactly in this run:

| Candidate | Score, both runs |
|---|---:|
| Weighted sum | 0.7575738587968999 |
| Weighted average | 0.7575738280599895 |
| SLERP | 0.7476678740059935 |
| TIES | 0.8745264679798531 |
| Consensus | 0.7489563006761201 |

The 48× SLERP reduction is **patch tensor storage**, not an equivalent reduction
in total ComfyUI RAM. Whole-test RSS includes CUDA host setup, allocator retention,
analysis and a genuinely dense final TIES winner; the initial RSS difference
is almost as large as the peak difference. This run therefore does **not**
establish a meaningful reduction in overall peak RSS. Timings are not claimed
as a speed benchmark. The user's complete H3 workflow has not been reproduced.

## Validation and remaining limits

- Weak-reference regressions cover changed settings, disabled cache, selection
  replay, tuning-only transitions, and valid cache hits.
- Compact SLERP tests cover signed/zero weights and FP32/FP16/BF16 factors,
  modified-input exclusions, non-finite/parity failures, and size rejection.
- Streamed statistics match conventional measurements across five merge modes,
  with and without SVD. QKV tests check native collisions, input release and
  requested GPU scoring even when refusion itself uses CPU.
- Real local ComfyUI round trips pass, including stock-loader inline parity,
  export/reload, SLERP, TIES, CLIP, and upstream patch preservation.

TIES, consensus, masks, refinements, and unsupported adapter formats can produce
genuinely dense final results. Three-or-more-LoRA SLERP is not compacted by the
new coefficient path. Interleaved or incomplete QKV groups remain pending until
their contributors finish. Full SVD still requires workspace for the current
projection; explicit unbounded RAM diff caching is still an opt-in risk.

The next real-workflow check is the resident-patch log, selected strategies,
adapter formats/ranks, and process RAM before merge, after merge, and during
sampling. Do not infer that all of the reported 37 GB excess has been eliminated.
