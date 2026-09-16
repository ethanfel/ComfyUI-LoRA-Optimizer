"""No-op sparsification must not win just by changing patch representation."""
import gc
from unittest import mock

import pytest
import torch

from tests.test_lora_optimizer import lora_optimizer as m
from tests.test_phase1_correctness import ToyPatcher, entry


def adapter(rows=96, cols=257, rank=16, seed=314, dtype=torch.float32, scale=1.):
    gen = torch.Generator().manual_seed(seed)
    return m.LoRAAdapter(set(), (
        torch.randn(rows, rank, generator=gen).to(dtype),
        torch.randn(rank, cols, generator=gen).to(dtype),
        rank * scale, None, None, None))


def score(patch, key="layer.weight", **kwargs):
    return m._score_merge_result({key: patch}, {}, compute_svd=False, **kwargs)


@pytest.mark.parametrize("cols", [31, 64, 257])
@pytest.mark.parametrize("scale", [1., -.7, 0.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_same_dense_and_factored_delta_has_same_sampled_sparsity(cols, scale, dtype):
    patch = adapter(cols=cols, scale=scale, dtype=dtype)
    dense = m._LoRAMergeBase._expand_patch_to_diff(patch)
    factored_score = score(patch)
    dense_score = score(("diff", (dense,)))
    assert factored_score["sparsity_mean"] == dense_score["sparsity_mean"]
    assert factored_score == pytest.approx(dense_score, rel=2e-6, abs=1e-7)


def test_unsampled_outlier_cannot_change_only_the_dense_threshold():
    cols, key = 257, "outlier.weight"
    chosen = set(m._score_sample_columns(cols, key, torch.device("cpu")).tolist())
    missing = next(i for i in range(cols) if i not in chosen)
    up, down = torch.ones(96, 1), torch.ones(1, cols)
    down[0, missing] = 1000.
    patch = m.LoRAAdapter(set(), (up, down, 1., None, None, None))
    dense = up @ down
    assert (dense < dense.max() * .01).float().mean() > .99  # old dense rule
    assert score(patch, key)["sparsity_mean"] == 0.
    assert score(("diff", (dense,)), key)["sparsity_mean"] == 0.


def test_sampling_does_not_consume_global_rng_and_handles_offset_keys():
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    columns = m._score_sample_columns(1024, "qkv.weight", torch.device("cpu"))
    assert torch.equal(before, torch.random.get_rng_state())
    assert columns.unique().numel() == 64
    assert torch.equal(columns, m._score_sample_columns(1024, ("qkv.weight", (0, 0, 96)), torch.device("cpu")))
    assert not torch.equal(columns, m._score_sample_columns(1024, "other.weight", torch.device("cpu")))


def test_output_projection_rename_does_not_change_inline_samples():
    before = "diffusion_model.layers.0.attention.to_out.0.weight"
    after = "diffusion_model.layers.0.attention.out.weight"
    device = torch.device("cpu")
    assert torch.equal(m._score_sample_columns(257, before, device),
                       m._score_sample_columns(257, after, device))
    dense = m._LoRAMergeBase._expand_patch_to_diff(adapter())
    inline = {id(dense): (dense, m._diff_score_stats(dense, False, before))}
    assert score(("diff", (dense,)), after, _inline_stats=inline) == score(("diff", (dense,)), after)


def test_inline_and_streamed_stats_use_the_final_target_key():
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    dense = m._LoRAMergeBase._expand_patch_to_diff(adapter())
    patch = ("diff", (dense,))
    stats = m._diff_score_stats(dense, False, key)
    expected = score(patch, key)
    inline = {id(dense): (dense, stats)}
    assert score(patch, key, _inline_stats=inline) == expected
    scalar = m._score_only_patch(patch, {"compute_svd": False}, target_key=key)
    assert score(scalar, key) == expected


@pytest.fixture
def merge_case(tmp_path, monkeypatch):
    model = ToyPatcher()
    model.model.layer = torch.nn.Linear(257, 96, bias=False)
    monkeypatch.setattr(m._LoRAMergeBase, "_get_compute_device", lambda self: torch.device("cpu"))
    monkeypatch.setattr(m.comfy.lora, "model_lora_keys_unet", lambda *a: {"layer": "layer.weight"})
    monkeypatch.setattr(m.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    monkeypatch.setattr(m, "AUTOTUNER_MEMORY_DIR", str(tmp_path))
    return model


def stack(captured, dtype=torch.float32):
    result = []
    for i in range(2):
        patch = adapter(seed=314 + i, dtype=dtype)
        if captured:
            item = entry({"layer.weight": patch}, name=f"captured-{i}", _precomputed_diffs=True)
        else:
            up, down, alpha = patch.weights[:3]
            item = entry({"layer.lora_A.weight": down, "layer.lora_B.weight": up,
                          "layer.alpha": torch.tensor(alpha)}, name=f"file-{i}")
        item["strength"] = .9 - i * .1
        result.append(item)
    return result


@pytest.mark.parametrize("cache_mode", ["disabled", "auto", "ram", "disk"])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("mode", ["weighted_sum", "weighted_average", "slerp"])
@pytest.mark.parametrize("sparsification", ["dare_conflict", "della_conflict"])
def test_noop_matches_baseline_with_cold_and_warm_cache(merge_case, cache_mode, captured, mode, sparsification):
    items = stack(captured)
    cache = None if cache_mode == "disabled" else m._DiffCache(mode=cache_mode)
    opt = m.LoRAOptimizer()
    options = dict(optimization_mode="global", merge_strategy_override=mode,
                   patch_compression="disabled", cache_patches="disabled",
                   _compact_output=False, _skip_model_apply=True, _skip_report=True)
    expected = opt.optimize_merge(merge_case, items, 1., **options)[4]["model_patches"]
    expected_score = m._score_merge_result(expected, {}, compute_svd=True, lora_svd=True)
    try:
        for attempt in range(2):
            data = opt.optimize_merge(merge_case, items, 1., **options,
                                      sparsification=sparsification, _diff_cache=cache)[4]
            assert data["sparsification_summary"] == {"applied_groups": 0, "skipped_conflict_groups": 1}
            actual = data["model_patches"]
            assert type(actual["layer.weight"]) is type(expected["layer.weight"])
            assert m._score_merge_result(actual, {}, compute_svd=True, lora_svd=True) == expected_score
            torch.testing.assert_close(opt._expand_patch_to_diff(actual["layer.weight"]),
                                       opt._expand_patch_to_diff(expected["layer.weight"]), rtol=0, atol=0)
    finally:
        if cache is not None:
            cache.clear()


@pytest.mark.parametrize("cache_mode", ["auto", "ram", "disk"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cache_hit_preserves_source_storage_dtype(merge_case, cache_mode, dtype):
    opt = m.LoRAOptimizer()
    items = stack(False, dtype=dtype)
    cache = m._DiffCache(mode=cache_mode)
    options = dict(optimization_mode="global", merge_strategy_override="slerp",
                   patch_compression="disabled", cache_patches="disabled",
                   _compact_output=False, _skip_model_apply=True, _skip_report=True)
    try:
        expected = opt.optimize_merge(merge_case, items, 1., **options)[4]["model_patches"]["layer.weight"]
        for attempt in range(2):
            actual = opt.optimize_merge(merge_case, items, 1., **options, _diff_cache=cache)[4]["model_patches"]["layer.weight"]
            assert actual[1][0].dtype == expected[1][0].dtype == dtype
            torch.testing.assert_close(actual[1][0], expected[1][0], rtol=0, atol=0)
    finally:
        cache.clear()


@pytest.mark.parametrize("cache_mode", ["disabled", "auto", "ram", "disk"])
@pytest.mark.parametrize("captured", [False, True])
def test_real_sweep_noop_candidates_tie_including_dampening(merge_case, cache_mode, captured, monkeypatch):
    base = dict(merge_mode="weighted_average", optimization_mode="global", auto_strength="disabled",
                sparsification="disabled", sparsification_density=.7, dare_dampening=0.,
                merge_refinement="none", strategy_set="full")
    grid = [base, dict(base, sparsification="dare_conflict"),
            dict(base, sparsification="dare_conflict", dare_dampening=.3),
            dict(base, sparsification="della_conflict")]
    monkeypatch.setattr(m, "_generate_param_grid", lambda **kwargs: grid)
    result = m.LoRAAutoTuner().auto_tune(merge_case, stack(captured), 1., top_n=4,
                scoring_device="cpu", scoring_svd="full", diff_cache_mode=cache_mode,
                memory_mode="disabled", cache_patches="disabled", community_cache="disabled",
                output_mode="tuning_only")
    rows = result[4]["top_n"]
    assert len(rows) == 4
    baseline = next(r for r in rows if r["config"]["sparsification"] == "disabled")
    for row in rows:
        assert row["score_measured"] == baseline["score_measured"]
        assert row["metrics"]["sparsity_mean"] == baseline["metrics"]["sparsity_mean"]
        if row is not baseline:
            assert row["metrics"]["sparsification_summary"]["applied_groups"] == 0
    assert "Sampled sparsity:" in result[2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_uses_same_columns_and_keeps_compute_on_gpu():
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    cpu_columns = m._score_sample_columns(1024, key, torch.device("cpu"))
    torch.cuda.manual_seed_all(19)
    state = torch.cuda.get_rng_state().clone()
    gpu_columns = m._score_sample_columns(1024, key, torch.device("cuda"))
    assert torch.equal(state, torch.cuda.get_rng_state())
    assert torch.equal(cpu_columns, gpu_columns.cpu())
    patch = adapter(cols=1024)
    dense = m._LoRAMergeBase._expand_patch_to_diff(patch)
    actual_devices = []
    original = m._sample_sparsity
    def check(sample):
        actual_devices.append(sample.device.type)
        return original(sample)
    with mock.patch.object(m, "_sample_sparsity", side_effect=check):
        factor_score = score(patch, key, score_device=torch.device("cuda"))
        dense_score = score(("diff", (dense,)), key, score_device=torch.device("cuda"))
    assert actual_devices == ["cuda", "cuda"]
    assert factor_score == pytest.approx(dense_score, rel=2e-6, abs=1e-7)
    assert dense_score["sparsity_mean"] == score(("diff", (dense,)), key)["sparsity_mean"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_scoring_scratch_is_bounded_by_sample_not_full_matrix():
    # 64 MiB diff; old abs/mask staging exceeded a full matrix of scratch.
    tensor = torch.ones(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stats = m._diff_score_stats(tensor, False, "bounded.weight")
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    assert stats[1] == 0.
    assert extra < tensor.numel() * tensor.element_size() // 2
    del tensor
    gc.collect()
