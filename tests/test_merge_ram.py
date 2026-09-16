"""Resident output and candidate RAM regressions (no full H3 weights needed)."""
import gc
import weakref
from unittest import mock

import pytest
import torch

from tests.test_lora_optimizer import lora_optimizer as m
from tests.test_phase1_correctness import ToyPatcher, entry


@pytest.fixture
def case(tmp_path, monkeypatch):
    model = ToyPatcher()
    model.model.layer = torch.nn.Linear(64, 96, bias=False)
    g = torch.Generator().manual_seed(29)
    stack = [dict(entry({"layer.lora_A.weight": torch.randn(4, 64, generator=g),
                         "layer.lora_B.weight": torch.randn(96, 4, generator=g)},
                        name=f"lora{i}"), strength=.7 + i * .2) for i in range(2)]
    monkeypatch.setattr(m._LoRAMergeBase, "_get_compute_device", lambda self: torch.device("cpu"))
    monkeypatch.setattr(m.comfy.lora, "model_lora_keys_unet", lambda *a: {"layer": "layer.weight"})
    monkeypatch.setattr(m, "AUTOTUNER_MEMORY_DIR", str(tmp_path))
    return model, stack


def merge(opt, case, **kwargs):
    options = dict(optimization_mode="global", merge_strategy_override="slerp",
                   cache_patches="disabled", patch_compression="disabled")
    options.update(kwargs)
    return opt.optimize_merge(*case, 1., **options)


@pytest.mark.parametrize("weights", [(1., 1.), (.2, .9), (-.7, .8), (-.9, -.3), (0., .8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_compact_pairwise_slerp_matches_dense_without_svd(case, weights, dtype):
    model, stack = case
    for item, weight in zip(stack, weights):
        item["strength"] = weight
        item["lora"] = {k: v.to(dtype) for k, v in item["lora"].items()}
    opt = m.LoRAOptimizer()
    dense = merge(opt, case, _compact_output=False)[4]["model_patches"]["layer.weight"]
    with mock.patch.object(opt, "_compress_to_lowrank", side_effect=AssertionError("no truncated SVD")):
        compact = merge(opt, case)[4]["model_patches"]["layer.weight"]
    assert isinstance(compact, m.LoRAAdapter)
    actual = opt._expand_patch_to_diff(compact)
    expected = opt._expand_patch_to_diff(dense)
    # Dense output had one native storage cast; compact factors retain FP32.
    torch.testing.assert_close(actual.to(dtype), expected.to(dtype), rtol=.01, atol=.01)
    assert torch.linalg.vector_norm(actual - expected) / expected.norm() < (.004 if dtype == torch.bfloat16 else .0005)
    assert opt._estimate_single_patch_bytes(compact) < expected.numel() * expected.element_size()


@pytest.mark.parametrize("kwargs", [dict(merge_refinement="balanced"), dict(sparsification="dare"),
                                    dict(star_eta=90.), dict(preserve=True), dict(conflict_mode="shared")])
def test_modified_inputs_do_not_take_unmodified_slerp_factor_path(case, kwargs):
    if "preserve" in kwargs or "conflict_mode" in kwargs:
        case[1][0].update(kwargs)
        kwargs = {}
    opt = m.LoRAOptimizer()
    with mock.patch.object(opt, "_compact_slerp_patch", wraps=opt._compact_slerp_patch) as compact:
        merge(opt, case, **kwargs)
    compact.assert_not_called()


def test_compact_rejects_wrong_unstable_or_larger_factors():
    patch = m.LoRAAdapter(set(), (torch.ones(64, 2), torch.ones(2, 64), 2., None, None, None))
    assert m.LoRAOptimizer._compact_slerp_patch(patch, torch.zeros(64, 64), torch.float32) is None
    assert m.LoRAOptimizer._compact_slerp_patch(patch, torch.full((64, 64), float("nan")), torch.float32) is None
    large = m.LoRAAdapter(set(), (torch.ones(4, 4), torch.ones(4, 4), 4., None, None, None))
    assert m.LoRAOptimizer._compact_slerp_patch(large, torch.full((4, 4), 4.), torch.float32) is None


@pytest.mark.parametrize("mode", ["slerp", "ties", "consensus", "weighted_sum", "weighted_average"])
@pytest.mark.parametrize("compute_svd", [False, True])
def test_streamed_candidate_score_matches_dense_candidate(case, mode, compute_svd):
    opt = m.LoRAOptimizer()
    options = dict(merge_strategy_override=mode, _compact_output=False)
    torch.manual_seed(47)  # consensus uses randomized low-rank spectral cleanup
    dense = merge(opt, case, **options)[4]
    torch.manual_seed(47)
    streamed = merge(opt, case, **options, _skip_model_apply=True, _skip_report=True,
                     _score_only={"device": None, "compute_svd": compute_svd})[4]
    def score(data):
        return m._score_merge_result(data["model_patches"], data["clip_patches"],
                                     compute_svd=compute_svd, lora_svd=True)
    assert score(streamed) == pytest.approx(score(dense), rel=1e-7, abs=1e-7)
    if mode in ("slerp", "ties", "consensus"):
        assert isinstance(streamed["model_patches"]["layer.weight"], m._ScoredDiff)
        assert opt._estimate_single_patch_bytes(streamed["model_patches"]["layer.weight"]) == 0


def test_score_only_cannot_be_applied_or_cached(case):
    with pytest.raises(ValueError, match="private"):
        merge(m.LoRAOptimizer(), case, _score_only={"device": None})


@pytest.mark.parametrize("native", [False, True])
def test_qkv_stream_scores_final_fusion_and_releases_components(native):
    target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    g = torch.Generator().manual_seed(37)
    patches = {(target, (0, i * 64, 64)): ("diff", (torch.randn(64, 64, generator=g),)) for i in range(3)}
    if native:
        patches[target] = ("diff", (torch.randn(192, 64, generator=g),))
    refs = [weakref.ref(p[1][0]) for p in patches.values()]
    expected = m._score_merge_result(m._LoRAMergeBase._refuse_fused_qkv_patches(patches), {}, compute_svd=True)
    streamed = m._LoRAMergeBase._refuse_fused_qkv_patches(
        patches, _consume=True, _score_only={"device": None, "compute_svd": True})
    assert all(ref() is None for ref in refs)
    assert isinstance(streamed[target], m._ScoredDiff)
    assert m._score_merge_result(streamed, {}, compute_svd=True) == pytest.approx(expected)


@pytest.mark.parametrize("cache_setting", ["enabled", "disabled"])
def test_optimizer_miss_releases_previous_output_before_analysis(case, cache_setting):
    opt = m.LoRAOptimizer()
    first = merge(opt, case, cache_patches="enabled")
    refs = [weakref.ref(t) for t in opt._patch_tensors(first[4]["model_patches"]["layer.weight"])]
    del first
    assert all(ref() is not None for ref in refs)
    original = opt._get_compute_device
    def check():
        assert all(ref() is None for ref in refs)
        return original()
    with mock.patch.object(opt, "_get_compute_device", side_effect=check):
        merge(opt, case, merge_strategy_override="weighted_sum", cache_patches=cache_setting)


@pytest.mark.parametrize("change", ["strength", "selection", "disable_cache", "tuning_only"])
def test_autotuner_releases_previous_output_before_new_merge(case, change):
    tuner = m.LoRAAutoTuner()
    options = dict(top_n=5, scoring_svd="full", scoring_device="cpu", cache_patches="enabled")
    first = tuner.auto_tune(*case, 1., **options)
    refs = [weakref.ref(t) for p in first[5]["model_patches"].values() for t in tuner._patch_tensors(p)]
    del first
    gc.collect()
    assert all(ref() is not None for ref in refs)
    if change == "selection":
        options["selection"] = 2
    elif change == "disable_cache":
        options["cache_patches"] = "disabled"
    elif change == "tuning_only":
        options["output_mode"] = "tuning_only"
    original = m.LoRAOptimizer.optimize_merge
    seen = []
    def checked(*args, **kwargs):
        seen.append(True)
        assert all(ref() is None for ref in refs)
        return original(*args, **kwargs)
    with mock.patch.object(m.LoRAOptimizer, "optimize_merge", new=checked):
        tuner.auto_tune(*case, .9 if change == "strength" else 1., **options)
    assert all(ref() is None for ref in refs)
    assert seen or change == "tuning_only"


def test_autotuner_valid_hit_does_not_merge_again(case):
    tuner = m.LoRAAutoTuner()
    options = dict(top_n=1, scoring_device="cpu")
    first = tuner.auto_tune(*case, 1., **options)
    with mock.patch.object(m.LoRAOptimizer, "optimize_merge", side_effect=AssertionError("cache hit")):
        assert tuner.auto_tune(*case, 1., **options) is first


def test_diff_cache_declines_over_budget_before_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(m.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    cache = m._DiffCache(mode="auto")
    cache._ram_limit = 1
    tensor = mock.Mock(spec=torch.Tensor)
    tensor.nelement.return_value = 4096
    tensor.element_size.return_value = 4
    cache.put(("key", 0), tensor)
    tensor.detach.assert_not_called()
    assert not cache._ram_store
    cache.clear()


def test_final_qkv_native_plus_sliced_factors_stay_compact():
    from tests.test_qkv_memory import adapter
    target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    patches = {(target, (0, i * 64, 64)): adapter(i + 1, rows=64, cols=64, rank=2) for i in range(3)}
    patches[target] = adapter(5, rows=192, cols=64, rank=2)
    expected = m._LoRAMergeBase._refuse_fused_qkv_patches(patches)[target]
    with mock.patch.object(m._LoRAMergeBase, "_expand_patch_to_diff", side_effect=AssertionError("no dense allocation")):
        compact = m._LoRAMergeBase._refuse_fused_qkv_patches(patches, _compact_output=True)[target]
    assert isinstance(compact, m.LoRAAdapter)
    torch.testing.assert_close(m._LoRAMergeBase._expand_patch_to_diff(compact), expected[1][0], atol=3e-6, rtol=3e-6)
    assert m._LoRAMergeBase._estimate_single_patch_bytes(compact) < m._LoRAMergeBase._estimate_single_patch_bytes(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("merge_device", ["cpu", "cuda"])
def test_streamed_qkv_keeps_requested_gpu_scoring(merge_device):
    target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    patches = {(target, (0, i * 64, 64)): ("diff", (torch.ones(64, 64, device=merge_device) * (i + 1),))
               for i in range(3)}
    calls = []
    original = m._diff_score_stats
    def check(tensor, compute_svd, target_key=None):
        calls.append((tensor.device.type, compute_svd, tuple(tensor.shape)))
        return original(tensor, compute_svd, target_key)
    with mock.patch.object(m, "_diff_score_stats", new=check):
        result = m._LoRAMergeBase._refuse_fused_qkv_patches(
            patches, _consume=True, _score_only={"device": torch.device("cuda"), "compute_svd": True})
    assert calls == [("cuda", True, (192, 64))]
    assert isinstance(result[target], m._ScoredDiff)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_slerp_compaction_and_streaming(case, monkeypatch):
    monkeypatch.setattr(m._LoRAMergeBase, "_get_compute_device", lambda self: torch.device("cuda"))
    opt = m.LoRAOptimizer()
    dense = merge(opt, case, _compact_output=False)[4]["model_patches"]
    compact = merge(opt, case)[4]["model_patches"]
    assert isinstance(compact["layer.weight"], m.LoRAAdapter)
    torch.testing.assert_close(opt._expand_patch_to_diff(compact["layer.weight"]),
                               opt._expand_patch_to_diff(dense["layer.weight"]), rtol=2e-5, atol=2e-5)
    config = {"device": torch.device("cuda"), "compute_svd": True}
    streamed = merge(opt, case, _compact_output=False, _skip_model_apply=True, _skip_report=True,
                     _score_only=config)[4]["model_patches"]
    expected = m._score_merge_result(dense, {}, compute_svd=True, score_device=config["device"])
    assert m._score_merge_result(streamed, {}, compute_svd=True) == pytest.approx(expected, rel=1e-6)


def test_h3_candidate_streams_complete_native_and_sliced_groups(case, monkeypatch):
    model, _ = case
    model.model.diffusion_model = torch.nn.Module()
    blocks = model.model.diffusion_model.blocks = torch.nn.ModuleList()
    mapping, payloads = {}, [{}, {}]
    gen = torch.Generator().manual_seed(492)
    for block_id in range(3):
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.qkv_proj = torch.nn.Linear(64, 192, bias=False)
        blocks.append(block)
        prefix = f"diffusion_model.blocks.{block_id}.attn."
        target = prefix + "qkv_proj.weight"
        mapping[prefix + "qkv_proj"] = target
        for component, name in enumerate(("to_q", "to_k", "to_v")):
            mapping[prefix + name] = (target, (0, component * 64, 64))
        # Both adapters have native AND split contributions. Native-plus-sliced
        # addition must happen before measuring the final projection's sparsity.
        for payload in payloads:
            for name, rows in (("qkv_proj", 192), ("to_q", 64), ("to_k", 64), ("to_v", 64)):
                payload[prefix + name + ".lora_A.weight"] = torch.randn(2, 64, generator=gen)
                payload[prefix + name + ".lora_B.weight"] = torch.randn(rows, 2, generator=gen)
    monkeypatch.setattr(m.comfy.lora, "model_lora_keys_unet", lambda *a: dict(mapping))
    stack = [entry(p, name=f"h3-{i}", h3_layout="comfy") for i, p in enumerate(payloads)]
    opt = m.LoRAOptimizer()
    original_normalize = opt._normalize_stack
    def normalize(*a, **kw):
        result = original_normalize(*a, **kw)
        opt._detected_arch = "minimax_h3"
        return result
    monkeypatch.setattr(opt, "_normalize_stack", normalize)
    dense = merge(opt, (model, stack), _compact_output=False)[4]["model_patches"]
    assert len(dense) == 3
    seen = []
    original_refuse = opt._refuse_fused_qkv_patches
    def refuse(patches, **kwargs):
        if kwargs.get("_score_only") is not None:
            seen.append(len(patches))
        return original_refuse(patches, **kwargs)
    monkeypatch.setattr(opt, "_refuse_fused_qkv_patches", refuse)
    streamed = merge(opt, (model, stack), _compact_output=False, _skip_model_apply=True,
                     _skip_report=True, _score_only={"device": None, "compute_svd": True})[4]["model_patches"]
    assert seen[:3] == [4, 4, 4]  # three incremental assemblies, not one full patch set
    assert all(isinstance(p, m._ScoredDiff) for p in streamed.values())
    assert m._score_merge_result(streamed, {}, compute_svd=True) == pytest.approx(
        m._score_merge_result(dense, {}, compute_svd=True), rel=1e-7)
