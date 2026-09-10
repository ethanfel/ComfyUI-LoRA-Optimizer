"""QKV reassembly must preserve deltas without unbudgeted CUDA materialization."""
import weakref
from unittest import mock

import pytest
import torch

from tests.test_lora_optimizer import lora_optimizer as m

B = m._LoRAMergeBase
TARGET = "diffusion_model.blocks.0.attn.qkv_proj.weight"


def diff(tensor):
    return ("diff", (tensor,))


def adapter(seed=1, rows=4, cols=5, rank=2, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return m.LoRAAdapter(set(), (
        torch.randn(rows, rank, generator=g).to(device),
        torch.randn(rank, cols, generator=g).to(device),
        float(rank) / 2, None, None, None))


def slices(parts, target=TARGET):
    return {(target, (0, i * 4, 4)): part for i, part in enumerate(parts)}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dense_preallocated_copy_is_exact_and_does_not_use_cat(dtype):
    parts = [diff(torch.arange(20).reshape(4, 5).to(dtype) + i) for i in range(3)]
    expected = torch.cat([p[1][0] for p in parts])
    with mock.patch.object(torch, "cat", side_effect=AssertionError("dense cat recreates the crash path")):
        out = B._refuse_fused_qkv_patches(slices(parts))
    assert torch.equal(out[TARGET][1][0], expected)
    assert out[TARGET][1][0].dtype == dtype


def test_dense_mixed_dtypes_and_noncontiguous_views_match_cat():
    parts = [diff(torch.arange(20).reshape(5, 4).t().to(dtype))
             for dtype in (torch.float16, torch.bfloat16, torch.float64)]
    expected = torch.cat([p[1][0] for p in parts])
    output, _ = B._qkv_refusion_bytes(parts)
    assert output == expected.numel() * expected.element_size()
    actual = B._fuse_qkv_component_patches(parts)[1][0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["dense", "factors", "mixed"])
@pytest.mark.parametrize("native", [False, True])
def test_component_and_native_contributions_survive_without_mutating_inputs(kind, native):
    parts = [adapter(i + 1) for i in range(3)]
    if kind == "dense":
        parts = [diff(B._expand_patch_to_diff(p).half()) for p in parts]
    elif kind == "mixed":
        parts[1] = diff(B._expand_patch_to_diff(parts[1]))
    patches = slices(parts)
    if native:
        patches[TARGET] = diff(torch.full((12, 5), .125, dtype=torch.float16))
    before = {k: B._expand_patch_to_diff(p).clone() for k, p in patches.items()}
    expected = torch.cat([before[k] for k in slices(parts)])
    if native:
        expected += before[TARGET]
    out = B._refuse_fused_qkv_patches(patches, gpu_budget_bytes=0)
    torch.testing.assert_close(B._expand_patch_to_diff(out[TARGET]), expected)
    assert set(patches) == set(before)
    for k, p in patches.items():
        torch.testing.assert_close(B._expand_patch_to_diff(p), before[k], rtol=0, atol=0)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("factored", [False, True])
def test_partial_h3_padding_preserves_missing_slices(index, factored):
    part = adapter() if factored else diff(torch.ones(4, 5, dtype=torch.float16))
    patches = {(TARGET, (0, index * 4, 4)): part}
    out = B._refuse_fused_qkv_patches(patches, gpu_budget_bytes=0)
    expected = torch.zeros(12, 5)
    expected[index * 4:(index + 1) * 4] = B._expand_patch_to_diff(part)
    torch.testing.assert_close(B._expand_patch_to_diff(out[TARGET]), expected)


def test_named_zimage_and_native_collision_accumulate_instead_of_overwrite():
    target = "diffusion_model.layers.0.attention.qkv.weight"
    patches = {target.replace("qkv", "to_" + c): diff(torch.ones(4, 5) * i)
               for i, c in enumerate("qkv", 1)}
    patches[target] = diff(torch.ones(12, 5) * 7)
    expected = torch.cat([torch.ones(4, 5) * (7 + i) for i in range(1, 4)])
    result = B._refuse_fused_qkv_patches(patches)
    assert set(result) == {target}
    torch.testing.assert_close(result[target][1][0], expected)


def test_consumption_removes_only_replaced_stats_and_releases_originals():
    patches = slices([diff(torch.ones(4, 5)) for _ in range(3)])
    unrelated = torch.ones(2, 2)
    patches["unrelated"] = diff(unrelated)
    refs = [weakref.ref(p[1][0]) for k, p in patches.items() if k != "unrelated"]
    collector = {"stats": {id(p[1][0]): (p[1][0], (1., 0., None)) for p in patches.values()}}
    out = B._refuse_fused_qkv_patches(patches, _consume=True, _score_collector=collector)
    assert out is patches
    assert set(out) == {TARGET, "unrelated"}
    assert set(collector["stats"]) == {id(unrelated)}
    assert all(ref() is None for ref in refs)


def fake_cuda_diff(device="cuda:0", elements=100):
    # Metadata-only planning tests run in CPU CI, without allocating CUDA.
    t = mock.Mock(spec=torch.Tensor)
    t.device = torch.device(device)
    t.is_cuda = t.device.type == "cuda"
    t.numel.return_value = elements
    t.element_size.return_value = 2
    return diff(t)


def test_device_plan_checks_cast_workspace_and_current_budget():
    parts = [fake_cuda_diff() for _ in range(3)]
    output, workspace = B._qkv_refusion_bytes(parts)
    assert output == 3 * 100 * 4  # include the possible FP32 result
    assert workspace >= 2 * output
    mm = m.comfy.model_management
    with mock.patch.object(mm, "get_free_memory", return_value=workspace + 512 * 1024**2):
        assert B._qkv_refusion_device(parts, gpu_allowance_bytes=output).type == "cuda"
        assert B._qkv_refusion_device(parts, gpu_allowance_bytes=output - 1).type == "cpu"
    with mock.patch.object(mm, "get_free_memory", return_value=workspace + 512 * 1024**2 - 1):
        assert B._qkv_refusion_device(parts).type == "cpu"
    with mock.patch.object(mm, "get_free_memory", side_effect=RuntimeError("no telemetry")):
        assert B._qkv_refusion_device(parts).type == "cpu"


@pytest.mark.parametrize("other_device", ["cpu", "cuda:1"])
def test_mixed_devices_never_pull_components_to_the_first_gpu(other_device):
    parts = [fake_cuda_diff(), fake_cuda_diff(other_device), fake_cuda_diff()]
    with mock.patch.object(m.comfy.model_management, "get_free_memory") as free:
        assert B._qkv_refusion_device(parts).type == "cpu"
    free.assert_not_called()


def test_factor_size_bound_includes_rank_sum_padding_and_native_expansion():
    parts = [adapter(rows=64, cols=64, rank=2) for _ in range(3)]
    output, workspace = B._qkv_refusion_bytes(parts)
    assert output == (192 * 6 + 6 * 64) * 4
    padded, _ = B._qkv_refusion_bytes(parts[:1], pad_missing=True)
    assert padded == output
    native = adapter(rows=192, cols=64, rank=2)
    dense_output, dense_workspace = B._qkv_refusion_bytes(parts, native)
    assert dense_output == 192 * 64 * 4
    assert dense_workspace > workspace


def test_cuda_oom_retries_on_cpu_before_consuming_inputs():
    parts = [diff(torch.ones(4, 5)) for _ in range(3)]
    patches = slices(parts)
    patches[TARGET] = diff(torch.ones(12, 5))
    original = B._fuse_qkv_component_patches
    attempts = []
    def fuse(components, device=None):
        attempts.append(device.type)
        assert len(patches) == 4  # failed build has not consumed any inputs
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("simulated allocation race")
        return original(components, device=device)
    with mock.patch.object(B, "_qkv_refusion_device", return_value=torch.device("cuda")), \
         mock.patch.object(B, "_fuse_qkv_component_patches", side_effect=fuse):
        result = B._refuse_fused_qkv_patches(patches, _consume=True)
    assert attempts == ["cuda", "cpu"]
    assert set(result) == {TARGET}
    torch.testing.assert_close(result[TARGET][1][0], torch.full((12, 5), 2.))


def test_non_oom_errors_are_not_silently_retried():
    patches = slices([diff(torch.ones(4, 5)) for _ in range(3)])
    with mock.patch.object(B, "_fuse_qkv_component_patches", side_effect=RuntimeError("invalid kernel")) as fuse:
        with pytest.raises(RuntimeError, match="invalid kernel"):
            B._refuse_fused_qkv_patches(patches, _consume=True)
    assert fuse.call_count == 1
    assert len(patches) == 3


def test_native_shape_mismatch_rejects_before_consuming_inputs():
    patches = slices([diff(torch.ones(4, 5)) for _ in range(3)])
    patches[TARGET] = diff(torch.ones(1, 5))  # must not silently broadcast
    with pytest.raises(ValueError, match="Incompatible native/sliced QKV shapes"):
        B._refuse_fused_qkv_patches(patches, _consume=True)
    assert len(patches) == 4


@pytest.mark.parametrize("kind", ["loha", "lokr"])
def test_exotic_components_expand_safely_and_preserve_values(kind):
    if kind == "loha":
        part = m.LoHaAdapter(set(), (torch.ones(4, 2), torch.ones(2, 5), 1.,
                                     torch.ones(4, 2), torch.ones(2, 5), None, None))
    else:
        part = m.LoKrAdapter(set(), (torch.ones(2, 1), torch.ones(2, 5), None,
                                     None, None, None, None, None))
    assert B._qkv_refusion_bytes([part] * 3) is None
    expected = torch.cat([B._expand_patch_to_diff(part)] * 3)
    actual = B._refuse_fused_qkv_patches(slices([part] * 3))
    torch.testing.assert_close(actual[TARGET][1][0], expected)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("kind", ["dense", "factors", "mixed"])
@pytest.mark.parametrize("mixed_device", [False, True])
def test_real_cuda_native_collision_offloads_before_fp32_expansion(kind, mixed_device):
    parts = [adapter(i + 1, device="cuda") for i in range(3)]
    if kind == "dense":
        parts = [diff(B._expand_patch_to_diff(p).half()) for p in parts]
    elif kind == "mixed":
        parts[1] = diff(B._expand_patch_to_diff(parts[1]).half())
    if mixed_device:
        parts[2] = B._move_patch_to_device(parts[2], torch.device("cpu"))
    patches = slices(parts)
    patches[TARGET] = diff(torch.ones(12, 5, device="cuda", dtype=torch.float16))
    expected = torch.cat([B._expand_patch_to_diff(p).cpu() for p in parts]) + 1
    original = B._expand_patch_to_diff
    def cpu_expand(p):
        assert all(t.device.type == "cpu" for t in B._patch_tensors(p))
        return original(p)
    with mock.patch.object(B, "_expand_patch_to_diff", side_effect=cpu_expand):
        out = B._refuse_fused_qkv_patches(patches, gpu_budget_bytes=0)
    assert out[TARGET][1][0].device.type == "cpu"
    torch.testing.assert_close(out[TARGET][1][0], expected)
    devices = []
    score_stats = m._diff_score_stats
    def gpu_stats(t, compute_svd):
        devices.append(t.device.type)
        return score_stats(t, compute_svd)
    with mock.patch.object(m, "_diff_score_stats", side_effect=gpu_stats):
        score = m._score_merge_result(out, {}, compute_svd=False, score_device=torch.device("cuda"))
    assert devices == ["cuda"]
    assert score == m._score_merge_result(out, {}, compute_svd=False, score_device=torch.device("cuda"))


@cuda
def test_real_cuda_budget_pressure_and_per_group_release():
    patches, refs, expected = {}, [], {}
    for group in range(4):
        target = TARGET.replace("blocks.0", f"blocks.{group}")
        parts = [diff(torch.full((4, 5), float(group + i), device="cuda", dtype=torch.float16))
                 for i in range(3)]
        patches.update(slices(parts, target))
        expected[target] = torch.cat([p[1][0].cpu() for p in parts])
        refs.append([weakref.ref(p[1][0]) for p in parts])
    del parts
    collector = {"stats": {id(p[1][0]): (p[1][0], (1., 0., None)) for p in patches.values()},
                 "compute_svd": False}
    original = B._fuse_qkv_component_patches
    calls = []
    def fuse(parts, device=None):
        if calls:
            assert all(ref() is None for ref in refs[len(calls) - 1])
        calls.append(device.type)
        return original(parts, device=device)
    # First group has no remaining storage allowance. Later groups can fuse
    # and score on CUDA once its inputs have been released.
    budget = B._cuda_patch_bytes(patches)
    with mock.patch.object(m.comfy.model_management, "get_free_memory", return_value=8 * 1024**3), \
         mock.patch.object(B, "_fuse_qkv_component_patches", new=staticmethod(fuse)):
        result = B._refuse_fused_qkv_patches(patches, gpu_budget_bytes=budget,
                                           _consume=True, _score_collector=collector)
    assert calls[0] == "cpu" and "cuda" in calls[1:]
    assert all(ref() is None for group in refs for ref in group)
    for target, p in result.items():
        assert not p[1][0].is_cuda
        torch.testing.assert_close(p[1][0], expected[target])
    assert all(not t.is_cuda for t, _ in collector["stats"].values())


@cuda
def test_real_cuda_partial_padding_uses_cpu_under_pressure():
    p = diff(torch.ones(4, 5, device="cuda", dtype=torch.float16))
    original = B._expand_patch_to_diff
    def expand(patch):
        assert all(not t.is_cuda for t in B._patch_tensors(patch))
        return original(patch)
    with mock.patch.object(B, "_expand_patch_to_diff", side_effect=expand):
        out = B._refuse_fused_qkv_patches({(TARGET, (0, 4, 4)): p}, gpu_budget_bytes=0)
    torch.testing.assert_close(out[TARGET][1][0].float(),
                               torch.cat([torch.zeros(4, 5), torch.ones(4, 5), torch.zeros(4, 5)]))
