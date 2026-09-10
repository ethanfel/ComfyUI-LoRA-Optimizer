"""Bounded synthetic CUDA regression; no ComfyUI server or model weights.

Run in separate processes:
  python tests/integration/qkv_memory_stress.py baseline
  python tests/integration/qkv_memory_stress.py fixed

The baseline is the three reassembly methods from release 1.8.6. A 768 MiB
per-process allocator limit reproduces accumulation without filling the GPU.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from test_lora_optimizer import lora_optimizer as m


def baseline_class():
    source = subprocess.check_output(
        ["git", "show", "27482bb:lora_optimizer.py"], text=True,
        cwd=Path(__file__).resolve().parents[2])
    tree = ast.parse(source)
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_LoRAMergeBase")
    names = {"_fuse_qkv_component_patches", "_refuse_fused_qkv_patches", "_expand_patch_to_diff"}
    methods = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == 3
    namespace = dict(vars(m))
    exec(compile(ast.Module(body=methods, type_ignores=[]), "<1.8.6-QKV>", "exec"), namespace)
    legacy = type("LegacyQKV", (m._LoRAMergeBase,), {name: namespace[name] for name in names})
    namespace["_LoRAMergeBase"] = legacy
    return legacy


def main(mode):
    assert mode in ("baseline", "fixed")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this bounded test")
    device = torch.device("cuda:0")
    cap = 768 * 1024**2
    total = torch.cuda.get_device_properties(device).total_memory
    free, _ = torch.cuda.mem_get_info(device)
    if free < 2 * cap:
        raise SystemExit("Not enough free VRAM for the bounded test; no allocations made")
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    base = baseline_class() if mode == "baseline" else m._LoRAMergeBase
    patches = {}
    for group in range(96):
        target = f"diffusion_model.blocks.{group}.attn.qkv_proj.weight"
        patches[target] = ("diff", (torch.full((2304, 512), 2., dtype=torch.float16, device=device),))
        for component in range(3):
            patches[(target, (0, component * 768, 768))] = (
                "diff", (torch.full((768, 512), float(component + 1), dtype=torch.float16, device=device),))
    collector = {"stats": {id(p[1][0]): (p[1][0], None) for p in patches.values()},
                 "compute_svd": False}
    initial = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    # Model-management telemetry includes free allocator blocks, but respects
    # this process's synthetic capacity rather than the machine's 32 GiB GPU.
    m.comfy.model_management.get_free_memory = lambda _: max(0, cap - torch.cuda.memory_allocated(device))
    gpu_scores = 0
    original_stats = m._diff_score_stats
    def stats(tensor, compute_svd):
        nonlocal gpu_scores
        assert tensor.is_cuda, "the fix must keep scoring on GPU"
        gpu_scores += 1
        return original_stats(tensor, compute_svd)
    m._diff_score_stats = stats
    start = time.perf_counter()
    oom = False
    try:
        if mode == "baseline":
            result = base._refuse_fused_qkv_patches(patches)
        else:
            result = base._refuse_fused_qkv_patches(
                patches, gpu_budget_bytes=initial, _consume=True, _score_collector=collector)
            assert len(result) == 96
            for patch in result.values():
                tensor = patch[1][0]
                assert not tensor.is_cuda
                for component in range(3):
                    assert torch.all(tensor[component * 768:(component + 1) * 768] == component + 3)
            m._score_merge_result(result, {}, compute_svd=False, score_device=device,
                                  _inline_stats=collector["stats"])
            assert gpu_scores == 96
    except torch.cuda.OutOfMemoryError:
        oom = True
    torch.cuda.synchronize(device)
    print(json.dumps(dict(mode=mode, allocator_cap_mib=cap // 1024**2,
                          initial_allocated_mib=initial / 1024**2,
                          peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
                          elapsed_seconds=round(time.perf_counter() - start, 3),
                          oom=oom, gpu_scored_groups=gpu_scores)))
    assert oom == (mode == "baseline"), "expected old code to OOM and fixed code to complete"


if __name__ == "__main__":
    main(sys.argv[1])
