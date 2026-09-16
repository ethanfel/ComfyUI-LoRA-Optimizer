"""Bounded five-candidate/full-SVD RAM comparison against release 1.8.7.

Run baseline/fixed in separate processes. Uses synthetic factors and meta base
weights, no server, downloads, generation, or user models. CUDA cap is 2 GiB.
  python tests/integration/merge_ram_stress.py baseline --device cuda
  python tests/integration/merge_ram_stress.py fixed --device cuda
"""
import argparse
import ast
import gc
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_lora_optimizer import lora_optimizer as m
import psutil
import torch


def run(args):
    logging.getLogger().setLevel(logging.WARNING)
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; no benchmark run")
        cap = 2 * 1024**3
        free, total = torch.cuda.mem_get_info()
        if free < 2 * cap:
            raise SystemExit("Need 4 GiB free VRAM; no benchmark allocations made")
        torch.cuda.set_per_process_memory_fraction(cap / total)
    ns = vars(m) if args.mode == "fixed" else dict(vars(m))
    if args.mode == "baseline":
        source = subprocess.check_output(["git", "show", "a844802:lora_optimizer.py"], text=True)
        classes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                   and n.name in ("_LoRAMergeBase", "LoRAOptimizer", "LoRAAutoTuner")]
        assert len(classes) == 3
        exec(compile(ast.Module(body=classes, type_ignores=[]), "<1.8.7>", "exec"), ns)
    base, optimizer, autotuner = (ns[n] for n in ("_LoRAMergeBase", "LoRAOptimizer", "LoRAAutoTuner"))
    base._get_compute_device = lambda self: device
    keys = {f"layer{i}": f"layer{i}.weight" for i in range(args.groups)}
    base._get_model_keys = lambda self, model: keys

    class Patcher:
        def __init__(self):
            self.model = types.SimpleNamespace(**{
                f"layer{i}": types.SimpleNamespace(weight=torch.empty(args.size, args.size, device="meta"))
                for i in range(args.groups)})
            self.patches = {}

        def clone(self):
            clone = Patcher()
            clone.model = self.model
            return clone

        def add_patches(self, patches, *a):
            self.patches.update(patches)
            return list(patches)

    gen = torch.Generator().manual_seed(938)
    stack = [dict(name=f"ram-{i}", strength=.8, lora={
        key: value for j in range(args.groups) for key, value in (
            (f"layer{j}.lora_A.weight", torch.randn(8, args.size, generator=gen) * .05),
            (f"layer{j}.lora_B.weight", torch.randn(args.size, 8, generator=gen) * .05))}) for i in range(2)]
    grid = [dict(optimization_mode="global", merge_mode=mode, merge_refinement="none",
                 sparsification="disabled", sparsification_density=.7, dare_dampening=0.,
                 auto_strength="disabled", strategy_set="full")
            for mode in ("weighted_sum", "weighted_average", "slerp", "ties", "consensus")]
    # Full SVD must actually run; random independent factors are otherwise
    # classified as orthogonal and deliberately skip merge-quality SVD.
    for key in stack[1]["lora"]:
        stack[1]["lora"][key] = stack[0]["lora"][key] * .9 + stack[1]["lora"][key] * .1
    ns["_generate_param_grid"] = lambda **kw: grid
    for name in ("_lora_cache_load", "_pair_cache_load", "_memory_load"):
        setattr(autotuner, name, staticmethod(lambda *a, **kw: None))
    for name in ("_lora_cache_save", "_pair_cache_save", "_memory_save"):
        setattr(autotuner, name, staticmethod(lambda *a, **kw: None))
    candidate_mib = []
    original_merge = optimizer.optimize_merge
    def tracked(self, *a, **kw):
        result = original_merge(self, *a, **kw)
        if "_score_collector" in kw:
            candidate_mib.append(sum(self._estimate_single_patch_bytes(p)
                                     for k in ("model_patches", "clip_patches")
                                     for p in result[4][k].values()) / 1024**2)
        return result
    optimizer.optimize_merge = tracked
    proc = psutil.Process()
    initial = proc.memory_info().rss
    peak = [initial]
    done = threading.Event()
    def sample():
        while not done.wait(.01):
            peak[0] = max(peak[0], proc.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="merge-ram-stress-") as directory:
        ns["AUTOTUNER_MEMORY_DIR"] = directory
        model = Patcher()
        tuner = autotuner()
        result = tuner.auto_tune(model, stack, 1., top_n=5, scoring_svd="full",
                                 scoring_device="gpu" if device.type == "cuda" else "cpu",
                                 cache_patches="disabled", architecture_preset="minimax_h3")
        scores = {r["config"]["merge_mode"]: r["score_measured"] for r in result[4]["top_n"]}
        del result, tuner
        gc.collect()
        # Separate persistent footprint of an eligible SLERP output, regardless
        # of which of the five candidates the statistical score selected.
        output = optimizer().optimize_merge(model, stack, 1., optimization_mode="global",
                    merge_strategy_override="slerp", patch_compression="disabled", cache_patches="disabled")
        resident = sum(base._estimate_single_patch_bytes(p) for p in output[4]["model_patches"].values())
    elapsed = time.perf_counter() - start
    done.set()
    sampler.join()
    print(json.dumps(dict(mode=args.mode, device=str(device), groups=args.groups, size=args.size,
        initial_rss_mib=round(initial / 1024**2, 1), peak_rss_mib=round(peak[0] / 1024**2, 1),
        candidate_patch_mib=candidate_mib, slerp_output_mib=resident / 1024**2,
        measured_scores=scores, elapsed_seconds=round(elapsed, 2)), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "fixed"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--size", type=int, default=1536)
    args = parser.parse_args()
    if not (1 <= args.groups <= 32 and 64 <= args.size <= 2048):
        parser.error("bounded test: groups 1–32, size 64–2048")
    run(args)
