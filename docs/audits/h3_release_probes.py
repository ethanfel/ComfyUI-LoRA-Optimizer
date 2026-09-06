"""Read-only audit probes for ComfyUI-LoRA-Optimizer HEAD a5e80a9."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
import types
from unittest import mock
import torch

ROOT = str(Path(__file__).resolve().parents[2])
COMFY_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/media/p5/Comfyui")
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("audit_test_helpers", ROOT + "/tests/test_lora_optimizer.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
m = helpers.lora_optimizer
b = m._LoRAMergeBase
opt = m.LoRAOptimizer()
results = {}

# Execute the actual generic-key loop from the installed ComfyUI mapper,
# avoiding its unrelated architecture imports and GPU initialization.
with open(COMFY_ROOT / "comfy" / "lora.py") as f:
    tree = ast.parse(f.read())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "model_lora_keys_unet")
fn.body = fn.body[:3] + [ast.Return(value=ast.Name(id="key_map", ctx=ast.Load()))]
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "comfy_generic_mapper", "exec"), namespace)
target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
sd = {target: torch.zeros(768, 2)}
mapping = namespace["model_lora_keys_unet"](types.SimpleNamespace(state_dict=lambda: sd), {})
b._add_minimax_h3_qkv_aliases(mapping, sd)
native = {
    "blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(1, 2),
    "blocks.0.attn.qkv_proj.lora_B.weight": torch.arange(768.).reshape(768, 1),
}
normal = b._normalize_keys_minimax_h3(native)
prefixes = b._collect_lora_prefixes([{"lora": normal}])
groups = opt._build_target_groups(prefixes, mapping, {})
results["real_comfy_target_mapping"] = {"normalized_attention_prefixes": len(prefixes), "resolved_groups": len(groups), "split_aliases_added": sum("to_q" in k or "to_k" in k or "to_v" in k for k in mapping)}

# Same contiguous Comfy QKV weights; PEFT's default adapter-name is not a layout.
peft = {k.replace(".weight", ".default.weight"): v for k, v in native.items()}
reordered = b._normalize_keys_minimax_h3(peft)
up_key = "diffusion_model.blocks.0.attn.to_q.lora_B.weight"
results["peft_layout_discriminator"] = {"expected_q_rows_128_131": normal[up_key][128:132, 0].tolist(), "actual_q_rows_128_131": reordered[up_key][128:132, 0].tolist(), "same_delta": torch.equal(normal[up_key], reordered[up_key])}

patches = {(target, (0, i * 256, 256)): m.LoRAAdapter([], (torch.full((256, 1), float(i + 1)), torch.ones(1, 2), 1., None, None, None)) for i in range(3)}
refused = b._refuse_fused_qkv_patches(patches)
results["qkv_refusion_weight_suffix"] = {"input_slices": len(patches), "output_patches": len(refused), "fused_target_present": target in refused}

# Dense rebased bias diffs are standard Comfy patches but not collected by files.
results["dense_bias_collection"] = b._collect_lora_prefixes([{"lora": {"diffusion_model.blocks.0.adaln_proj.linear.diff": torch.ones(2, 8), "diffusion_model.blocks.0.adaln_proj.linear.diff_b": torch.ones(2)}}])

# Non-H3 regression: DoRA factors are accepted as plain LoRA in file mode.
dora = {"layer.lora_up.weight": torch.ones(2, 1), "layer.lora_down.weight": torch.ones(1, 2), "layer.dora_scale": torch.zeros(2)}
up, down, alpha, mid = opt._get_lora_key_info(dora, "layer")
results["file_dora_ignored"] = {"returned_plain_delta": opt._compute_lora_diff(up, down, alpha, mid, (2, 2)).tolist(), "base_weight": torch.eye(2).tolist(), "dora_zero_magnitude_expected_delta": (-torch.eye(2)).tolist()}

# Export LoCon factors including a nontrivial mid tensor.
saved = {}
def capture(state, path, metadata=None):
    saved.update(state)
locon = m.LoRAAdapter([], (torch.ones(2, 1, 1, 1), torch.ones(1, 2, 1, 1), 1., torch.ones(1, 1, 3, 3), None, None))
data = {"model_patches": {"layer.weight": locon}, "clip_patches": {}, "key_map": {"layer.weight": "layer"}, "output_strength": 1., "clip_strength": 1., "sum_rank": 1}
with mock.patch.object(m, "save_file", side_effect=capture), mock.patch.object(m, "_resolve_safe_output_path", return_value="/tmp/h3-audit-no-file-written.safetensors"):
    m.SaveMergedLoRA().save_lora(data, "/tmp", "unused", save_rank=1)
results["locon_export"] = {"saved_keys": sorted(saved), "original_delta_shape": list(b._expand_patch_to_diff(locon).shape), "exported_factor_product_shape": list((saved["layer.lora_up.weight"].flatten(1) @ saved["layer.lora_down.weight"].flatten(1)).shape)}

# In-memory payload fingerprints should change when tensor contents change.
entry = helpers._make_lora_entry({"layer": 1.}, name="same")
key1 = opt._compute_cache_key([entry], 1., 1., "disabled")
entry["lora"]["layer.lora_up.weight"] = torch.tensor([[99.]])
key2 = opt._compute_cache_key([entry], 1., 1., "disabled")
results["in_memory_cache_fingerprint"] = {"before": key1, "after_tensor_replacement": key2, "equal": key1 == key2}

sdxl_ff = {
    "unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_A.weight": torch.ones(1, 2),
    "unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.lora_B.weight": torch.arange(8.).reshape(8, 1),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 2),
}
sdxl_arch = b._detect_architecture(sdxl_ff)
sdxl_norm = b._normalize_keys(sdxl_ff, sdxl_arch)
results["sdxl_misdetected_as_h3"] = {"detected": sdxl_arch, "normalized_unet_keys": [k for k in sdxl_norm if "down_blocks" in k]}

# Single-file shortcut sends all tensors straight to Comfy even when filtered.
single = helpers._make_lora_entry({"blocks.0.self_attn.q": 1., "blocks.0.ffn.0": 2.}, name="single", key_filter="attention_only")
with mock.patch.object(m.comfy.sd, "load_lora_for_models", return_value=(None, None)) as loader:
    out = m.LoRAOptimizer().optimize_merge(None, [single], 1.)
    forwarded = loader.call_args.args[2]
results["single_lora_filter_bypass"] = {"requested_filter": single["key_filter"], "mlp_keys_forwarded": [k for k in forwarded if "ffn" in k], "lora_data": out[4]}

# Confirm stale cache is actually replayed by optimize_merge, not only a hash.
cache_opt = m.LoRAOptimizer()
stack = [helpers._make_lora_entry({"layer": 1.}, name="one"), helpers._make_lora_entry({"layer": 2.}, name="two")]
cache_key = cache_opt._compute_cache_key(stack, 1., 1., "disabled") + f"|mid={id(None)}"
cache_opt._merge_cache[cache_key] = ({}, {}, "STALE_AUDIT_SENTINEL", 1., {"audit": "old payload"})
stack[0]["lora"]["layer.lora_up.weight"] = torch.tensor([[999.]])
cached_out = cache_opt.optimize_merge(None, stack, 1.)
results["stale_cache_replayed"] = cached_out[2] == "STALE_AUDIT_SENTINEL"

# A finite float32 rank-1 delta can overflow merely by exporting its factors.
saved.clear()
large_factor = m.LoRAAdapter([], (torch.full((2, 1), 100000.), torch.full((1, 2), 0.00001), 1., None, None, None))
data["model_patches"] = {"layer.weight": large_factor}
with mock.patch.object(m, "save_file", side_effect=capture), mock.patch.object(m, "_resolve_safe_output_path", return_value="/tmp/h3-audit-no-file-written.safetensors"):
    m.SaveMergedLoRA().save_lora(data, "/tmp", "unused", save_rank=1)
results["float32_export_overflow"] = {"finite_input_delta": bool(torch.isfinite(b._expand_patch_to_diff(large_factor)).all()), "saved_dtype": str(saved["layer.lora_up.weight"].dtype), "finite_saved_up": bool(torch.isfinite(saved["layer.lora_up.weight"]).all())}
extracted = m._extract_lora_svd(torch.eye(8), rank=1, rank_mode="auto", energy_threshold=0.99)
results["extract_auto_rank_cap"] = {"requested_max_rank": 1, "returned_rank": int(extracted[1].shape[0])}
print(json.dumps(results, indent=2))
