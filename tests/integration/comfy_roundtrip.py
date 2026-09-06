"""Run separately: python tests/integration/comfy_roundtrip.py /path/to/ComfyUI.

Real ComfyUI + safetensors, CPU only, no checkpoints and no unit-test stubs.
The miniature module tests loader semantics, not H3 generation quality.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types

root = Path(__file__).resolve().parents[2]
comfy_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(comfy_root))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.lora
import comfy.sd
from comfy.model_patcher import ModelPatcher
from safetensors.torch import load_file

spec = importlib.util.spec_from_file_location("lora_optimizer_integration", root / "lora_optimizer.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m._LoRAMergeBase._get_compute_device = staticmethod(lambda: torch.device("cpu"))


class TinyH3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = types.SimpleNamespace(unet_config={})
        self.diffusion_model = torch.nn.Module()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.qkv_proj = torch.nn.Linear(4, 768)
        block.mlp = torch.nn.Module()
        block.mlp.fc1 = torch.nn.Linear(4, 8)
        block.adaln_proj = torch.nn.Module()
        block.adaln_proj.linear = torch.nn.Linear(4, 6)
        self.diffusion_model.blocks = torch.nn.ModuleList([block])
        self.diffusion_model.norm = torch.nn.LayerNorm(4)
        self.diffusion_model.conv = torch.nn.Conv2d(2, 2, 3)


class TinyClip:
    """Only CLIP's container API is small; mapping/patching are real ComfyUI."""
    def __init__(self, patcher=None):
        if patcher is None:
            module = torch.nn.Module()
            parent = module
            for segment in 'clip_l.transformer.text_model.encoder.layers.0.self_attn'.split('.'):
                child = torch.nn.Module()
                parent.add_module(segment, child)
                parent = child
            parent.q_proj = torch.nn.Linear(4, 4)
            patcher = ModelPatcher(module, torch.device('cpu'), torch.device('cpu'))
        self.patcher = patcher
        self.cond_stage_model = patcher.model

    def clone(self):
        return TinyClip(self.patcher.clone())

    def add_patches(self, patches, strength):
        return self.patcher.add_patches(patches, strength)


torch.manual_seed(123)
base = ModelPatcher(TinyH3(), torch.device("cpu"), torch.device("cpu"))
prefix = "diffusion_model.blocks.0.attn"
q = torch.randn(256, 2) * .01
a = torch.randn(2, 4)
sd = {f"{prefix}.to_q.lora_A.weight": a,
      f"{prefix}.to_q.lora_B.weight": q,
      f"{prefix}.to_q.alpha": torch.tensor(3.),
      "diffusion_model.blocks.0.adaln_proj.linear.diff": torch.randn(6, 4) * .01,
      "diffusion_model.blocks.0.adaln_proj.linear.diff_b": torch.randn(6) * .01,
      "diffusion_model.norm.diff": torch.randn(4) * .01}
item = dict(name="tiny-h3", lora=sd, strength=-.7, h3_layout="comfy", _architecture="minimax_h3")
opt = m.LoRAOptimizer()
opt._save_report_to_disk = lambda *a, **kw: None
result = opt.optimize_merge(base, [item], 1.3, optimization_mode="additive",
                            patch_compression="disabled", normalize_keys="enabled")
data = result[4]
assert len(data["model_patches"]) == 4, data["coverage"]
hooks = m.MergedLoRAToHook().convert(data)[0]
assert len(hooks.hooks) == 1
assert hooks.hooks[0].weights is data['model_patches']
assert hooks.hooks[0].strength_model == data['output_strength']
target = f"{prefix}.qkv_proj.weight"
expected_qkv = torch.cat([q @ a * 1.5, torch.zeros(512, 4)]) * -.7
torch.testing.assert_close(m._LoRAMergeBase._expand_patch_to_diff(data["model_patches"][target]),
                           expected_qkv, atol=2e-6, rtol=2e-5)

# Include LoCon and the fp32 -> fp16 overflow counterexample in real serialization.
data["model_patches"]["diffusion_model.conv.weight"] = m.LoRAAdapter(set(), (
    torch.randn(2, 1, 1, 1), torch.randn(1, 2, 1, 1), 1., torch.randn(1, 1, 3, 3), None, None))
data["model_patches"]["diffusion_model.blocks.0.mlp.fc1.weight"] = m.LoRAAdapter(set(), (
    torch.full((8, 1), 1e5), torch.full((1, 4), 1e-5), .5, None, None, None))
with tempfile.TemporaryDirectory() as tmp:
    path = m.SaveMergedLoRA().save_lora(data, tmp, "roundtrip")[0]
    saved = load_file(path)
    assert all(torch.isfinite(v).all() for v in saved.values())
    reloaded, _ = comfy.sd.load_lora_for_models(base, None, saved, 1., 1.)
    assert set(reloaded.patches) == set(data["model_patches"]), (set(reloaded.patches), set(data["model_patches"]))
    for key, patch in data["model_patches"].items():
        weight = base.model.state_dict()[key].float()
        expected = comfy.lora.calculate_weight([(1.3, patch, 1., None, None)], weight.clone(), key)
        actual = comfy.lora.calculate_weight(reloaded.patches[key], weight.clone(), key)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5, msg=key)
    original_bytes = Path(path).read_bytes()
    data["model_patches"][target] = ("diff", (torch.full((768, 4), float("inf")),))
    try:
        m.SaveMergedLoRA().save_lora(data, tmp, "roundtrip")
        raise AssertionError("Non-finite export was accepted")
    except ValueError as exc:
        assert "Non-finite" in str(exc)
    assert Path(path).read_bytes() == original_bytes
    # CLIP branch uses its own signed strength, including dense bias updates.
    clip = TinyClip()
    te = 'text_encoder.text_model.encoder.layers.0.self_attn.q_proj'
    clip_item = dict(name='tiny-clip', lora={te + '.diff': torch.ones(4, 4),
                                          te + '.diff_b': torch.arange(4.)}, strength=1., clip_strength=-.4)
    clip_result = opt.optimize_merge(base, [clip_item], 1.5, clip=clip,
        optimization_mode='additive', patch_compression='disabled')
    clip_data = clip_result[4]
    clip_path = m.SaveMergedLoRA().save_lora(clip_data, tmp, 'clip')[0]
    _, loaded_clip = comfy.sd.load_lora_for_models(base, clip, load_file(clip_path), 1., 1.)
    assert set(loaded_clip.patcher.patches) == set(clip_data['clip_patches'])
    for key, patch in clip_data['clip_patches'].items():
        weight = clip.cond_stage_model.state_dict()[key].float()
        expected = comfy.lora.calculate_weight([(clip_data['clip_strength'], patch, 1., None, None)], weight.clone(), key)
        actual = comfy.lora.calculate_weight(loaded_clip.patcher.patches[key], weight.clone(), key)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
# Experimental global modes: real H3 partial QKV + AdaLN/norm payloads, signed
# model/CLIP weights, complete save -> stock reload, and method provenance.
import json
from safetensors import safe_open
te = 'text_encoder.text_model.encoder.layers.0.self_attn.q_proj'
exp_a = {**item, 'lora': {**sd, te + '.diff': torch.eye(4), te + '.diff_b': torch.arange(4.)},
         'clip_strength': -.3}
exp_b = {**item, 'name': 'tiny-h3-style', 'strength': .4, 'clip_strength': .6,
         'lora': {k: (v.clone() if k.endswith('.alpha') else v.clone() * -1.5) for k, v in exp_a['lora'].items()}}
for mode, cfg in (('np_lora', dict(version=1, subject_slot=1, style_slot=2, strength=.5, rank=0, energy=1.)),
                  ('ct_merge', dict(version=1, common_rank=1, residual_rank=2, scale=1.))):
    exp_result = opt.optimize_merge(base, [exp_a, exp_b], 1.3, clip=clip,
        optimization_mode='global', merge_strategy_override=mode,
        _experimental_config=cfg, patch_compression='disabled', normalize_keys='enabled')
    exp_data = exp_result[4]
    assert exp_data['merge_metadata']['mode'] == mode
    assert exp_data['merge_metadata']['experimental'] == cfg
    actual_qkv = m._LoRAMergeBase._expand_patch_to_diff(exp_data['model_patches'][target])
    torch.testing.assert_close(actual_qkv[256:], torch.zeros(512, 4))
    bias_key = 'diffusion_model.blocks.0.adaln_proj.linear.bias'
    torch.testing.assert_close(m._LoRAMergeBase._expand_patch_to_diff(exp_data['model_patches'][bias_key]),
                               sd['diffusion_model.blocks.0.adaln_proj.linear.diff_b'] * (-.7 - .6))
    with tempfile.TemporaryDirectory() as tmp:
        exp_path = m.SaveMergedLoRA().save_lora(exp_data, tmp, mode)[0]
        with safe_open(exp_path, framework='pt') as f:
            assert json.loads(f.metadata()['merge_experimental']) == cfg
        loaded_model, loaded_exp_clip = comfy.sd.load_lora_for_models(base, clip, load_file(exp_path), 1., 1.)
        for patches, reloaded_patcher, module, strength in (
                (exp_data['model_patches'], loaded_model, base.model, exp_data['output_strength']),
                (exp_data['clip_patches'], loaded_exp_clip.patcher, clip.cond_stage_model, exp_data['clip_strength'])):
            assert set(reloaded_patcher.patches) == set(patches)
            for key, patch in patches.items():
                weight = module.state_dict()[key].float()
                expected = comfy.lora.calculate_weight([(strength, patch, 1., None, None)], weight.clone(), key)
                actual = comfy.lora.calculate_weight(reloaded_patcher.patches[key], weight.clone(), key)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5, msg=mode + ': ' + key)
    del exp_result, loaded_model, loaded_exp_clip, reloaded_patcher
print("PASS: real ComfyUI mapping/application, partial QKV, signed alpha, dense AdaLN/bias/norm, LoCon, fp32 export, atomic rejection, signed CLIP weights/bias, NP-LoRA/CT-Merging round trips and metadata")
# Release while Comfy's modules are still alive (avoid interpreter-shutdown
# ModelPatcher destructor warnings masking useful test output).
del result, reloaded, base, opt, clip, clip_result, loaded_clip, _
import gc
gc.collect()
