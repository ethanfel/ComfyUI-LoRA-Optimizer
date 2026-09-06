"""Release regressions: tiny tensors, actual Comfy parameter naming."""
import copy
import tempfile
import types
import unittest
from unittest import mock

import torch
from tests.test_lora_optimizer import lora_optimizer as m


def entry(sd, name="adapter", **kwargs):
    return dict(name=name, lora=sd, strength=1.0, **kwargs)


def factors(prefix="layer", rows=2, cols=2):
    return {prefix + ".lora_A.weight": torch.ones(1, cols),
            prefix + ".lora_B.weight": torch.ones(rows, 1)}


class ToyPatcher:
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.layer = torch.nn.Linear(2, 2)
        self.model.audio_layer = torch.nn.Linear(2, 2)
        self.patches = {}

    def clone(self):
        result = copy.copy(self)
        result.patches = dict(self.patches)
        return result

    def add_patches(self, patches, strength=1.0):
        self.patches.update(patches)
        return list(patches)


class Phase1Correctness(unittest.TestCase):
    def setUp(self):
        self.opt = m.LoRAOptimizer()
        self.model = ToyPatcher()
        self.mapping = {"layer": "layer.weight", "audio_layer": "audio_layer.weight"}
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(m._LoRAMergeBase, "_get_compute_device", return_value=torch.device("cpu")).start()
        mock.patch.object(m.comfy.lora, "model_lora_keys_unet", side_effect=lambda *_: dict(self.mapping)).start()
        mock.patch.object(m.LoRAOptimizer, "_save_report_to_disk").start()

    def merge(self, stack, **kwargs):
        return self.opt.optimize_merge(self.model, stack, 1.0, optimization_mode="additive",
                                       patch_compression="disabled", **kwargs)

    def save(self, patch, key="layer.weight", prefix="layer"):
        captured = {}
        data = dict(model_patches={key: patch}, clip_patches={}, key_map={key: prefix},
                    output_strength=1.0, clip_strength=1.0, sum_rank=1)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                m, "save_file", side_effect=lambda sd, *a, **kw: captured.update(sd)):
            m.SaveMergedLoRA().save_lora(data, tmp, "roundtrip", save_rank=1)
        return captured

    def test_h3_real_parameter_targets(self):
        target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
        keys = {target[:-7]: target}
        m._LoRAMergeBase._add_minimax_h3_qkv_aliases(keys, {target: torch.zeros(12, 2)})
        self.assertEqual(keys["diffusion_model.blocks.0.attn.to_q"], (target, (0, 0, 4)))

    def test_h3_partial_qkv_refuses_with_zero_missing_slices(self):
        target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
        diff = torch.ones(4, 2)
        merged = m._LoRAMergeBase._refuse_fused_qkv_patches({(target, (0, 0, 4)): ("diff", (diff,))})
        self.assertIn(target, merged)
        torch.testing.assert_close(m._LoRAMergeBase._expand_patch_to_diff(merged[target]),
                                   torch.cat([diff, torch.zeros(8, 2)]))

    def test_sdxl_ff_not_h3(self):
        sd = factors("unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj")
        sd.update(factors("text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj"))
        self.assertEqual(self.opt._detect_architecture(sd), "sdxl")

    def test_ambiguous_peft_layout_rejected(self):
        sd = {k.replace(".weight", ".default.weight"): v
              for k, v in factors("blocks.0.attn.qkv_proj", rows=768).items()}
        with self.assertRaisesRegex(ValueError, "layout"):
            self.opt._normalize_keys_minimax_h3(sd)

    def test_sd15_unet_and_text_encoder_not_h3_or_acestep(self):
        sd = factors('unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj')
        sd.update(factors('text_encoder.text_model.encoder.layers.0.self_attn.q_proj'))
        self.assertEqual(self.opt._detect_architecture(sd), 'sd15')

    def test_invalid_header_alpha_rejected(self):
        for alpha in ('NaN', 'Infinity', 'invalid'):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, 'alpha'):
                self.opt._apply_minimax_h3_metadata_alpha(factors('blocks.0.mlp.fc1'), {'alpha': alpha})

    def test_mixed_dense_dtype_does_not_overflow_to_fp16(self):
        out = self.merge([entry({'layer.diff': torch.ones(2, 2, dtype=torch.float16)}),
                          entry({'layer.diff': torch.full((2, 2), 1e5)}, name='b')])
        actual = self.opt._expand_patch_to_diff(out[4]['model_patches']['layer.weight'])
        torch.testing.assert_close(actual, torch.full((2, 2), 100001.))

    def test_explicit_comfy_peft_layout_not_permuted(self):
        sd = {k.replace(".weight", ".default.weight"): v
              for k, v in factors("blocks.0.attn.qkv_proj", rows=768).items()}
        sd["blocks.0.attn.qkv_proj.lora_B.default.weight"] = torch.arange(768.).view(768, 1)
        norm = self.opt._normalize_keys_minimax_h3(sd, layout="comfy")
        torch.testing.assert_close(norm["diffusion_model.blocks.0.attn.to_q.lora_B.weight"],
                                   torch.arange(256.).view(256, 1))

    def test_dense_and_bias_are_applied(self):
        out = self.merge([entry({"layer.diff": torch.ones(2, 2), "layer.diff_b": torch.ones(2)})])
        patches = out[4]["model_patches"]
        self.assertEqual(set(patches), {"layer.weight", "layer.bias"})
        torch.testing.assert_close(m._LoRAMergeBase._expand_patch_to_diff(patches["layer.bias"]), torch.ones(2))

    def test_file_dora_rejected(self):
        sd = factors()
        sd["layer.dora_scale"] = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "DoRA"):
            self.merge([entry(sd), entry(factors(), name="other")])

    def test_locon_middle_exported(self):
        mid = torch.ones(1, 1, 3, 3)
        patch = m.LoRAAdapter(set(), (torch.ones(2, 1, 1, 1), torch.ones(1, 2, 1, 1), 1., mid, None, None))
        self.assertIn("layer.lora_mid.weight", self.save(patch))

    def test_export_preserves_finite_float32_factors(self):
        patch = m.LoRAAdapter(set(), (torch.full((2, 1), 1e5), torch.full((1, 2), 1e-5), 1., None, None, None))
        sd = self.save(patch)
        self.assertTrue(torch.isfinite(sd["layer.lora_up.weight"]).all())
        torch.testing.assert_close(sd["layer.lora_up.weight"] @ sd["layer.lora_down.weight"], torch.ones(2, 2))

    def test_in_memory_cache_tracks_tensor_replacement_and_mutation(self):
        item = entry(factors())
        key = lambda: self.opt._compute_cache_key([item], 1., 1., "disabled")
        first = key()
        item["lora"]["layer.lora_B.weight"] = torch.full((2, 1), 3.)
        second = key()
        item["lora"]["layer.lora_B.weight"].add_(1)
        third = key()
        self.assertEqual(len({first, second, third}), 3)

    def test_single_lora_honors_audio_filter_and_emits_data(self):
        sd = dict(factors(), **factors("audio_layer"))
        out = self.merge([entry(sd, key_filter="audio_only")])
        self.assertIsNotNone(out[4])
        self.assertEqual(set(out[4]["model_patches"]), {"audio_layer.weight"})

    def test_extract_auto_rank_is_bounded(self):
        up, down, alpha = m._extract_lora_svd(torch.eye(8), 1, "auto", .99)
        self.assertEqual(down.shape[0], 1)

    def test_pdd_rejected_before_partial_apply(self):
        sd = factors()
        sd["proj_out.weight"] = torch.ones(32, 2, 2)
        with self.assertRaisesRegex(ValueError, "PDD"):
            self.merge([entry(sd, metadata={"pdd_num_steps": "32"})])

    def test_normalization_is_idempotent_and_retains_layout_provenance(self):
        sd = factors('blocks.0.attn.qkv_proj', rows=768)
        sd['blocks.0.attn.qkv_proj.lora_B.weight'] = torch.arange(768.).view(-1, 1)
        item = entry(sd, h3_layout='diffsynth')
        first = self.opt._normalize_stack([item], 'enabled')
        second = self.opt._normalize_stack(first, 'enabled')
        for key in first[0]['lora']:
            torch.testing.assert_close(first[0]['lora'][key], second[0]['lora'][key])
        self.assertEqual(item['h3_layout'], 'diffsynth')
        self.assertEqual(first[0]['h3_layout'], 'comfy')
        self.assertEqual(second[0]['h3_source_layout'], 'diffsynth')

    def test_dense_interleaved_qkv_conversion(self):
        raw = torch.arange(768.).view(-1, 1)
        norm = self.opt._normalize_keys_minimax_h3({'blocks.0.attn.qkv_proj.diff': raw}, 'diffsynth')
        expected = torch.cat([raw[0:128], raw[384:512], raw[128:256], raw[512:640], raw[256:384], raw[640:768]])
        torch.testing.assert_close(norm['diffusion_model.blocks.0.attn.qkv_proj.diff'], expected)

    def test_dense_missing_or_wrong_shape_rejected(self):
        for sd in ({'absent.diff': torch.ones(2, 2)}, {'layer.diff_b': torch.ones(3)}):
            with self.subTest(sd=list(sd)), self.assertRaisesRegex(ValueError, 'missing|shape'):
                self.merge([entry(sd)])

    def test_duplicate_normalized_h3_target_rejected(self):
        sd = dict(factors('blocks.0.mlp.fc1'), **factors('diffusion_model.blocks.0.mlp.fc1'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.opt._normalize_keys_minimax_h3(sd)

    def test_dense_vectors_survive_strategies(self):
        for mode in ('weighted_average', 'weighted_sum', 'normalize', 'ties', 'slerp', 'consensus'):
            with self.subTest(mode=mode):
                out = self.opt.optimize_merge(self.model, [entry({'layer.diff_b': torch.tensor([1., -2.])}),
                    entry({'layer.diff_b': torch.tensor([2., -1.])}, name='b')], 1.,
                    optimization_mode='global', merge_strategy_override=mode, patch_compression='disabled')
                self.assertIn('layer.bias', out[4]['model_patches'])
                self.assertTrue(torch.isfinite(self.opt._expand_patch_to_diff(out[4]['model_patches']['layer.bias'])).all())

    def test_cached_merge_recomputes_changed_payload(self):
        item = entry(factors())
        a = self.merge([item])
        b = self.merge([item])
        torch.testing.assert_close(self.opt._expand_patch_to_diff(a[4]['model_patches']['layer.weight']),
                                   self.opt._expand_patch_to_diff(b[4]['model_patches']['layer.weight']))
        item['lora']['layer.lora_B.weight'].mul_(3.)
        c = self.merge([item])
        torch.testing.assert_close(self.opt._expand_patch_to_diff(c[4]['model_patches']['layer.weight']), torch.full((2, 2), 3.))

    def test_content_hash_invalidates_on_inplace_change(self):
        item = entry(factors())
        a = m.LoRAAutoTuner._memo_content_hash(item)
        item['lora']['layer.lora_B.weight'].add_(2.)
        self.assertNotEqual(a, m.LoRAAutoTuner._memo_content_hash(item))

    def test_tuner_change_detection_tracks_payload_and_cleaning(self):
        item = entry(factors())
        changed = lambda **kw: m.LoRAAutoTuner.IS_CHANGED(self.model, [item], 1., **kw)
        a = changed()
        self.assertNotEqual(a, changed(star_eta=80.))
        item['lora']['layer.lora_B.weight'].add_(1.)
        self.assertNotEqual(a, changed())

    def test_tuner_fresh_and_cached_preserve_cleaning_settings(self):
        tuner = m.LoRAAutoTuner()
        stack = [entry(factors(), name='a'), entry(factors(), name='b')]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(m, 'AUTOTUNER_MEMORY_DIR', tmp), \
                mock.patch.object(m.LoRAAutoTuner, '_gc_autotuner_memory'):
            out = tuner.auto_tune(self.model, stack, 1., top_n=1, scoring_device='cpu',
                                  star_eta=75., tame_layers=.5, community_cache='disabled')
            self.assertEqual(out[4]['star_eta'], 75.)
            self.assertEqual(out[4]['tame_layers'], .5)
            cached = tuner.auto_tune(self.model, stack, 1., top_n=1, scoring_device='cpu',
                                     star_eta=75., tame_layers=.5, community_cache='disabled')
            self.assertIs(cached, out)
            selector = m.LoRAMergeSelector()
            selected = selector.select_merge(self.model, stack, out[4], 1, 1.)
            torch.testing.assert_close(self.opt._expand_patch_to_diff(selected[3]['model_patches']['layer.weight']),
                                       self.opt._expand_patch_to_diff(out[5]['model_patches']['layer.weight']))

    def test_partial_refiner_qv_independent_ranks_and_signed_alpha(self):
        target = 'diffusion_model.token_refiner.blocks.0.attn.qkv_proj.weight'
        q = m.LoRAAdapter(set(), (torch.ones(4, 1), torch.ones(1, 2), -2., None, None, None))
        v = m.LoRAAdapter(set(), (torch.ones(4, 2), torch.ones(2, 2), 1., None, None, None))
        fused = self.opt._refuse_fused_qkv_patches({(target, (0, 0, 4)): q, (target, (0, 8, 4)): v})
        torch.testing.assert_close(self.opt._expand_patch_to_diff(fused[target]),
                                   torch.cat([torch.full((4, 2), -2.), torch.zeros(4, 2), torch.ones(4, 2)]))

    def test_zeroed_second_adapter_retains_single_filter(self):
        sd = dict(factors(), **factors('audio_layer'))
        b = entry(factors(), name='b')
        self.merge([entry(sd, key_filter='audio_only'), b])
        b['strength'] = 0.
        out = self.merge([entry(sd, key_filter='audio_only'), b])
        self.assertEqual(set(out[4]['model_patches']), {'audio_layer.weight'})
        self.assertEqual(out[4]['coverage'][0]['filtered'], 1)

    def test_h3_known_basis_conflict_and_normalization_requirement(self):
        sd = factors('blocks.0.attn.qkv_proj', rows=768)
        with self.assertRaisesRegex(ValueError, 'different AdaLN bases'):
            self.opt._normalize_stack([entry(sd, h3_basis='a'), entry(sd, name='b', h3_basis='b')], 'enabled')
        with self.assertRaisesRegex(ValueError, 'requires normalize_keys'):
            self.opt._normalize_stack([entry(sd, h3_layout='diffsynth')], 'disabled')

    def test_rejected_patch_target_is_not_reported_successful(self):
        with mock.patch.object(ToyPatcher, 'add_patches', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'rejected patch targets'):
                self.merge([entry(factors())])

    def test_inactive_ambiguous_h3_does_not_change_active_architecture(self):
        sd = {k.replace('.weight', '.default.weight'): v for k, v in factors('blocks.0.attn.qkv_proj', rows=768).items()}
        inactive = entry(sd)
        inactive['strength'] = 0.
        active = entry(factors('unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj'), name='sd15')
        self.opt._normalize_stack([inactive, active], 'enabled')
        self.assertEqual(self.opt._detected_arch, 'sd15')
