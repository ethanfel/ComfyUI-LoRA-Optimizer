"""Opt-in contract, independent matrix references, and complete tuner replay."""
import copy
import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from tests.test_lora_optimizer import lora_optimizer as m
from tests.test_phase1_correctness import ToyPatcher, entry, factors

exp = m._experimental
NP = dict(version=1, subject_slot=1, style_slot=2, strength=0.5, rank=0, energy=1.0)
CT = dict(version=1, common_rank=1, residual_rank=2, scale=1.0)


class ExperimentalMath(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(51)

    def rand(self, rows=8, cols=6):
        return torch.randn(rows, cols, generator=self.g, dtype=torch.float64)

    def test_np_matches_dense_closed_form_with_signed_strengths(self):
        content, style = self.rand(), self.rand()
        params = {**NP, "rank": 2, "strength": 2.5}
        _, _, vh = torch.linalg.svd(style, full_matrices=False)
        p = vh[:2].T @ vh[:2]
        expected = 0.3 * style + (-1.4 * content) @ torch.linalg.inv(torch.eye(6) + 2.5 * p)
        actual = exp.merge([(content, -1.4), (style, .3)], "np_lora", params)
        torch.testing.assert_close(actual, expected)

    def test_np_zero_projection_is_additive(self):
        c, s = self.rand(), self.rand()
        with mock.patch.object(exp, "_svd", side_effect=AssertionError("no SVD needed")):
            torch.testing.assert_close(exp.merge([(c, 2), (s, -1)], "np_lora", {**NP, "strength": 0}), 2*c-s)

    def test_np_roles_are_explicit_and_order_aware(self):
        c, s = self.rand(), self.rand()
        expected = exp.merge([(c, 1), (s, 1)], "np_lora", NP)
        actual = exp.merge([(s, 1), (c, 1)], "np_lora", NP, [8, 3], [3, 8])
        torch.testing.assert_close(actual, expected)
        swapped = exp.merge([(c, 1), (s, 1)], "np_lora", {**NP, "style_slot": 1, "subject_slot": 2})
        self.assertFalse(torch.allclose(swapped, expected))

    def test_np_orthogonal_overlap_and_energy(self):
        style = torch.diag(torch.tensor([4., 1., 0.]))
        content = torch.eye(3)
        result = exp.merge([(content, 1), (style, 1)], "np_lora", {**NP, "energy": .9, "strength": 1.})
        torch.testing.assert_close(result, style + torch.diag(torch.tensor([.5, 1., 1.])))

    def test_zero_style_does_not_invent_a_subspace(self):
        c = self.rand()
        torch.testing.assert_close(exp.merge([(c, 1), (c, 0)], "np_lora", NP), c)

    def test_ct_matches_independent_projector_reference(self):
        # Explicit dense eigendecomposition (production uses thin SVD).
        ds = [self.rand(12, 9), -0.7*self.rand(12, 9), .2*self.rand(12, 9)]
        tasks = [torch.linalg.svd(d, full_matrices=False) for d in ds]
        us = [t[0][:, :2] for t in tasks]
        vs = [t[2][:2].T for t in tasks]
        ss = [t[1][:2] for t in tasks]
        projector = sum(u @ u.T for u in us)/3
        _, eig = torch.linalg.eigh(projector)
        common = eig[:, -1:]
        p, sc, vh = torch.linalg.svd(sum(common.T @ d for d in ds)/3, full_matrices=False)
        left = torch.cat([common @ p] + [u-common@(common.T@u) for u in us], 1)
        right = torch.cat([vh.T] + vs, 1)
        def polar(x):
            u, s, v = torch.linalg.svd(x, full_matrices=False)
            live = s > s.max()*max(x.shape)*torch.finfo(x.dtype).eps
            return u[:, live] @ v[live]
        scales = torch.cat([sc.square().mean().sqrt().expand(1)] +
                           [s.square().mean().sqrt().expand(2) for s in ss])
        expected = (polar(left)*scales) @ polar(right).T
        torch.testing.assert_close(exp.merge([(d, 1.) for d in ds], "ct_merge", CT), expected)

    def test_factor_svd_reconstructs_mixed_rank_and_signed_updates(self):
        up, down = self.rand(40, 3), self.rand(3, 24)
        u, s, vh = exp.factor_svd(-.7*up, down)
        torch.testing.assert_close((u*s)@vh, -.7*up@down)
        u2, s2, vh2 = exp.factor_svd(-.7*up, down, rank=2)
        dense_u, dense_s, dense_vh = torch.linalg.svd(-.7*up@down, full_matrices=False)
        torch.testing.assert_close((u2*s2)@vh2, (dense_u[:, :2]*dense_s[:2])@dense_vh[:2])

    def test_ct_permutation_and_scale(self):
        ds = [(self.rand(), 1.), (self.rand(), -.3), (self.rand(), 2.)]
        result = exp.merge(ds, "ct_merge", CT)
        torch.testing.assert_close(exp.merge(ds[::-1], "ct_merge", CT), result)
        torch.testing.assert_close(exp.merge(ds, "ct_merge", {**CT, "scale": .5}), .5*result)

    def test_ct_tied_consensus_is_order_independent(self):
        a = torch.diag(torch.tensor([1., 0., 0.]))
        b = torch.diag(torch.tensor([0., 2., 0.]))
        torch.testing.assert_close(exp.merge([(a, 1), (b, 1)], "ct_merge", CT),
                                   exp.merge([(b, 1), (a, 1)], "ct_merge", CT))

    def test_ct_identical_and_cancelling_updates(self):
        a = torch.diag(torch.tensor([2., 0., 0.]))
        torch.testing.assert_close(exp.merge([(a, 1), (a, 1)], "ct_merge", CT), a)
        torch.testing.assert_close(exp.merge([(a, 1), (a, -1)], "ct_merge", CT), torch.zeros_like(a))

    def test_vectors_unique_targets_and_zero_ct_are_additive(self):
        a = torch.tensor([1., -2.])
        for mode, cfg in (("np_lora", NP), ("ct_merge", CT)):
            torch.testing.assert_close(exp.merge([(a, 2), (a, -1)], mode, cfg), a)
            torch.testing.assert_close(exp.merge([(a.reshape(1, 2), 2)], mode, cfg), 2*a.reshape(1, 2))
        z = torch.zeros(2, 2)
        torch.testing.assert_close(exp.merge([(z, 1), (z, 1)], "ct_merge", CT), z)

    def test_shared_convolution_and_nonfinite_inputs_rejected(self):
        for mode, cfg in (("np_lora", NP), ("ct_merge", CT)):
            for a in (torch.ones(2, 2, 3, 3), torch.full((2, 2), float("nan"))):
                with self.subTest(mode=mode), self.assertRaises(exp.UnsupportedMerge):
                    exp.merge([(a, 1), (a, 1)], mode, cfg)

    def test_preserved_overlay_bypasses_projection_and_scaling(self):
        c, s, p = self.rand(), self.rand(), self.rand()
        opt = m.LoRAOptimizer()
        for mode, cfg in (("np_lora", NP), ("ct_merge", CT)):
            expected = exp.merge([(c, 1), (s, .5)], mode, cfg) + 2*p
            actual = opt._merge_diffs([(p, 2), (c, 1), (s, .5)], mode,
                                     preserve_flags=[True, False, False],
                                     experimental_config=cfg, source_indices=[0, 1, 2], role_indices=[1, 2])
            torch.testing.assert_close(actual, expected.to(actual.dtype))


class ExperimentalIntegration(unittest.TestCase):
    def setUp(self):
        self.model = ToyPatcher()
        self.opt = m.LoRAOptimizer()
        self.stack = [entry(factors(), name="subject"), entry(factors(), name="style")]
        self.stack[1]["lora"]["layer.lora_A.weight"] = torch.tensor([[1., -1.]])
        self.mapping = {"layer": "layer.weight", "audio_layer": "audio_layer.weight"}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(m._LoRAMergeBase, "_get_compute_device", return_value=torch.device("cpu")).start()
        mock.patch.object(m.comfy.lora, "model_lora_keys_unet", side_effect=lambda *_: dict(self.mapping)).start()
        mock.patch.object(m, "AUTOTUNER_MEMORY_DIR", self.tmp.name).start()
        mock.patch.object(m.LoRAAutoTuner, "_gc_autotuner_memory").start()

    def tune(self, node=None, **kw):
        settings = {"top_n": 1, "scoring_device": "cpu", **kw}
        return (node or m.LoRAAutoTuner()).auto_tune(
            self.model, self.stack, 1., **settings)

    def delta(self, data, key="layer.weight"):
        return self.opt._expand_patch_to_diff(data["model_patches"][key])

    def test_node_registration_and_disabled_options(self):
        self.assertIs(m.NODE_CLASS_MAPPINGS["LoRAExperimentalOptions"], m.LoRAExperimentalOptions)
        for cls in (m.LoRAAutoTuner, m.LoRAAutoTunerSettings):
            self.assertEqual(cls.INPUT_TYPES()["optional"]["experimental_options"][0], "LORA_EXPERIMENTAL_OPTIONS")
            self.assertEqual(list(cls.INPUT_TYPES()["optional"])[-1], "experimental_options")
        for value in (None, {}, {"enabled": False}, {"np_lora": False, "ct_merge": False}):
            self.assertIsNone(exp.options(value))

    def test_invalid_options_fail_before_work(self):
        for value in ({"version": 999}, {"np_strength": float("nan")}, {"np_style_slot": 1},
                      {"ct_residual_rank": 0}, {"trials_per_method": 4}, {"np_rank": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.tune(experimental_options=value)

    def test_disabled_changes_nothing_in_grid_identity_or_cached_output(self):
        grid = copy.deepcopy(m._generate_param_grid())
        node = m.LoRAAutoTuner()
        normal = self.tune(node)
        for opts in ({"enabled": False}, {"np_lora": False, "ct_merge": False}, {}):
            self.assertIs(self.tune(node, experimental_options=opts), normal)
            self.assertEqual(m.LoRAAutoTuner.IS_CHANGED(self.model, self.stack, 1.),
                             m.LoRAAutoTuner.IS_CHANGED(self.model, self.stack, 1., experimental_options=opts))
        self.assertEqual(m._generate_param_grid(), grid)
        self.assertNotIn("experimental_options", normal[4])

    def test_trials_added_without_reducing_stable_budget(self):
        stable = self.tune()[4]["top_n"][0]["config"]
        result = self.tune(experimental_options=exp.DEFAULTS)
        configs = [r["config"] for r in result[4]["top_n"]]
        self.assertEqual(len(configs), 4)
        self.assertIn(stable, configs)
        self.assertEqual({c["merge_mode"] for c in configs if "experimental" in c}, set(exp.MODES))
        self.assertTrue(any(c.get("experimental_baseline") for c in configs))
        self.assertIn("EXPERIMENTAL", result[2])
        self.assertIn("1: subject, 2: style", result[2])

    def test_independent_switches_and_bounded_trials(self):
        for mode in exp.MODES:
            opts = {"np_lora": mode == "np_lora", "ct_merge": mode == "ct_merge", "trials_per_method": 3}
            configs, skipped = exp.candidates(opts, self.stack)
            self.assertEqual(len(configs), 4)
            self.assertFalse(skipped)
            self.assertEqual({c["merge_mode"] for c in configs[1:]}, {mode})

    def test_three_participants_skip_np_but_keep_ct(self):
        self.stack.append(entry(factors(), name="third"))
        result = self.tune(experimental_options=exp.DEFAULTS)
        self.assertTrue(any("exactly two" in s for s in result[4]["experimental_skipped"]))
        self.assertEqual([r["config"]["merge_mode"] for r in result[4]["top_n"] if "experimental" in r["config"]], ["ct_merge"])

    def test_fresh_cached_and_selector_replay_both_methods(self):
        node = m.LoRAAutoTuner()
        result = self.tune(node, experimental_options=exp.DEFAULTS, output_mode="tuning_only")
        # JSON roundtrip also proves that no tensors/models leak into settings.
        td = json.loads(json.dumps(result[4]))
        for row in td["top_n"]:
            if "experimental" not in row["config"]:
                continue
            rank = row["rank"]
            selected = m.LoRAMergeSelector().select_merge(self.model, self.stack, td, rank, 1.)
            replay = self.tune(node, experimental_options=exp.DEFAULTS, selection=rank)
            torch.testing.assert_close(self.delta(selected[3]), self.delta(replay[5]))
            self.assertEqual(selected[3]["merge_metadata"]["experimental"], row["config"]["experimental"])
            self.assertIs(self.tune(node, experimental_options=exp.DEFAULTS, selection=rank), replay)

    def test_persistent_memory_preserves_all_candidates_and_parameters(self):
        result = self.tune(experimental_options=exp.DEFAULTS, memory_mode="auto", output_mode="tuning_only")
        with mock.patch.object(m, "_generate_param_grid", side_effect=AssertionError("must load memory")):
            cached = self.tune(experimental_options=exp.DEFAULTS, memory_mode="auto", output_mode="tuning_only")
        self.assertEqual(result[4]["top_n"], cached[4]["top_n"])
        self.assertEqual(len(cached[4]["top_n"]), 4)

    def test_persistent_replay_each_experiment_matches_selector(self):
        result = self.tune(experimental_options=exp.DEFAULTS, memory_mode="auto", output_mode="tuning_only")
        for row in result[4]["top_n"]:
            if "experimental" not in row["config"]:
                continue
            expected = m.LoRAMergeSelector().select_merge(self.model, self.stack, result[4], row["rank"], 1.)
            with mock.patch.object(m, "_generate_param_grid", side_effect=AssertionError("memory replay")):
                actual = self.tune(experimental_options=exp.DEFAULTS, memory_mode="auto", selection=row["rank"])
            torch.testing.assert_close(self.delta(actual[5]), self.delta(expected[3]))

    def test_factor_path_matches_dense_math_with_rank_alpha_and_signed_strengths(self):
        self.model.model.layer = torch.nn.Linear(48, 64)
        g = torch.Generator().manual_seed(72)
        self.stack = []
        dense = []
        for i, (rank, alpha, weight) in enumerate(((2, -3., -.7), (3, 2., .4))):
            up = torch.randn(64, rank, generator=g)
            down = torch.randn(rank, 48, generator=g)
            self.stack.append(dict(name=str(i), strength=weight, lora={
                "layer.lora_A.weight": down, "layer.lora_B.weight": up, "layer.alpha": torch.tensor(alpha)}))
            dense.append((up@down*(alpha/rank), weight))
        for mode, cfg in (("np_lora", NP), ("ct_merge", CT)):
            expected = exp.merge(dense, mode, cfg)
            with mock.patch.object(exp, "factor_svd", wraps=exp.factor_svd) as thin:
                actual = self.opt.optimize_merge(self.model, self.stack, 1., optimization_mode="global",
                    merge_strategy_override=mode, _experimental_config=cfg, patch_compression="disabled")
            self.assertGreater(thin.call_count, 0)
            # CT polar alignment amplifies tiny float32 SVD differences near
            # rank deficiency. Both paths are within 7e-6 relative Frobenius
            # error of a float64 reference on this fixture (NP within 1e-7).
            delta = self.delta(actual[4])
            self.assertLess(float((delta-expected).norm()/expected.norm()), 2e-5)
            torch.testing.assert_close(delta, expected, atol=2e-4, rtol=2e-4)

    def test_single_active_adapter_reports_skipped_experiments(self):
        self.stack[0]["strength"] = 0.
        result = self.tune(experimental_options=exp.DEFAULTS)
        self.assertIn("Experimental candidates skipped", result[2])

    def test_legacy_and_simple_tuner_data_bridges_replay_experiments(self):
        result = self.tune(experimental_options=exp.DEFAULTS, output_mode="tuning_only")
        for row in result[4]["top_n"]:
            if "experimental" not in row["config"]:
                continue
            td = dict(result[4], top_n=[row])
            expected = m.LoRAMergeSelector().select_merge(self.model, self.stack, td, 1, 1.)[3]
            simple = m.LoRAOptimizerSimple().execute_simple(self.model, self.stack, 1., tuner_data=td)
            legacy = self.opt.execute_node(self.model, self.stack, 1., tuner_data=td, settings_source="from_tuner_data")
            torch.testing.assert_close(self.delta(simple[4]), self.delta(expected))
            torch.testing.assert_close(self.delta(legacy["result"][4]), self.delta(expected))
            self.assertNotIn("ui", legacy)
            bridge = self.opt.execute_node(self.model, self.stack, 1., tuner_data=td, settings_source="from_autotuner")
            self.assertNotIn("ui", bridge)
            self.assertIn("Merge Selector", bridge["result"][2])

    def test_disconnecting_and_changing_parameters_isolates_caches(self):
        node = m.LoRAAutoTuner()
        experiment = self.tune(node, experimental_options=exp.DEFAULTS)
        changed = self.tune(node, experimental_options={**exp.DEFAULTS, "np_strength": 3.})
        self.assertIsNot(experiment, changed)
        stable = self.tune(node)
        self.assertEqual(len(stable[4]["top_n"]), 1)
        self.assertNotIn("experimental_options", stable[4])

    def test_unsupported_candidate_reports_and_keeps_stable_results(self):
        original = exp.merge
        def fail_np(ds, mode, *args):
            if mode == "np_lora":
                raise exp.UnsupportedMerge("synthetic unsupported tensor")
            return original(ds, mode, *args)
        with mock.patch.object(exp, "merge", side_effect=fail_np):
            result = self.tune(experimental_options=exp.DEFAULTS)
        self.assertEqual(len(result[4]["top_n"]), 3)
        self.assertIn("synthetic unsupported", result[2])

    def test_actual_spatial_convolution_is_not_missed_by_fast_scoring(self):
        self.model.model.conv = torch.nn.Conv2d(2, 2, 3)
        self.mapping["conv"] = "conv.weight"
        for item in self.stack:
            item["lora"]["conv.diff"] = torch.ones(2, 2, 3, 3)
        result = self.tune(experimental_options=exp.DEFAULTS, scoring_speed="turbo+", top_n=2)
        self.assertEqual(len(result[4]["experimental_skipped"]), 2)
        self.assertTrue(all("spatial convolution" in r for r in result[4]["experimental_skipped"]))
        self.assertFalse(any("experimental" in r["config"] for r in result[4]["top_n"]))
        self.assertIn("conv.weight", result[5]["model_patches"])

    def test_experimental_dataset_does_not_pollute_stable_records(self):
        with mock.patch.object(m.folder_paths, "get_user_directory", return_value=self.tmp.name):
            self.tune(experimental_options=exp.DEFAULTS, record_dataset="enabled")
        directory = os.path.join(self.tmp.name, "lora_optimizer_reports")
        self.assertFalse(os.path.exists(os.path.join(directory, "autotuner_dataset.jsonl")))
        with open(os.path.join(directory, "autotuner_experimental_dataset.jsonl")) as f:
            record = json.loads(f.readline())
        self.assertIn("experimental_notice", record["analysis"])

    def test_missing_method_metadata_and_formula_rejected(self):
        with self.assertRaisesRegex(ValueError, "settings"):
            self.opt.optimize_merge(self.model, self.stack, 1., optimization_mode="global", merge_strategy_override="np_lora")
        self.stack.append({"_merge_formula": "(1+2)"})
        with self.assertRaisesRegex(ValueError, "flat stack"):
            self.tune(experimental_options=exp.DEFAULTS)

    def test_bias_unique_targets_and_preserved_overlay_roundtrip(self):
        self.stack[0]["lora"].update({"layer.diff_b": torch.tensor([1., 2.]), **factors("audio_layer")})
        self.stack[1]["lora"]["layer.diff_b"] = torch.tensor([3., -2.])
        self.stack.insert(0, entry(factors(), name="turbo overlay", preserve=True))
        for mode, cfg in (("np_lora", NP), ("ct_merge", CT)):
            out = self.opt.optimize_merge(self.model, self.stack, 1., optimization_mode="global",
                                         merge_strategy_override=mode, _experimental_config=cfg)
            torch.testing.assert_close(self.delta(out[4], "layer.bias"), torch.tensor([4., 0.]))
            torch.testing.assert_close(self.delta(out[4], "audio_layer.weight"), torch.ones(2, 2))
            self.assertEqual(out[4]["merge_metadata"]["experimental_roles"], ["subject", "style"])
            captured = {}
            with mock.patch.object(m, "save_file", side_effect=lambda sd, path, metadata: captured.update(metadata)):
                m.SaveMergedLoRA().save_lora(out[4], self.tmp.name, mode)
            self.assertEqual(json.loads(captured["merge_experimental"]), cfg)

    def test_settings_node_defaults_and_simple_forwarding(self):
        cls = m.LoRAAutoTunerSettings
        defaults = {k: spec[1]["default"] for k, spec in cls.INPUT_TYPES()["required"].items()}
        normal = cls().build_settings(**defaults)[0]
        disabled = cls().build_settings(**defaults, experimental_options={"enabled": False})[0]
        self.assertEqual(normal, disabled)
        settings = cls().build_settings(**defaults, experimental_options=exp.DEFAULTS)[0]
        node = m.LoRAOptimizerSimple()
        call = mock.Mock(return_value=(self.model, None, "", "", {}, {}))
        node._autotuner_delegate = types.SimpleNamespace(auto_tune=call)
        node.execute_simple(self.model, self.stack, 1., settings=settings)
        self.assertEqual(call.call_args.kwargs["experimental_options"], exp.options(exp.DEFAULTS))


if __name__ == "__main__":
    unittest.main()
