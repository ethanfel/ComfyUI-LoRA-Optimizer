import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _install_stubs():
    tmpdir = tempfile.gettempdir()

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = tmpdir
    folder_paths.add_model_folder_path = lambda *args, **kwargs: None
    folder_paths.get_temp_directory = lambda: tmpdir
    folder_paths.get_user_directory = lambda: tmpdir
    folder_paths.get_folder_paths = lambda _kind: [tmpdir]
    folder_paths.get_filename_list = lambda _kind: []
    folder_paths.get_full_path_or_raise = lambda _kind, name: name
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")

    def get_attr(obj, path):
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.value = 0

        def update(self, amount):
            self.value += amount

    utils.get_attr = get_attr
    utils.load_torch_file = lambda _path, safe_load=True: {}
    utils.ProgressBar = ProgressBar

    sd = types.ModuleType("comfy.sd")
    sd.load_lora_for_models = lambda model, clip, lora_dict, model_strength, clip_strength: (model, clip)

    lora = types.ModuleType("comfy.lora")
    lora.model_lora_keys_unet = lambda model, mapping: {}
    lora.model_lora_keys_clip = lambda clip, mapping: {}

    model_management = types.ModuleType("comfy.model_management")
    model_management.get_free_memory = lambda _device: 0

    weight_adapter = types.ModuleType("comfy.weight_adapter")
    weight_adapter_lora = types.ModuleType("comfy.weight_adapter.lora")
    weight_adapter_lokr = types.ModuleType("comfy.weight_adapter.lokr")
    weight_adapter_loha = types.ModuleType("comfy.weight_adapter.loha")

    class LoRAAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights

    class LoKrAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights

    class LoHaAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights

    weight_adapter_lora.LoRAAdapter = LoRAAdapter
    weight_adapter_lokr.LoKrAdapter = LoKrAdapter
    weight_adapter_loha.LoHaAdapter = LoHaAdapter

    comfy.utils = utils
    comfy.sd = sd
    comfy.lora = lora
    comfy.model_management = model_management

    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils
    sys.modules["comfy.sd"] = sd
    sys.modules["comfy.lora"] = lora
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.weight_adapter"] = weight_adapter
    sys.modules["comfy.weight_adapter.lora"] = weight_adapter_lora
    sys.modules["comfy.weight_adapter.lokr"] = weight_adapter_lokr
    sys.modules["comfy.weight_adapter.loha"] = weight_adapter_loha

    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = mock.MagicMock()
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = lambda state_dict, path, metadata=None: None
    safetensors.torch = safetensors_torch
    sys.modules["safetensors"] = safetensors
    sys.modules["safetensors.torch"] = safetensors_torch


if torch is not None:
    _install_stubs()
    MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lora_optimizer.py")
    SPEC = importlib.util.spec_from_file_location("lora_optimizer_under_test", MODULE_PATH)
    lora_optimizer = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(lora_optimizer)
else:
    lora_optimizer = None


def _make_model():
    layer = types.SimpleNamespace(weight=torch.zeros(1, 1))
    return types.SimpleNamespace(model=types.SimpleNamespace(layer=layer))


def _make_lora_entry(prefix_to_value, strength=1.0, clip_strength=None, key_filter="all", conflict_mode="all", name="demo"):
    lora = {}
    for prefix, value in prefix_to_value.items():
        lora[f"{prefix}.lora_up.weight"] = torch.tensor([[float(value)]], dtype=torch.float32)
        lora[f"{prefix}.lora_down.weight"] = torch.tensor([[1.0]], dtype=torch.float32)
    return {
        "name": name,
        "lora": lora,
        "strength": strength,
        "clip_strength": clip_strength,
        "key_filter": key_filter,
        "conflict_mode": conflict_mode,
    }


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LoRAOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.optimizer = lora_optimizer.LoRAOptimizer()
        self.model = _make_model()

    def test_target_groups_merge_aliases_for_same_target(self):
        groups = self.optimizer._build_target_groups(
            ["alias_a", "alias_b", "other"],
            {"alias_a": "layer.weight", "alias_b": "layer.weight", "other": "other.weight"},
            {},
        )

        self.assertEqual(set(groups.keys()), {"alias_a", "other"})
        self.assertEqual(groups["alias_a"]["aliases"], ["alias_a", "alias_b"])

    def test_group_analysis_detects_alias_overlap(self):
        active_loras = [
            _make_lora_entry({"alias_a": 1.0}, name="A"),
            _make_lora_entry({"alias_b": -1.0}, name="B"),
        ]
        target_groups = self.optimizer._build_target_groups(
            ["alias_a", "alias_b"],
            {"alias_a": "layer.weight", "alias_b": "layer.weight"},
            {},
        )

        analysis = self.optimizer._run_group_analysis(
            target_groups, active_loras, self.model, None, torch.device("cpu")
        )

        self.assertEqual(analysis["prefix_count"], 1)
        stats = analysis["prefix_stats"]["alias_a"]
        self.assertEqual(stats["n_loras"], 2)
        self.assertGreater(stats["conflict_ratio"], 0.99)

    def test_same_lora_aliases_are_aggregated_before_analysis(self):
        target_group = {
            "target_key": "layer.weight",
            "is_clip": False,
            "aliases": ["alias_a", "alias_b"],
            "label_prefix": "alias_a",
        }
        active_loras = [
            _make_lora_entry({"alias_a": 1.0, "alias_b": 2.0}, name="A"),
        ]

        prepared = self.optimizer._prepare_group_diffs(
            target_group, active_loras, self.model, None, torch.device("cpu")
        )

        self.assertAlmostEqual(prepared["diffs"][0].item(), 3.0)

    def test_exact_linear_patch_matches_dense_sum(self):
        target_group = {
            "target_key": "layer.weight",
            "is_clip": False,
            "aliases": ["alias_a", "alias_b"],
            "label_prefix": "alias_a",
        }
        active_loras = [
            _make_lora_entry({"alias_a": 1.0}, name="A"),
            _make_lora_entry({"alias_b": 2.0}, name="B"),
        ]

        patch_info = self.optimizer._build_exact_linear_patch(
            target_group, active_loras, raw_n_loras=2, mode="weighted_sum"
        )

        diff = self.optimizer._expand_patch_to_diff(patch_info["patch"])
        self.assertAlmostEqual(diff.item(), 3.0)

    def test_expand_patch_to_diff_supports_lokr_and_loha(self):
        lokr_patch = lora_optimizer.LoKrAdapter(
            set(),
            (
                torch.tensor([[2.0]], dtype=torch.float32),
                torch.tensor([[3.0]], dtype=torch.float32),
                1.0,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
        loha_patch = lora_optimizer.LoHaAdapter(
            set(),
            (
                torch.tensor([[2.0]], dtype=torch.float32),
                torch.tensor([[3.0]], dtype=torch.float32),
                1.0,
                torch.tensor([[4.0]], dtype=torch.float32),
                torch.tensor([[5.0]], dtype=torch.float32),
                None,
                None,
                None,
            ),
        )

        self.assertAlmostEqual(self.optimizer._expand_patch_to_diff(lokr_patch).item(), 6.0)
        self.assertAlmostEqual(self.optimizer._expand_patch_to_diff(loha_patch).item(), 120.0)

    def test_auto_strength_uses_exact_streamed_energy(self):
        active_loras = [
            {"name": "A", "strength": 1.0, "clip_strength": None},
            {"name": "B", "strength": 1.0, "clip_strength": None},
        ]
        branch_energy = {
            "model": {
                "norm_sq": [1.0, 1.0],
                "dot": {(0, 1): 1.0},
            },
            "clip": {
                "norm_sq": [0.0, 0.0],
                "dot": {(0, 1): 0.0},
            },
        }

        info = self.optimizer._compute_auto_strengths(active_loras, branch_energy)
        self.assertAlmostEqual(info["model_scale"], 0.5)
        self.assertAlmostEqual(info["model_strengths"][0], 0.5)
        self.assertAlmostEqual(info["model_strengths"][1], 0.5)

    def test_pair_metrics_capture_excess_conflict_and_subspace_overlap(self):
        diff_a = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        diff_b = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        basis_a = self.optimizer._compute_subspace_basis(diff_a, rank_hint=1)
        basis_b = self.optimizer._compute_subspace_basis(diff_b, rank_hint=1)

        metrics = self.optimizer._sample_pair_metrics(diff_a, diff_b, basis_a=basis_a, basis_b=basis_b)

        self.assertEqual(metrics["overlap"], 0)
        self.assertAlmostEqual(metrics["subspace_overlap"], 0.0, places=4)
        self.assertAlmostEqual(metrics["excess_conflict"], 0.0, places=4)

    def test_block_smoothing_populates_decision_metrics(self):
        prefix_stats = {
            "block_0.attn.q": {
                "n_loras": 2,
                "conflict_ratio": 0.10,
                "excess_conflict": 0.10,
                "avg_cos_sim": 0.20,
                "avg_subspace_overlap": 0.30,
                "magnitude_ratio": 1.0,
                "per_lora_norm_sq": {0: 1.0, 1: 1.0},
            },
            "block_0.attn.k": {
                "n_loras": 2,
                "conflict_ratio": 0.50,
                "excess_conflict": 0.50,
                "avg_cos_sim": 0.20,
                "avg_subspace_overlap": 0.30,
                "magnitude_ratio": 1.0,
                "per_lora_norm_sq": {0: 1.0, 1: 1.0},
            },
        }

        smoothed = self.optimizer._apply_block_smoothing(prefix_stats, strength=0.5)

        self.assertIn("decision_conflict", smoothed["block_0.attn.q"])
        self.assertGreater(smoothed["block_0.attn.q"]["decision_conflict"], 0.10)
        self.assertLess(smoothed["block_0.attn.q"]["decision_conflict"], 0.50)
        self.assertEqual(smoothed["block_0.attn.q"]["block_name"], smoothed["block_0.attn.k"]["block_name"])

    def test_auto_select_uses_excess_conflict_and_subspace(self):
        mode, _density, _sign, _reasoning = self.optimizer._auto_select_params(
            0.55, 1.0, avg_cos_sim=0.0,
            avg_excess_conflict=0.05, avg_subspace_overlap=0.10,
        )
        self.assertEqual(mode, "weighted_average")

        mode, _density, _sign, _reasoning = self.optimizer._auto_select_params(
            0.55, 1.0, avg_cos_sim=0.15,
            avg_excess_conflict=0.40, avg_subspace_overlap=0.85,
        )
        self.assertEqual(mode, "ties")

    def test_python_evaluator_spec_and_runner(self):
        builder = lora_optimizer.BuildAutoTunerPythonEvaluator()
        evaluator, = builder.build(
            module_path=os.path.join(tempfile.gettempdir(), "dummy_eval.py"),
            callable_name="evaluate_candidate",
            combine_mode="blend",
            weight=0.7,
            context_json='{"prompt":"test"}',
        )

        with open(evaluator["module_path"], "w") as f:
            f.write(
                "def evaluate_candidate(model=None, clip=None, lora_data=None, config=None, context=None, analysis_summary=None):\n"
                "    assert context['prompt'] == 'test'\n"
                "    return {'score': 0.75, 'details': {'ok': True}}\n"
            )

        result = lora_optimizer._run_autotuner_evaluator(
            evaluator,
            model=self.model,
            clip=None,
            lora_data={"model_patches": {}, "clip_patches": {}},
            config={"merge_mode": "weighted_average"},
            analysis_summary={"avg_conflict_ratio": 0.1},
        )

        self.assertAlmostEqual(result["score"], 0.75)
        self.assertEqual(result["details"]["ok"], True)

    def test_conflict_editor_preserves_key_filter_for_tuple_stacks(self):
        editor = lora_optimizer.LoRAConflictEditor()
        editor.loaded_loras["demo"] = {
            "alias_a.lora_up.weight": torch.tensor([[1.0]], dtype=torch.float32),
            "alias_a.lora_down.weight": torch.tensor([[1.0]], dtype=torch.float32),
        }

        enriched, _report, _strategy = editor.analyze_and_enrich(
            [("demo", 1.0, 1.0, "all", "shared_only")],
            "auto",
            conflict_mode_1="auto",
        )

        self.assertEqual(len(enriched[0]), 5)
        self.assertEqual(enriched[0][4], "shared_only")

    def test_save_merged_lora_uses_canonical_prefix(self):
        saver = lora_optimizer.SaveMergedLoRA()
        patch = lora_optimizer.LoRAAdapter(
            set(),
            (torch.tensor([[1.0]]), torch.tensor([[1.0]]), 1.0, None, None, None),
        )
        captured = {}

        with mock.patch.object(lora_optimizer, "save_file", side_effect=lambda state_dict, path, metadata=None: captured.update({"state_dict": state_dict, "path": path, "metadata": metadata})):
            save_path, = saver.save_lora(
                {
                    "model_patches": {"layer.weight": patch},
                    "clip_patches": {},
                    "key_map": {
                        "layer.weight": {
                            "canonical_prefix": "canonical_alias",
                            "aliases": ["alias_a", "canonical_alias"],
                        }
                    },
                    "output_strength": 1.0,
                    "clip_strength": 1.0,
                },
                tempfile.gettempdir(),
                "merged_test",
                save_rank=0,
                bake_strength=False,
            )

        self.assertTrue(save_path.endswith(".safetensors"))
        self.assertIn("canonical_alias.lora_up.weight", captured["state_dict"])

    def test_save_nodes_block_directory_traversal(self):
        merged_saver = lora_optimizer.SaveMergedLoRA()
        tuner_saver = lora_optimizer.SaveTunerData()
        patch = lora_optimizer.LoRAAdapter(
            set(),
            (torch.tensor([[1.0]]), torch.tensor([[1.0]]), 1.0, None, None, None),
        )

        with self.assertRaises(ValueError):
            merged_saver.save_lora(
                {
                    "model_patches": {"layer.weight": patch},
                    "clip_patches": {},
                    "key_map": {"layer.weight": "alias"},
                    "output_strength": 1.0,
                    "clip_strength": 1.0,
                },
                tempfile.gettempdir(),
                "../escape",
                save_rank=0,
                bake_strength=False,
            )

        with self.assertRaises(ValueError):
            tuner_saver.save_tuner_data({"top_n": []}, tempfile.gettempdir(), "../escape")

    def test_bridge_passthrough_returns_ui_payload(self):
        tuner_data = {
            "top_n": [{
                "config": {
                    "optimization_mode": "global",
                    "merge_mode": "ties",
                    "merge_refinement": "none",
                    "sparsification": "disabled",
                    "sparsification_density": 0.7,
                    "dare_dampening": 0.0,
                    "auto_strength": "enabled",
                },
                "score_final": 0.91,
            }],
            "decision_smoothing": 0.25,
            "auto_strength_floor": 0.95,
        }

        result = self.optimizer.execute_node(
            self.model, [], 1.0,
            tuner_data=tuner_data,
            settings_source="from_autotuner",
        )

        self.assertIsInstance(result, dict)
        self.assertIn("ui", result)
        applied = json.loads(result["ui"]["applied_settings"][0])
        self.assertEqual(applied["merge_mode"], "ties")
        self.assertEqual(applied["auto_strength_floor"], 0.95)

    def test_widget_order_keeps_upstream_workflow_compatibility(self):
        optimizer_keys = list(lora_optimizer.LoRAOptimizer.INPUT_TYPES()["optional"].keys())
        self.assertIn("settings_source", optimizer_keys)
        self.assertIn("decision_smoothing", optimizer_keys)
        self.assertLess(optimizer_keys.index("settings_source"), optimizer_keys.index("decision_smoothing"))

        autotuner_keys = list(lora_optimizer.LoRAAutoTuner.INPUT_TYPES()["optional"].keys())
        self.assertIn("output_mode", autotuner_keys)
        self.assertIn("decision_smoothing", autotuner_keys)
        self.assertLess(autotuner_keys.index("output_mode"), autotuner_keys.index("decision_smoothing"))

    def test_optimizer_exposes_tuner_data_output_and_compatibility_analyzer_node(self):
        self.assertEqual(
            lora_optimizer.LoRAOptimizer.RETURN_TYPES,
            ("MODEL", "CLIP", "STRING", "TUNER_DATA", "LORA_DATA"),
        )
        self.assertIn("LoRACompatibilityAnalyzer", lora_optimizer.NODE_CLASS_MAPPINGS)

    def test_autotune_resolve_tree_calls_auto_tune_for_subgroups(self):
        """_autotune_resolve_tree should call auto_tune for sub-groups with 2+ items."""
        from lora_optimizer import LoRAAutoTuner, _parse_merge_formula

        tuner = LoRAAutoTuner()
        tree = _parse_merge_formula("(1+2)+3", 3)

        # Build a minimal normalized stack with 3 fake LoRAs
        fake_lora_a = {"key_a": torch.randn(4, 4)}
        fake_lora_b = {"key_a": torch.randn(4, 4)}
        fake_lora_c = {"key_c": torch.randn(4, 4)}
        normalized_stack = [
            {"name": "lora_a", "lora": fake_lora_a, "strength": 1.0,
             "clip_strength": None, "metadata": {}},
            {"name": "lora_b", "lora": fake_lora_b, "strength": 1.0,
             "clip_strength": None, "metadata": {}},
            {"name": "lora_c", "lora": fake_lora_c, "strength": 1.0,
             "clip_strength": None, "metadata": {}},
        ]

        # Track auto_tune calls
        calls = []

        def mock_auto_tune(model, lora_stack, output_strength, **kwargs):
            calls.append({"n_loras": len(lora_stack), "names": [l["name"] for l in lora_stack]})
            # Return a minimal 6-tuple with virtual LoRA patches
            virtual_patches = {"key_a": ("diff", (torch.randn(4, 4),))}
            lora_data = {"model_patches": virtual_patches, "clip_patches": {}}
            return (model, None, "sub-report", "", None, lora_data)

        tuner.auto_tune = mock_auto_tune

        at_kwargs = {
            "clip_strength_multiplier": 1.0,
            "top_n": 3,
            "normalize_keys": "disabled",
            "scoring_svd": "disabled",
            "scoring_device": "cpu",
            "architecture_preset": "dit",
            "auto_strength_floor": -1.0,
            "decision_smoothing": 0.25,
            "smooth_slerp_gate": False,
            "vram_budget": 0.0,
            "scoring_speed": "turbo",
            "scoring_formula": "v2",
            "diff_cache_mode": "disabled",
            "diff_cache_ram_pct": 0.5,
        }

        resolved_stack, sub_reports = tuner._autotune_resolve_tree(
            tree, normalized_stack, None, None, **at_kwargs)

        # Should have called auto_tune once for the (1+2) sub-group
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["n_loras"], 2)
        self.assertEqual(calls[0]["names"], ["lora_a", "lora_b"])

        # Resolved stack should have 2 items: virtual LoRA + lora_c
        self.assertEqual(len(resolved_stack), 2)
        self.assertTrue(resolved_stack[0].get("_precomputed_diffs"))  # virtual
        self.assertEqual(resolved_stack[1]["name"], "lora_c")
        self.assertEqual(len(sub_reports), 1)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LoRASettingsNodeTests(unittest.TestCase):
    """Tests for LoRAMergeSettings, LoRAOptimizerSettings and LoRAAutoTunerSettings nodes."""

    def _build_defaults(self, inputs):
        """Extract default values from INPUT_TYPES required spec."""
        defaults = {}
        for key, spec in inputs["required"].items():
            if isinstance(spec[0], list):
                defaults[key] = spec[1].get("default", spec[0][0])
            elif spec[0] == "FLOAT":
                defaults[key] = spec[1]["default"]
            elif spec[0] == "INT":
                defaults[key] = spec[1]["default"]
            elif spec[0] == "BOOLEAN":
                defaults[key] = spec[1]["default"]
        return defaults

    def test_merge_settings_build_returns_dict(self):
        node = lora_optimizer.LoRAMergeSettings()
        inputs = lora_optimizer.LoRAMergeSettings.INPUT_TYPES()
        defaults = self._build_defaults(inputs)
        result = node.build_settings(**defaults)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        settings = result[0]
        expected_keys = {"normalize_keys", "architecture_preset",
                         "auto_strength_floor", "decision_smoothing",
                         "smooth_slerp_gate", "vram_budget", "cache_patches"}
        self.assertEqual(set(settings.keys()), expected_keys)
        self.assertEqual(settings["normalize_keys"], "enabled")
        self.assertEqual(settings["architecture_preset"], "auto")
        self.assertAlmostEqual(settings["auto_strength_floor"], -1.0)
        self.assertAlmostEqual(settings["decision_smoothing"], 0.25)
        self.assertFalse(settings["smooth_slerp_gate"])
        self.assertAlmostEqual(settings["vram_budget"], 0.0)
        self.assertEqual(settings["cache_patches"], "enabled")

    def test_optimizer_settings_build_returns_advanced_mode(self):
        node = lora_optimizer.LoRAOptimizerSettings()
        inputs = lora_optimizer.LoRAOptimizerSettings.INPUT_TYPES()
        defaults = self._build_defaults(inputs)
        result = node.build_settings(**defaults)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        settings = result[0]
        self.assertEqual(settings["mode"], "advanced")
        self.assertEqual(settings["auto_strength"], "enabled")
        self.assertEqual(settings["optimization_mode"], "per_prefix")
        self.assertEqual(settings["sparsification"], "disabled")
        self.assertAlmostEqual(settings["sparsification_density"], 0.7)
        self.assertEqual(settings["merge_strategy_override"], "")
        # Common settings should use defaults when merge_settings not connected
        self.assertEqual(settings["normalize_keys"], "enabled")
        self.assertEqual(settings["architecture_preset"], "auto")

    def test_optimizer_settings_with_strategy_override(self):
        node = lora_optimizer.LoRAOptimizerSettings()
        inputs = lora_optimizer.LoRAOptimizerSettings.INPUT_TYPES()
        defaults = self._build_defaults(inputs)
        defaults["merge_strategy_override"] = "slerp"
        result = node.build_settings(**defaults)
        self.assertEqual(result[0]["merge_strategy_override"], "slerp")

    def test_autotuner_settings_build_returns_autotuner_mode(self):
        node = lora_optimizer.LoRAAutoTunerSettings()
        inputs = lora_optimizer.LoRAAutoTunerSettings.INPUT_TYPES()
        defaults = self._build_defaults(inputs)
        result = node.build_settings(**defaults)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        settings = result[0]
        self.assertEqual(settings["mode"], "autotuner")
        self.assertEqual(settings["top_n"], 3)
        self.assertEqual(settings["scoring_speed"], "turbo")
        self.assertEqual(settings["output_mode"], "merge")
        self.assertFalse(settings["smooth_slerp_gate"])
        self.assertIsNone(settings["evaluator"])
        # Common settings should use defaults when merge_settings not connected
        self.assertEqual(settings["normalize_keys"], "enabled")
        self.assertEqual(settings["cache_patches"], "enabled")

    def test_autotuner_settings_with_evaluator(self):
        node = lora_optimizer.LoRAAutoTunerSettings()
        inputs = lora_optimizer.LoRAAutoTunerSettings.INPUT_TYPES()
        defaults = self._build_defaults(inputs)
        evaluator = {"type": "python", "code": "return 0.5"}
        defaults["evaluator"] = evaluator
        result = node.build_settings(**defaults)
        self.assertEqual(result[0]["evaluator"], evaluator)

    def test_mode_settings_merge_base_settings(self):
        """Both mode nodes correctly merge base settings from LoRAMergeSettings."""
        custom_ms = {
            "normalize_keys": "disabled",
            "architecture_preset": "dit",
            "auto_strength_floor": 0.5,
            "decision_smoothing": 0.8,
            "smooth_slerp_gate": True,
            "vram_budget": 0.3,
            "cache_patches": "disabled",
        }

        # Test LoRAOptimizerSettings with custom merge_settings
        opt_node = lora_optimizer.LoRAOptimizerSettings()
        opt_inputs = lora_optimizer.LoRAOptimizerSettings.INPUT_TYPES()
        opt_defaults = self._build_defaults(opt_inputs)
        opt_defaults["merge_settings"] = custom_ms
        opt_result = opt_node.build_settings(**opt_defaults)[0]
        for key, val in custom_ms.items():
            self.assertEqual(opt_result[key], val,
                             f"OptimizerSettings: {key} should be {val}, got {opt_result[key]}")

        # Test LoRAAutoTunerSettings with custom merge_settings
        at_node = lora_optimizer.LoRAAutoTunerSettings()
        at_inputs = lora_optimizer.LoRAAutoTunerSettings.INPUT_TYPES()
        at_defaults = self._build_defaults(at_inputs)
        at_defaults["merge_settings"] = custom_ms
        at_result = at_node.build_settings(**at_defaults)[0]
        for key, val in custom_ms.items():
            self.assertEqual(at_result[key], val,
                             f"AutoTunerSettings: {key} should be {val}, got {at_result[key]}")

    def test_settings_nodes_registered_in_mappings(self):
        self.assertIn("LoRAMergeSettings", lora_optimizer.NODE_CLASS_MAPPINGS)
        self.assertIn("LoRAOptimizerSettings", lora_optimizer.NODE_CLASS_MAPPINGS)
        self.assertIn("LoRAAutoTunerSettings", lora_optimizer.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS["LoRAMergeSettings"],
            "LoRA Merge Settings",
        )
        self.assertEqual(
            lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS["LoRAOptimizerSettings"],
            "LoRA Optimizer Settings",
        )
        self.assertEqual(
            lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS["LoRAAutoTunerSettings"],
            "LoRA AutoTuner Settings",
        )

    def test_settings_nodes_return_types(self):
        self.assertEqual(lora_optimizer.LoRAMergeSettings.RETURN_TYPES, ("MERGE_SETTINGS",))
        self.assertEqual(lora_optimizer.LoRAOptimizerSettings.RETURN_TYPES, ("OPTIMIZER_SETTINGS",))
        self.assertEqual(lora_optimizer.LoRAAutoTunerSettings.RETURN_TYPES, ("OPTIMIZER_SETTINGS",))

    def test_simple_node_accepts_settings_input(self):
        inputs = lora_optimizer.LoRAOptimizerSimple.INPUT_TYPES()
        self.assertIn("settings", inputs["optional"])
        self.assertEqual(inputs["optional"]["settings"][0], "OPTIMIZER_SETTINGS")

    def test_simple_is_changed_includes_settings_hash(self):
        settings = {"mode": "advanced", "auto_strength": "enabled"}
        result_with = lora_optimizer.LoRAOptimizerSimple.IS_CHANGED(
            None, None, 1.0, settings=settings)
        result_without = lora_optimizer.LoRAOptimizerSimple.IS_CHANGED(
            None, None, 1.0)
        self.assertIn("|settings=", result_with)
        self.assertNotIn("|settings=", result_without)

    def test_simple_is_changed_different_settings_produce_different_hashes(self):
        settings_a = {"mode": "advanced", "auto_strength": "enabled"}
        settings_b = {"mode": "advanced", "auto_strength": "disabled"}
        result_a = lora_optimizer.LoRAOptimizerSimple.IS_CHANGED(
            None, None, 1.0, settings=settings_a)
        result_b = lora_optimizer.LoRAOptimizerSimple.IS_CHANGED(
            None, None, 1.0, settings=settings_b)
        self.assertNotEqual(result_a, result_b)

    def test_optimizer_settings_defaults_match_simple_defaults(self):
        """Verify LoRAOptimizerSettings + LoRAMergeSettings defaults align with _SIMPLE_DEFAULTS."""
        opt_inputs = lora_optimizer.LoRAOptimizerSettings.INPUT_TYPES()
        merge_inputs = lora_optimizer.LoRAMergeSettings.INPUT_TYPES()
        simple = lora_optimizer.LoRAOptimizerSimple._SIMPLE_DEFAULTS
        # Keys on LoRAOptimizerSettings
        for key in ["auto_strength", "optimization_mode", "sparsification",
                     "merge_refinement", "strategy_set", "patch_compression",
                     "svd_device", "free_vram_between_passes"]:
            spec = opt_inputs["required"][key]
            if isinstance(spec[0], list):
                default = spec[1].get("default", spec[0][0])
            else:
                default = spec[1]["default"]
            self.assertEqual(default, simple[key],
                             f"Default mismatch for {key}: settings={default}, simple={simple[key]}")
        # Keys on LoRAMergeSettings
        for key in ["normalize_keys", "architecture_preset", "cache_patches",
                     "smooth_slerp_gate"]:
            spec = merge_inputs["required"][key]
            if isinstance(spec[0], list):
                default = spec[1].get("default", spec[0][0])
            else:
                default = spec[1]["default"]
            self.assertEqual(default, simple[key],
                             f"Default mismatch for {key}: merge_settings={default}, simple={simple[key]}")

    def test_merge_settings_defaults_match_input_types(self):
        """Verify _DEFAULTS dict stays in sync with INPUT_TYPES defaults."""
        inputs = lora_optimizer.LoRAMergeSettings.INPUT_TYPES()
        defaults_dict = lora_optimizer.LoRAMergeSettings._DEFAULTS
        for key, spec in inputs["required"].items():
            if isinstance(spec[0], list):
                input_default = spec[1].get("default", spec[0][0])
            else:
                input_default = spec[1]["default"]
            self.assertEqual(defaults_dict[key], input_default,
                             f"_DEFAULTS[{key}]={defaults_dict[key]} != INPUT_TYPES default={input_default}")
        self.assertEqual(set(defaults_dict.keys()), set(inputs["required"].keys()),
                         "_DEFAULTS keys don't match INPUT_TYPES keys")


    # --- Merge formula parser tests ---

    def test_parse_merge_formula_simple(self):
        """Simple flat formula parses to group of leaves."""
        tree = lora_optimizer._parse_merge_formula("1 + 2 + 3", 3)
        self.assertEqual(tree["type"], "group")
        self.assertEqual(len(tree["children"]), 3)
        for i, child in enumerate(tree["children"]):
            self.assertEqual(child["type"], "leaf")
            self.assertEqual(child["index"], i)

    def test_parse_merge_formula_nested(self):
        """Nested formula parses to tree with sub-group."""
        tree = lora_optimizer._parse_merge_formula("(1+2) + 3", 3)
        self.assertEqual(tree["type"], "group")
        self.assertEqual(len(tree["children"]), 2)
        sub = tree["children"][0]
        self.assertEqual(sub["type"], "group")
        self.assertEqual(len(sub["children"]), 2)
        leaf3 = tree["children"][1]
        self.assertEqual(leaf3["type"], "leaf")
        self.assertEqual(leaf3["index"], 2)

    def test_parse_merge_formula_weights(self):
        """Weights are parsed from :N.N suffix."""
        tree = lora_optimizer._parse_merge_formula("(1+2):0.6 + 3:0.4", 3)
        self.assertAlmostEqual(tree["children"][0]["weight"], 0.6)
        self.assertAlmostEqual(tree["children"][1]["weight"], 0.4)

    def test_parse_merge_formula_deep_nesting(self):
        """Deep nesting: ((1+2)+3) + 4."""
        tree = lora_optimizer._parse_merge_formula("((1+2)+3) + 4", 4)
        self.assertEqual(tree["type"], "group")
        self.assertEqual(len(tree["children"]), 2)
        inner = tree["children"][0]
        self.assertEqual(inner["type"], "group")
        self.assertEqual(len(inner["children"]), 2)
        innermost = inner["children"][0]
        self.assertEqual(innermost["type"], "group")
        self.assertEqual(len(innermost["children"]), 2)

    def test_parse_merge_formula_single_item(self):
        """Single item is valid."""
        tree = lora_optimizer._parse_merge_formula("1", 1)
        self.assertEqual(tree["type"], "leaf")
        self.assertEqual(tree["index"], 0)

    def test_parse_merge_formula_out_of_range(self):
        """Out of range index raises ValueError."""
        with self.assertRaises(ValueError):
            lora_optimizer._parse_merge_formula("1 + 5", 3)

    def test_parse_merge_formula_malformed(self):
        """Malformed formula raises ValueError."""
        with self.assertRaises(ValueError):
            lora_optimizer._parse_merge_formula("((1+2", 3)

    def test_parse_merge_formula_empty(self):
        """Empty/whitespace formula raises ValueError."""
        with self.assertRaises(ValueError):
            lora_optimizer._parse_merge_formula("", 3)
        with self.assertRaises(ValueError):
            lora_optimizer._parse_merge_formula("   ", 3)


    def test_merge_formula_node_registered(self):
        """LoRAMergeFormula is registered in NODE_CLASS_MAPPINGS."""
        self.assertIn("LoRAMergeFormula", lora_optimizer.NODE_CLASS_MAPPINGS)
        self.assertIn("LoRAMergeFormula", lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS)

    def test_merge_formula_node_passthrough(self):
        """LoRAMergeFormula passes stack through with formula metadata."""
        node = lora_optimizer.LoRAMergeFormula()
        stack = [{"name": "a", "lora": {}, "strength": 1.0}]
        result = node.apply_formula(stack, "(1)")
        self.assertIsInstance(result, tuple)
        output_stack = result[0]
        has_formula = any(isinstance(item, dict) and "_merge_formula" in item for item in output_stack)
        self.assertTrue(has_formula)

    def test_merge_formula_node_validates(self):
        """LoRAMergeFormula validates formula syntax — invalid returns stack without formula."""
        node = lora_optimizer.LoRAMergeFormula()
        stack = [{"name": "a", "lora": {}, "strength": 1.0}]
        result = node.apply_formula(stack, "(1+2)")  # only 1 LoRA — out of range
        output_stack = result[0]
        self.assertIsInstance(output_stack, list)
        # Should NOT have formula metadata since validation failed
        has_formula = any(isinstance(item, dict) and "_merge_formula" in item for item in output_stack)
        self.assertFalse(has_formula)


    # ------------------------------------------------------------------
    #  Merge formula tree executor + optimize_merge integration
    # ------------------------------------------------------------------

    def test_normalize_stack_filters_formula_metadata(self):
        """_normalize_stack filters out formula metadata entries."""
        stack = [
            {"name": "a", "lora": {}, "strength": 1.0},
            {"_merge_formula": "(1+2)"},
            {"name": "b", "lora": {}, "strength": 1.0},
        ]
        opt = lora_optimizer.LoRAOptimizer()
        result = opt._normalize_stack(stack)
        self.assertEqual(len(result), 2)
        names = [item["name"] for item in result]
        self.assertEqual(names, ["a", "b"])

    def test_optimize_merge_extracts_formula_metadata(self):
        """optimize_merge strips formula metadata before normalization."""
        stack = [
            {"name": "a", "lora": {"key1": ("diff", (torch.randn(4, 4),))}, "strength": 1.0},
            {"name": "b", "lora": {"key1": ("diff", (torch.randn(4, 4),))}, "strength": 1.0},
            {"_merge_formula": "1 + 2"},
        ]
        opt = lora_optimizer.LoRAOptimizer()
        # model=None → optimize_merge returns early with a report (no model to patch)
        result = opt.optimize_merge(None, stack, 1.0)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)  # 5-tuple

    def test_optimize_merge_invalid_formula_falls_back(self):
        """Invalid formula logs a warning and falls back to flat merge."""
        stack = [
            {"name": "a", "lora": {"key1": ("diff", (torch.randn(4, 4),))}, "strength": 1.0},
            {"name": "b", "lora": {"key1": ("diff", (torch.randn(4, 4),))}, "strength": 1.0},
            {"_merge_formula": "((1+2"},  # malformed
        ]
        opt = lora_optimizer.LoRAOptimizer()
        result = opt.optimize_merge(None, stack, 1.0)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)

    def test_model_to_virtual_lora_label(self):
        """_model_to_virtual_lora produces correct tree labels."""
        tree_node = {
            "type": "group",
            "weight": None,
            "children": [
                {"type": "leaf", "index": 0, "weight": None},
                {"type": "leaf", "index": 1, "weight": None},
            ],
        }
        virtual = lora_optimizer.LoRAOptimizer._model_to_virtual_lora(
            {}, {}, tree_node)
        self.assertEqual(virtual["name"], "(1+2)")
        self.assertEqual(virtual["strength"], 1.0)
        self.assertIsInstance(virtual["lora"], dict)
        self.assertTrue(virtual["_precomputed_diffs"])

    def test_resolve_tree_to_stack_flat(self):
        """_resolve_tree_to_stack resolves a flat group to the original items."""
        opt = lora_optimizer.LoRAOptimizer()
        normalized = [
            {"name": "a", "lora": {}, "strength": 0.8, "clip_strength": None,
             "conflict_mode": "all", "key_filter": "all", "metadata": {}},
            {"name": "b", "lora": {}, "strength": 0.6, "clip_strength": None,
             "conflict_mode": "all", "key_filter": "all", "metadata": {}},
        ]
        tree = {
            "type": "group",
            "weight": None,
            "children": [
                {"type": "leaf", "index": 0, "weight": None},
                {"type": "leaf", "index": 1, "weight": 0.5},
            ],
        }
        resolved, reports = opt._resolve_tree_to_stack(tree, normalized, None, None)
        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[0]["strength"], 0.8)  # unchanged
        self.assertEqual(resolved[1]["strength"], 0.5)  # overridden by weight
        self.assertEqual(reports, [])


if __name__ == "__main__":
    unittest.main()
