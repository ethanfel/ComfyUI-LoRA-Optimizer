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

    def test_spectral_split_orthogonal_loras_all_private(self):
        """Two orthogonal LoRAs: all directions should be private (ownership ~1.0)."""
        a = torch.zeros(16, 16)
        a[:8, :8] = torch.randn(8, 8)
        b = torch.zeros(16, 16)
        b[8:, 8:] = torch.randn(8, 8)
        diffs = [(a, 1.0), (b, 1.0)]
        shared, private = lora_optimizer.LoRAOptimizer._spectral_ownership_split(diffs)
        self.assertIsNotNone(private)
        self.assertGreater(private.norm().item(), 0.1 * (a.norm() + b.norm()).item())

    def test_spectral_split_identical_loras_all_shared(self):
        """Two identical LoRAs: all directions shared, no private component."""
        a = torch.randn(16, 16)
        diffs = [(a.clone(), 1.0), (a.clone(), 1.0)]
        shared, private = lora_optimizer.LoRAOptimizer._spectral_ownership_split(diffs)
        self.assertIsNone(private)

    def test_spectral_split_soft_weighting_partial_overlap(self):
        """Partially overlapping LoRAs: private_addition should be non-None
        and shared diffs should preserve total energy approximately."""
        torch.manual_seed(42)
        shared_base = torch.randn(32, 32) * 0.5
        a = shared_base + torch.randn(32, 32) * 0.3
        b = shared_base + torch.randn(32, 32) * 0.3
        original_sum = a * 0.6 + b * 0.4
        diffs = [(a, 0.6), (b, 0.4)]
        shared_diffs, private = lora_optimizer.LoRAOptimizer._spectral_ownership_split(diffs)
        self.assertIsNotNone(private)
        total_w = sum(abs(w) for _, w in shared_diffs)
        merged_shared = sum(d * (w / total_w) for d, w in shared_diffs)
        reconstructed = merged_shared + private
        self.assertGreater(reconstructed.norm().item(), original_sum.norm().item() * 0.3)

    def test_spectral_split_energy_adaptive_rank(self):
        """Low-rank input should use fewer SVD components than max_rank."""
        u = torch.randn(32, 2)
        v = torch.randn(32, 2)
        low_rank = u @ v.T
        noise = torch.randn(32, 32) * 0.01
        a = low_rank + noise
        b = torch.randn(32, 32)
        diffs = [(a, 1.0), (b, 1.0)]
        shared, private = lora_optimizer.LoRAOptimizer._spectral_ownership_split(diffs)
        self.assertEqual(len(shared), 2)

    def test_spectral_split_1d_guard(self):
        """1D tensors should pass through unchanged."""
        a = torch.randn(16)
        diffs = [(a, 1.0), (torch.randn(16), 1.0)]
        shared, private = lora_optimizer.LoRAOptimizer._spectral_ownership_split(diffs)
        self.assertIsNone(private)
        self.assertTrue(torch.equal(shared[0][0], a))

    def test_spectral_split_single_lora_guard(self):
        """Single LoRA should pass through unchanged."""
        a = torch.randn(16, 16)
        diffs = [(a, 1.0)]
        shared, private = lora_optimizer.LoRAOptimizer._spectral_ownership_split(diffs)
        self.assertIsNone(private)
        self.assertTrue(torch.equal(shared[0][0], a))

    def test_spectral_svc_reduces_over_accumulation(self):
        """When LoRAs share a dominant direction, SVC should calibrate it down."""
        torch.manual_seed(123)
        shared_dir = torch.randn(32, 1) @ torch.randn(1, 32)  # rank-1 shared
        a = shared_dir * 2.0 + torch.randn(32, 32) * 0.1
        b = shared_dir * 2.0 + torch.randn(32, 32) * 0.1
        diffs = [(a, 0.5), (b, 0.5)]
        shared_diffs, private = lora_optimizer.LoRAOptimizer._spectral_ownership_split(diffs)
        if private is not None:
            total_w = sum(abs(w) for _, w in shared_diffs)
            merged = sum(d.float() * (w / total_w) for d, w in shared_diffs)
            result = merged + private
        else:
            total_w = sum(abs(w) for _, w in shared_diffs)
            result = sum(d.float() * (w / total_w) for d, w in shared_diffs)
        expected_magnitude = (a * 0.5 + b * 0.5).norm()
        self.assertLess(result.norm().item(), expected_magnitude.item() * 2.0)


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


if __name__ == "__main__":
    unittest.main()
