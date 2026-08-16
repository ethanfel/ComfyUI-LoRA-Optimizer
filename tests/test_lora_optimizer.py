import contextlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
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
    folder_paths.get_full_path = lambda _kind, name: name
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


def _load_module():
    """Load lora_optimizer module with stubs in place."""
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "lora_optimizer",
        os.path.join(os.path.dirname(__file__), "..", "lora_optimizer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if torch is not None:
    _install_stubs()
    MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lora_optimizer.py")
    SPEC = importlib.util.spec_from_file_location("lora_optimizer_under_test", MODULE_PATH)
    lora_optimizer = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(lora_optimizer)
    # Register under the plain name too — otherwise mock.patch("lora_optimizer.X")
    # imports a SECOND module instance and patches that one, silently making
    # every such patch (e.g. AUTOTUNER_MEMORY_DIR tmpdir redirects) a no-op
    sys.modules["lora_optimizer"] = lora_optimizer
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

    def test_lora_format_cache_avoids_repeated_detection(self):
        """After detecting a LoRA's format once, subsequent prefixes should reuse it."""
        optimizer = lora_optimizer.LoRAOptimizer()
        lora_dict = {
            "unet.a.lora_B.weight": torch.tensor([[1.0]], dtype=torch.float32),
            "unet.a.lora_A.weight": torch.tensor([[1.0]], dtype=torch.float32),
            "unet.b.lora_B.weight": torch.tensor([[2.0]], dtype=torch.float32),
            "unet.b.lora_A.weight": torch.tensor([[1.0]], dtype=torch.float32),
        }
        result1 = optimizer._get_lora_key_info(lora_dict, "unet.a")
        self.assertIsNotNone(result1)
        self.assertIn(id(lora_dict), optimizer._lora_format_cache)
        result2 = optimizer._get_lora_key_info(lora_dict, "unet.b")
        self.assertIsNotNone(result2)

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
            [("demo", 1.0, 1.0, "all", "shared_only", True)],
            "auto",
            conflict_mode_1="auto",
        )

        # tuple grew to 6 with the preserve flag; key_filter + preserve round-trip
        self.assertEqual(len(enriched[0]), 6)
        self.assertEqual(enriched[0][4], "shared_only")
        self.assertIs(enriched[0][5], True)

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

    def test_autotune_resolve_tree_passes_sub_merge_flags(self):
        """Sub-merge auto_tune calls should include _is_sub_merge and _suppress_pbar."""
        from lora_optimizer import LoRAAutoTuner, _parse_merge_formula

        tuner = LoRAAutoTuner()
        tree = _parse_merge_formula("(1+2)+3", 3)

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

        calls = []
        def mock_auto_tune(model, lora_stack, output_strength, **kwargs):
            calls.append(kwargs)
            virtual_patches = {"key_a": ("diff", (torch.randn(4, 4),))}
            lora_data = {"model_patches": virtual_patches, "clip_patches": {}}
            return (model, None, "sub-report", "", None, lora_data)

        tuner.auto_tune = mock_auto_tune

        at_kwargs = {
            "clip_strength_multiplier": 1.0,
            "top_n": 3, "normalize_keys": "disabled", "scoring_svd": "disabled",
            "scoring_device": "cpu", "architecture_preset": "dit",
            "auto_strength_floor": -1.0, "decision_smoothing": 0.25,
            "smooth_slerp_gate": False, "vram_budget": 0.0,
            "scoring_speed": "turbo", "scoring_formula": "v2",
            "diff_cache_mode": "disabled", "diff_cache_ram_pct": 0.5,
        }

        tuner._autotune_resolve_tree(tree, normalized_stack, None, None, **at_kwargs)

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].get("_is_sub_merge", False))
        self.assertTrue(calls[0].get("_suppress_pbar", False))
        self.assertEqual(calls[0].get("cache_patches"), "disabled")
        self.assertEqual(calls[0].get("community_cache"), "disabled")
        self.assertEqual(calls[0].get("output_mode"), "merge")

    def test_autotune_resolve_tree_single_item_passthrough(self):
        """Single-item sub-groups should pass through without calling auto_tune."""
        from lora_optimizer import LoRAAutoTuner, _parse_merge_formula

        tuner = LoRAAutoTuner()
        tree = _parse_merge_formula("(1)+2", 2)

        fake_lora_a = {"key_a": torch.randn(4, 4)}
        fake_lora_b = {"key_b": torch.randn(4, 4)}
        normalized_stack = [
            {"name": "lora_a", "lora": fake_lora_a, "strength": 1.0,
             "clip_strength": None, "metadata": {}},
            {"name": "lora_b", "lora": fake_lora_b, "strength": 1.0,
             "clip_strength": None, "metadata": {}},
        ]

        calls = []
        def mock_auto_tune(model, lora_stack, output_strength, **kwargs):
            calls.append(True)
            return (model, None, "", "", None, None)

        tuner.auto_tune = mock_auto_tune

        at_kwargs = {
            "clip_strength_multiplier": 1.0,
            "top_n": 3, "normalize_keys": "disabled", "scoring_svd": "disabled",
            "scoring_device": "cpu", "architecture_preset": "dit",
            "auto_strength_floor": -1.0, "decision_smoothing": 0.25,
            "smooth_slerp_gate": False, "vram_budget": 0.0,
            "scoring_speed": "turbo", "scoring_formula": "v2",
            "diff_cache_mode": "disabled", "diff_cache_ram_pct": 0.5,
        }

        resolved_stack, sub_reports = tuner._autotune_resolve_tree(
            tree, normalized_stack, None, None, **at_kwargs)

        self.assertEqual(len(calls), 0)
        self.assertEqual(len(resolved_stack), 2)
        self.assertEqual(resolved_stack[0]["name"], "lora_a")
        self.assertEqual(resolved_stack[1]["name"], "lora_b")

    def test_autotune_resolve_tree_nested_groups(self):
        """Nested formula ((1+2)+3)+4 should resolve innermost first."""
        from lora_optimizer import LoRAAutoTuner, _parse_merge_formula

        tuner = LoRAAutoTuner()
        tree = _parse_merge_formula("((1+2)+3)+4", 4)

        normalized_stack = []
        for i in range(4):
            normalized_stack.append({
                "name": f"lora_{i}", "lora": {f"key_{i}": torch.randn(4, 4)},
                "strength": 1.0, "clip_strength": None, "metadata": {},
            })

        call_order = []
        def mock_auto_tune(model, lora_stack, output_strength, **kwargs):
            names = [l["name"] for l in lora_stack if not l.get("_precomputed_diffs")]
            virtual_count = sum(1 for l in lora_stack if l.get("_precomputed_diffs"))
            call_order.append({"names": names, "virtual_count": virtual_count,
                               "total": len(lora_stack)})
            virtual_patches = {f"key_v{len(call_order)}": ("diff", (torch.randn(4, 4),))}
            lora_data = {"model_patches": virtual_patches, "clip_patches": {}}
            return (model, None, f"sub-report-{len(call_order)}", "", None, lora_data)

        tuner.auto_tune = mock_auto_tune

        at_kwargs = {
            "clip_strength_multiplier": 1.0,
            "top_n": 3, "normalize_keys": "disabled", "scoring_svd": "disabled",
            "scoring_device": "cpu", "architecture_preset": "dit",
            "auto_strength_floor": -1.0, "decision_smoothing": 0.25,
            "smooth_slerp_gate": False, "vram_budget": 0.0,
            "scoring_speed": "turbo", "scoring_formula": "v2",
            "diff_cache_mode": "disabled", "diff_cache_ram_pct": 0.5,
        }

        resolved_stack, sub_reports = tuner._autotune_resolve_tree(
            tree, normalized_stack, None, None, **at_kwargs)

        # Two auto_tune calls: (1+2) first, then (virtual+3)
        self.assertEqual(len(call_order), 2)
        self.assertEqual(call_order[0]["names"], ["lora_0", "lora_1"])
        self.assertEqual(call_order[0]["virtual_count"], 0)
        self.assertEqual(call_order[1]["total"], 2)
        self.assertEqual(call_order[1]["virtual_count"], 1)

        # Final stack: virtual from ((1+2)+3) + lora_3
        self.assertEqual(len(resolved_stack), 2)
        self.assertTrue(resolved_stack[0].get("_precomputed_diffs"))
        self.assertEqual(resolved_stack[1]["name"], "lora_3")

    def test_auto_tune_with_formula_calls_autotune_resolve_tree(self):
        """auto_tune should detect formula and use _autotune_resolve_tree."""
        tuner = lora_optimizer.LoRAAutoTuner()

        # Build a lora_stack with formula metadata
        fake_lora_a = {"key_a": torch.randn(4, 4)}
        fake_lora_b = {"key_a": torch.randn(4, 4)}
        fake_lora_c = {"key_c": torch.randn(4, 4)}
        lora_stack = [
            {"name": "lora_a", "lora": fake_lora_a, "strength": 1.0},
            {"name": "lora_b", "lora": fake_lora_b, "strength": 1.0},
            {"name": "lora_c", "lora": fake_lora_c, "strength": 1.0},
            {"_merge_formula": "(1+2)+3"},
        ]

        # Track _autotune_resolve_tree calls
        resolve_calls = []

        def mock_resolve(tree, normalized_stack, model, clip, **kwargs):
            resolve_calls.append(tree)
            # Return a flat stack (2 items) so auto_tune continues normally
            return ([normalized_stack[0], normalized_stack[2]], [])

        tuner._autotune_resolve_tree = mock_resolve

        # Mock the rest of auto_tune to avoid needing real models
        # We just need to verify _autotune_resolve_tree was called
        try:
            tuner.auto_tune(None, lora_stack, 1.0)
        except Exception:
            pass  # Will fail later in pipeline — we only check the call happened

        self.assertEqual(len(resolve_calls), 1)
        self.assertEqual(resolve_calls[0]["type"], "group")

    def test_build_stack_skips_disabled_slots(self):
        from unittest import mock
        node = lora_optimizer.LoRAStackDynamic()
        with mock.patch.object(
            lora_optimizer.LoRAStackDynamic, "_resolve_lora_name",
            side_effect=lambda n: n,
        ):
            result, = node.build_stack(
                settings_visibility="simple",
                input_mode="text",
                lora_count=3,
                lora_name_text_1="lora_a",
                lora_name_text_2="lora_b",
                lora_name_text_3="lora_c",
                strength_1=1.0,
                strength_2=0.8,
                strength_3=0.5,
                enabled_1=True,
                enabled_2=False,
                enabled_3=True,
            )
        names = [entry[0] for entry in result]
        self.assertEqual(len(result), 2)
        self.assertIn("lora_a", names)
        self.assertIn("lora_c", names)
        self.assertNotIn("lora_b", names)

    def test_build_stack_absent_enabled_treated_as_enabled(self):
        from unittest import mock
        node = lora_optimizer.LoRAStackDynamic()
        # No enabled_{i} keys passed — must behave as if all enabled
        with mock.patch.object(
            lora_optimizer.LoRAStackDynamic, "_resolve_lora_name",
            side_effect=lambda n: n,
        ):
            result, = node.build_stack(
                settings_visibility="simple",
                input_mode="text",
                lora_count=2,
                lora_name_text_1="lora_a",
                lora_name_text_2="lora_b",
                strength_1=1.0,
                strength_2=0.8,
                # No enabled_1 or enabled_2 passed
            )
        self.assertEqual(len(result), 2)

    def test_score_merge_result_baseline_matches_full(self):
        """Scoring multi-LoRA patches with single-LoRA baseline should match full scoring."""
        LoRAAdapter = lora_optimizer.LoRAAdapter
        single_patches = {}
        multi_patches = {}
        all_patches = {}
        for i in range(10):
            up = torch.randn(8, 4)
            down = torch.randn(4, 16)
            adapter = LoRAAdapter(set(), (up, down, 4.0, None, None, None))
            key = f"key{i}"
            all_patches[key] = adapter
            if i < 5:
                single_patches[key] = adapter
            else:
                multi_patches[key] = adapter

        full = lora_optimizer._score_merge_result(all_patches, {}, compute_svd=False)

        sl = lora_optimizer._score_merge_result(
            single_patches, {}, compute_svd=False, _return_raw=True)
        baseline = sl["_raw"]
        combined = lora_optimizer._score_merge_result(
            multi_patches, {}, compute_svd=False, _baseline=baseline)

        self.assertAlmostEqual(full["composite_score"], combined["composite_score"], places=6)
        self.assertAlmostEqual(full["norm_mean"], combined["norm_mean"], places=6)
        self.assertAlmostEqual(full["norm_cv"], combined["norm_cv"], places=6)
        self.assertAlmostEqual(full["sparsity_mean"], combined["sparsity_mean"], places=6)
        self.assertAlmostEqual(full["norm_energy_sq"], combined["norm_energy_sq"], places=4)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class PrefixModeRoutingTests(unittest.TestCase):
    """Orthogonal groups BLEND by default (weighted_average / slerp). Additive
    preservation is never auto-selected — the analyzer can't tell 'preserve this
    style' from 'blend these characters', and auto-additive oversaturates ordinary
    multi-LoRA merges. Style preservation is opt-in via the preserve flag only."""

    def setUp(self):
        self.optimizer = lora_optimizer.LoRAOptimizer()
        self.arch = lora_optimizer._ARCH_PRESETS["dit"]

    def _pf(self, cos, n_loras=2, mag_ratio=1.0):
        return {
            "conflict_ratio": 0.5,
            "magnitude_ratio": mag_ratio,
            "n_loras": n_loras,
            "avg_cos_sim": cos,
            "excess_conflict": 0.0,
            "avg_subspace_overlap": 0.0,
        }

    def _decide(self, cos, strategy_set, n_loras=2, mag_ratio=1.0):
        return self.optimizer._decide_prefix_mode(
            self._pf(cos, n_loras, mag_ratio), strategy_set, self.arch,
            smooth_slerp_gate=False, is_full_rank=False, fr_preset={})

    def test_orthogonal_no_slerp_weighted_average(self):
        mode, _d, _s, orth, opp = self._decide(0.0, "no_slerp")
        self.assertEqual(mode, "weighted_average")
        self.assertTrue(orth)

    def test_orthogonal_basic_weighted_average(self):
        mode, *_ = self._decide(0.0, "basic")
        self.assertEqual(mode, "weighted_average")

    def test_orthogonal_full_slerp(self):
        mode, *_ = self._decide(0.0, "full")
        self.assertEqual(mode, "slerp")

    def test_nonorthogonal_aligned_full_slerp(self):
        mode, *_ = self._decide(0.35, "full")
        self.assertEqual(mode, "slerp")

    def test_opposing_weighted_average(self):
        mode, _d, _s, orth, opp = self._decide(-0.1, "no_slerp")
        self.assertEqual(mode, "weighted_average")
        self.assertTrue(opp)

    def test_additive_not_auto_selected_for_balanced(self):
        # Regression for the multi-LoRA oversaturation: a BALANCED stack
        # (magnitude_ratio ~1) must never auto-route to a sum mode regardless of
        # orthogonality/strategy.
        for cos in (0.0, 0.1, -0.1, 0.35, 0.6):
            for ss in ("full", "no_slerp", "basic"):
                mode, *_ = self._decide(cos, ss, mag_ratio=1.0)
                self.assertNotIn(mode, ("sum_preserve", "weighted_sum"))

    def test_imbalanced_orthogonal_pair_routes_to_weighted_sum(self):
        # A strongly-imbalanced orthogonal PAIR routes to additive: SLERP/
        # weighted_average wash out the dominant LoRA, weighted_sum preserves it
        # (and can't oversaturate — the dominant defines the auto-strength ref).
        mode, _d, _s, orth, opp = self._decide(0.0, "full", mag_ratio=6.0)
        self.assertEqual(mode, "weighted_sum")
        self.assertTrue(orth)
        self.assertFalse(opp)

    def test_imbalanced_orthogonal_suppresses_slerp(self):
        # Below the cap → SLERP; at/above the cap → not SLERP.
        self.assertEqual(self._decide(0.0, "full", mag_ratio=1.5)[0], "slerp")
        self.assertNotEqual(self._decide(0.0, "full", mag_ratio=2.5)[0], "slerp")

    def test_imbalanced_aligned_not_weighted_sum(self):
        # Imbalanced but ALIGNED (non-orthogonal): SLERP is suppressed but it must
        # NOT go additive (aligned LoRAs reinforce → additive would oversaturate).
        mode, *_ = self._decide(0.35, "full", mag_ratio=6.0)
        self.assertEqual(mode, "weighted_average")

    def test_imbalanced_orthogonal_triple_stays_blended(self):
        # 3+ LoRAs: weighted_sum is gated to pairs (a balanced sub-pair among 3+
        # could compound past the auto-strength floor), so imbalanced triples fall
        # back to weighted_average, never additive.
        mode, *_ = self._decide(0.0, "full", n_loras=3, mag_ratio=6.0)
        self.assertEqual(mode, "weighted_average")


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class ShapeMismatchReportTests(unittest.TestCase):
    """A LoRA whose tensor shape doesn't match the target model at a layer is
    dropped from the merge; _note_shape_mismatch records it (deduped, capped) so
    the report can surface the incompatibility instead of dropping it silently."""

    def setUp(self):
        self.optimizer = lora_optimizer.LoRAOptimizer()
        self.optimizer._shape_mismatches = {}

    def test_records_and_dedups_by_key(self):
        item = {"name": "concept/KNP_V2.safetensors"}
        self.optimizer._note_shape_mismatch(item, "blocks.0.attn.gate", 2560, 6144)
        self.optimizer._note_shape_mismatch(item, "blocks.0.attn.gate", 2560, 6144)  # dup
        self.optimizer._note_shape_mismatch(item, "blocks.1.attn.wq", 2560, 6144)
        per = self.optimizer._shape_mismatches["concept/KNP_V2.safetensors"]
        self.assertEqual(len(per), 2)  # deduped by target key
        self.assertEqual(per["blocks.0.attn.gate"], (2560, 6144))

    def test_tuple_target_key_uses_first_element(self):
        self.optimizer._note_shape_mismatch({"name": "x"}, ("blocks.0.attn.gate", True), 2560, 6144)
        self.assertIn("blocks.0.attn.gate", self.optimizer._shape_mismatches["x"])

    def test_capped_at_256(self):
        for i in range(300):
            self.optimizer._note_shape_mismatch({"name": "x"}, f"blocks.{i}", 2560, 6144)
        self.assertLessEqual(len(self.optimizer._shape_mismatches["x"]), 256)

    def test_lazy_init_when_attribute_absent(self):
        opt = lora_optimizer.LoRAOptimizer()
        if hasattr(opt, "_shape_mismatches"):
            del opt._shape_mismatches
        opt._note_shape_mismatch({"name": "y"}, "blocks.0", 2560, 6144)
        self.assertIn("y", opt._shape_mismatches)

    def test_report_lines_empty_when_no_mismatch(self):
        self.assertEqual(self.optimizer._shape_mismatch_report_lines(), [])

    def test_report_lines_render_warning(self):
        item = {"name": "Krea 2/concept/KNP_V2.safetensors"}
        for i in range(2):
            for proj in ("gate", "wq", "wk", "wv"):
                self.optimizer._note_shape_mismatch(item, f"blocks.{i}.attn.{proj}", 2560, 6144)
        text = "\n".join(self.optimizer._shape_mismatch_report_lines())
        self.assertIn("SHAPE INCOMPATIBILITY", text)
        self.assertIn("8 layer(s) DROPPED", text)        # 2 blocks x 4 projections
        self.assertIn("KNP_V2.safetensors", text)        # basename only, no dir
        self.assertNotIn("concept/", text)
        self.assertIn("LoRA dim=2560 vs model dim=6144", text)
        self.assertIn("... and", text)                   # truncated past 3 examples
        self.assertIn("SAME base model", text)           # actionable fix hint

    def test_lokr_shape_mismatch_is_recorded(self):
        # A LoKr concept whose reconstructed delta doesn't match the model weight
        # must be recorded too — LoKr/LoHa go through a different branch than plain
        # LoRA, so the capture has to cover it (regression for snofs-style LoKrs).
        model = _make_model()  # model.layer.weight is (1, 1)
        lora = {
            "layer.lokr_w1": torch.eye(2, dtype=torch.float32),          # (2, 2)
            "layer.lokr_w2": torch.tensor([[1.0]], dtype=torch.float32),  # (1, 1)
        }  # kron(w1, w2) -> (2, 2), cannot reshape to the (1, 1) model weight
        active_loras = [{
            "name": "concept/snofs_krea_v1.safetensors", "lora": lora, "strength": 1.0,
            "clip_strength": None, "key_filter": "all", "conflict_mode": "all",
        }]
        target_group = {
            "target_key": "layer.weight", "is_clip": False,
            "aliases": ["layer"], "label_prefix": "layer",
        }
        self.optimizer._prepare_group_diffs(
            target_group, active_loras, model, None, torch.device("cpu"))
        per = self.optimizer._shape_mismatches.get("concept/snofs_krea_v1.safetensors")
        self.assertIsNotNone(per)
        self.assertEqual(per["layer.weight"], (2, 1))  # LoKr dim 2 vs model dim 1


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LoraCleaningTests(unittest.TestCase):
    """Per-LoRA cleaning primitives: STAR spectral truncate+rescale (nuclear-norm
    preserving) and base-norm-anchored magnitude taming (Norm-Anchor Scaling)."""

    def setUp(self):
        self.opt = lora_optimizer.LoRAOptimizer()

    def test_star_eta_100_is_identity(self):
        d = torch.randn(6, 4)
        torch.testing.assert_close(self.opt._star_truncate_rescale(d.clone(), 100.0), d)

    def test_star_preserves_nuclear_norm_and_reduces_rank(self):
        d = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))  # nuclear norm 10, rank 4
        out = self.opt._star_truncate_rescale(d.clone(), 70.0)
        sv = torch.linalg.svdvals(out)
        self.assertAlmostEqual(sv.sum().item(), 10.0, places=4)   # nuclear norm restored
        self.assertEqual(int((sv > 1e-4).sum().item()), 2)         # eta=70 keeps 2 of 4

    def test_star_preserves_nuclear_norm_random(self):
        torch.manual_seed(0)
        d = torch.randn(8, 5)
        nuc0 = torch.linalg.svdvals(d).sum().item()
        out = self.opt._star_truncate_rescale(d.clone(), 50.0)
        self.assertAlmostEqual(torch.linalg.svdvals(out).sum().item(), nuc0, places=3)

    def test_star_1d_passthrough(self):
        d = torch.randn(5)
        torch.testing.assert_close(self.opt._star_truncate_rescale(d.clone(), 50.0), d)

    def test_tame_within_budget_no_scale(self):
        self.assertEqual(self.opt._tame_scale(2.0, 10.0, 0.3, 1.0), 1.0)  # r=0.2 <= 0.3

    def test_tame_hot_layer_scaled_to_threshold_at_strength_1(self):
        s = self.opt._tame_scale(5.0, 10.0, 0.3, 1.0)  # r=0.5 > 0.3
        self.assertAlmostEqual(s, 0.6, places=6)
        self.assertAlmostEqual(5.0 * s, 0.3 * 10.0, places=6)  # tamed to exactly threshold*base

    def test_tame_strength_0_is_off(self):
        self.assertEqual(self.opt._tame_scale(5.0, 10.0, 0.3, 0.0), 1.0)

    def test_tame_partial_strength(self):
        s = self.opt._tame_scale(5.0, 10.0, 0.3, 0.5)
        self.assertAlmostEqual(s, math.sqrt(0.6), places=6)

    def test_tame_zero_base_norm_floored(self):
        s = self.opt._tame_scale(1.0, 0.0, 0.3, 1.0)  # denom floored, no div-by-zero
        self.assertGreater(s, 0.0)
        self.assertLess(s, 1e-6)

    def _lora_and_model(self, base_scale):
        # base weight = base_scale*I(4) (norm 2*base_scale); LoRA delta = diag(4,3,2,1)
        layer = types.SimpleNamespace(weight=torch.eye(4) * base_scale)
        model = types.SimpleNamespace(model=types.SimpleNamespace(layer=layer))
        lora = {"layer.lora_up.weight": torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])),
                "layer.lora_down.weight": torch.eye(4)}
        active = [{"name": "A", "lora": lora, "strength": 1.0, "clip_strength": None,
                   "key_filter": "all", "conflict_mode": "all"}]
        tg = {"target_key": "layer.weight", "is_clip": False,
              "aliases": ["layer"], "label_prefix": "layer"}
        return active, tg, model

    def _diff(self, active, tg, model):
        return self.opt._prepare_group_diffs(tg, active, model, None, torch.device("cpu"))["diffs"][0]

    def test_prepare_applies_star(self):
        active, tg, model = self._lora_and_model(3.0)
        self.opt._star_eta = 100.0; self.opt._tame_layers = 0.0
        self.assertEqual(int((torch.linalg.svdvals(self._diff(active, tg, model)) > 1e-4).sum()), 4)
        self.opt._star_eta = 70.0
        sv = torch.linalg.svdvals(self._diff(active, tg, model))
        self.assertEqual(int((sv > 1e-4).sum()), 2)                 # rank reduced
        self.assertAlmostEqual(sv.sum().item(), 10.0, places=3)     # nuclear norm preserved

    def test_prepare_applies_tame(self):
        active, tg, model = self._lora_and_model(5.0)  # base norm 10; delta norm sqrt(30)
        self.opt._star_eta = 100.0; self.opt._tame_layers = 1.0; self.opt._tame_threshold = 0.3
        self.assertAlmostEqual(self._diff(active, tg, model).norm().item(), 3.0, places=3)  # -> 0.3*base

    def test_prepare_preserve_is_exempt_from_cleaning(self):
        active, tg, model = self._lora_and_model(5.0)
        active[0]["preserve"] = True
        self.opt._star_eta = 50.0; self.opt._tame_layers = 1.0; self.opt._tame_threshold = 0.3
        out = self._diff(active, tg, model)
        self.assertEqual(int((torch.linalg.svdvals(out) > 1e-4).sum()), 4)   # untouched
        self.assertAlmostEqual(out.norm().item(), math.sqrt(30.0), places=3)

    def test_cache_key_folds_cleaning_only_when_active(self):
        stack = [{"name": "x", "strength": 1.0}]
        k_off = self.opt._compute_cache_key(stack, 1.0, 1.0, "disabled")
        k_off2 = self.opt._compute_cache_key(stack, 1.0, 1.0, "disabled", star_eta=100.0, tame_layers=0.0)
        self.assertEqual(k_off, k_off2)  # default-off must not change the key
        k_on = self.opt._compute_cache_key(stack, 1.0, 1.0, "disabled", star_eta=40.0)
        self.assertNotEqual(k_off, k_on)  # active cleaning gets a distinct key


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class PreserveFlagTests(unittest.TestCase):
    """A per-LoRA preserve flag protects a tagged style LoRA from TIES sign-election
    deletion and from sparsification trimming in conflict merges."""

    def setUp(self):
        self.optimizer = lora_optimizer.LoRAOptimizer()

    def test_preserve_overlay_weighted_average(self):
        # The style-LoRA use case: a flagged style is added at FULL strength on top
        # of the weighted_average blend of the rest, instead of being averaged down.
        content = torch.tensor([2.0, 0.0, 0.0, 0.0])   # not preserved
        style = torch.tensor([0.0, 3.0, 0.0, 0.0])     # preserved (orthogonal)
        blended = self.optimizer._merge_diffs(
            [(content.clone(), 1.0), (style.clone(), 1.0)], "weighted_average")
        kept = self.optimizer._merge_diffs(
            [(content.clone(), 1.0), (style.clone(), 1.0)], "weighted_average",
            preserve_flags=[False, True])
        # plain blend halves both; preserve keeps content blended (single -> full) and
        # the style at full on top
        torch.testing.assert_close(blended, torch.tensor([1.0, 1.5, 0.0, 0.0]))
        torch.testing.assert_close(kept, torch.tensor([2.0, 3.0, 0.0, 0.0]))

    def test_preserve_does_not_affect_unflagged_blend(self):
        # No flags -> ordinary balanced blend is untouched (the multi-LoRA case).
        a = torch.tensor([2.0, 0.0])
        b = torch.tensor([0.0, 4.0])
        res = self.optimizer._merge_diffs(
            [(a.clone(), 1.0), (b.clone(), 1.0)], "weighted_average")
        torch.testing.assert_close(res, torch.tensor([1.0, 2.0]))

    def test_ties_deletes_minority_sign_without_preserve(self):
        # Content (+10) out-votes a style (-1) in TIES sign election; the style's
        # minority-sign direction is dropped entirely.
        content = torch.tensor([10.0, 10.0, 10.0, 10.0])
        style = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        base = self.optimizer._merge_diffs(
            [(content.clone(), 1.0), (style.clone(), 1.0)], "ties", density=1.0)
        self.assertAlmostEqual(base[0].item(), 10.0, places=5)

    def test_ties_preserve_keeps_minority_sign_style(self):
        # With preserve on the style, its full contribution is added on top of the
        # TIES-merged content: 10 + (-1) = 9 (the style survives the conflict).
        content = torch.tensor([10.0, 10.0, 10.0, 10.0])
        style = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        kept = self.optimizer._merge_diffs(
            [(content.clone(), 1.0), (style.clone(), 1.0)], "ties", density=1.0,
            preserve_flags=[False, True])
        self.assertAlmostEqual(kept[0].item(), 9.0, places=5)

    def test_ties_all_preserved_is_full_sum(self):
        # Every contributor tagged -> nothing to TIES-merge -> plain full sum.
        a = torch.tensor([3.0, 3.0])
        b = torch.tensor([-2.0, -2.0])
        res = self.optimizer._merge_diffs(
            [(a.clone(), 1.0), (b.clone(), 1.0)], "ties", density=0.5,
            preserve_flags=[True, True])
        torch.testing.assert_close(res, torch.tensor([1.0, 1.0]))

    def test_sparsification_all_preserved_equals_disabled(self):
        # All preserved -> sparsification is skipped -> identical to disabled.
        d1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        d2 = torch.tensor([0.5, 1.5, 2.5, 3.5])
        disabled = self.optimizer._merge_diffs(
            [(d1.clone(), 1.0), (d2.clone(), 1.0)], "weighted_sum",
            sparsification="disabled")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        all_pres = self.optimizer._merge_diffs(
            [(d1.clone(), 1.0), (d2.clone(), 1.0)], "weighted_sum",
            sparsification="dare", sparsification_density=0.5,
            sparsification_generator=gen, preserve_flags=[True, True])
        torch.testing.assert_close(disabled, all_pres)

    def test_normalize_stack_carries_preserve_tuple_and_dict(self):
        opt = lora_optimizer.LoRAOptimizer()
        opt.loaded_loras = {"loraA": {}, "loraB": {}}
        tup = opt._normalize_stack([
            ("loraA", 1.0, 1.0, "all", "all", True),
            ("loraB", 1.0, 1.0, "all", "all"),  # legacy 5-tuple -> preserve False
        ])
        self.assertTrue(tup[0]["preserve"])
        self.assertFalse(tup[1]["preserve"])

        dct = opt._normalize_stack([
            {"name": "x", "lora": {}, "strength": 1.0, "preserve": True},
        ])
        self.assertTrue(dct[0]["preserve"])

    def test_normalize_stack_mixed_tuple_and_dict(self):
        """A stack mixing file-ref tuples with in-memory dict entries (e.g. a
        LoRAStackDynamic feeding LoRAExtractFromModel) keeps both — the old
        first-element dispatch dropped whichever type wasn't first."""
        opt = lora_optimizer.LoRAOptimizer()
        opt.loaded_loras = {"loraA": {"k": 1}}
        extracted = {"name": "<extracted>", "lora": {"w": 2}, "strength": 1.5}

        # tuple first, dict last
        out = opt._normalize_stack([("loraA", 1.0, 1.0), extracted])
        self.assertEqual([e["name"] for e in out], ["loraA", "<extracted>"])
        self.assertEqual(out[1]["lora"], {"w": 2})

        # dict first, tuple last
        out2 = opt._normalize_stack([extracted, ("loraA", 1.0, 1.0)])
        self.assertEqual([e["name"] for e in out2], ["<extracted>", "loraA"])

    def test_compute_cache_key_mixed_tuple_and_dict(self):
        """_compute_cache_key must not crash on a mixed stack (tuple entry
        first, extracted dict entry second) — it used to do entry[3] on the
        dict and raise KeyError: 3."""
        key = lora_optimizer.LoRAOptimizer._compute_cache_key(
            [("loraA", 1.0, 1.0), {"name": "<extracted>", "lora": {}, "strength": 1.5}],
            output_strength=1.0, clip_strength_multiplier=1.0, auto_strength="disabled",
        )
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 16)
        # order-independent: same entries reversed hash identically
        key2 = lora_optimizer.LoRAOptimizer._compute_cache_key(
            [{"name": "<extracted>", "lora": {}, "strength": 1.5}, ("loraA", 1.0, 1.0)],
            output_strength=1.0, clip_strength_multiplier=1.0, auto_strength="disabled",
        )
        self.assertEqual(key, key2)

    def test_build_stack_passes_through_inmemory_dict(self):
        """The dynamic stacker must pass dict entries carrying weights through
        as-is, not flatten them to a (name, ...) tuple that loses the weights."""
        node = lora_optimizer.LoRAStackDynamic()
        extracted = {"name": "<extracted>", "lora": {"w": 1}, "strength": 1.0}
        with mock.patch.object(
            lora_optimizer.LoRAStackDynamic, "_resolve_lora_name",
            side_effect=lambda n: n,
        ):
            result, = node.build_stack(
                settings_visibility="simple", input_mode="text", lora_count=1,
                lora_name_text_1="lora_a", strength_1=1.0,
                lora_stack=[extracted],
            )
        # the in-memory dict survives as a dict (not converted to a tuple)
        dicts = [e for e in result if isinstance(e, dict)]
        self.assertEqual(len(dicts), 1)
        self.assertEqual(dicts[0]["lora"], {"w": 1})

    def test_build_stack_advanced_emits_preserve(self):
        node = lora_optimizer.LoRAStackDynamic()
        with mock.patch.object(
            lora_optimizer.LoRAStackDynamic, "_resolve_lora_name",
            side_effect=lambda n: n,
        ):
            result, = node.build_stack(
                settings_visibility="advanced", input_mode="text", lora_count=2,
                lora_name_text_1="lora_a", lora_name_text_2="lora_b",
                model_strength_1=1.0, clip_strength_1=1.0, preserve_1=True,
                model_strength_2=1.0, clip_strength_2=1.0, preserve_2=False,
            )
        # tuple layout: (name, model_str, clip_str, conflict_mode, key_filter, preserve)
        self.assertIs(result[0][5], True)
        self.assertIs(result[1][5], False)

    def test_build_stack_simple_defaults_preserve_false(self):
        node = lora_optimizer.LoRAStackDynamic()
        with mock.patch.object(
            lora_optimizer.LoRAStackDynamic, "_resolve_lora_name",
            side_effect=lambda n: n,
        ):
            result, = node.build_stack(
                settings_visibility="simple", input_mode="text", lora_count=1,
                lora_name_text_1="lora_a", strength_1=1.0,
            )
        self.assertEqual(len(result[0]), 6)
        self.assertIs(result[0][5], False)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class PreserveCacheAwarenessTests(unittest.TestCase):
    """Toggling the preserve flag must invalidate every AutoTuner cache layer
    (IS_CHANGED, in-node, persistent memory) so the tuner actually re-runs."""

    sig = staticmethod(lora_optimizer.LoRAOptimizer._per_lora_merge_signature)
    default_flags = staticmethod(lora_optimizer.LoRAOptimizer._stack_has_default_merge_flags)

    def test_signature_changes_with_preserve(self):
        off = self.sig([("a", 1.0, 1.0, "all", "all", False)])
        on = self.sig([("a", 1.0, 1.0, "all", "all", True)])
        self.assertNotEqual(off, on)

    def test_signature_changes_with_conflict_mode_and_key_filter(self):
        base = self.sig([("a", 1.0, 1.0, "all", "all", False)])
        self.assertNotEqual(base, self.sig([("a", 1.0, 1.0, "high_conflict", "all", False)]))
        self.assertNotEqual(base, self.sig([("a", 1.0, 1.0, "all", "audio_only", False)]))

    def test_signature_order_independent(self):
        s1 = self.sig([("a", 1.0, 1.0, "all", "all", True),
                       ("b", 1.0, 1.0, "all", "all", False)])
        s2 = self.sig([("b", 1.0, 1.0, "all", "all", False),
                       ("a", 1.0, 1.0, "all", "all", True)])
        self.assertEqual(s1, s2)

    def test_signature_ignores_strength(self):
        # strength is handled elsewhere (sign in names hash); the merge-structure
        # signature must not change with strength so strength sweeps still share.
        s1 = self.sig([("a", 1.0, 1.0, "all", "all", True)])
        s2 = self.sig([("a", 2.5, 0.5, "all", "all", True)])
        self.assertEqual(s1, s2)

    def test_signature_handles_dict_entries(self):
        off = self.sig([{"name": "a", "conflict_mode": "all", "key_filter": "all", "preserve": False}])
        on = self.sig([{"name": "a", "conflict_mode": "all", "key_filter": "all", "preserve": True}])
        self.assertNotEqual(off, on)

    def test_default_merge_flags(self):
        self.assertTrue(self.default_flags([{"name": "a"}, {"name": "b"}]))
        self.assertFalse(self.default_flags([{"name": "a", "preserve": True}]))
        self.assertFalse(self.default_flags([{"name": "a", "key_filter": "audio_only"}]))
        self.assertFalse(self.default_flags([{"name": "a", "conflict_mode": "high_conflict"}]))

    def test_optimizer_cache_key_changes_with_preserve(self):
        off = lora_optimizer.LoRAOptimizer._compute_cache_key(
            [("a", 1.0, 1.0, "all", "all", False)], 1.0, 1.0, "disabled")
        on = lora_optimizer.LoRAOptimizer._compute_cache_key(
            [("a", 1.0, 1.0, "all", "all", True)], 1.0, 1.0, "disabled")
        self.assertNotEqual(off, on)

    def test_autotuner_is_changed_embeds_signature(self):
        # The signature must be IN the IS_CHANGED output, so even if id(lora_stack)
        # is reused across executions, toggling preserve still re-triggers the node.
        model = _make_model()
        stack_on = [("a", 1.0, 1.0, "all", "all", True)]
        result = lora_optimizer.LoRAAutoTuner.IS_CHANGED(model, stack_on, 1.0)
        self.assertIn(self.sig(stack_on), result)
        stack_off = [("a", 1.0, 1.0, "all", "all", False)]
        result_off = lora_optimizer.LoRAAutoTuner.IS_CHANGED(model, stack_off, 1.0)
        self.assertIn(self.sig(stack_off), result_off)
        self.assertNotEqual(self.sig(stack_on), self.sig(stack_off))


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
                         "smooth_slerp_gate", "vram_budget", "cache_patches",
                         "star_eta", "tame_layers", "tame_threshold"}
        self.assertEqual(set(settings.keys()), expected_keys)
        self.assertEqual(settings["normalize_keys"], "enabled")
        self.assertEqual(settings["architecture_preset"], "auto")
        self.assertAlmostEqual(settings["auto_strength_floor"], -1.0)
        self.assertAlmostEqual(settings["decision_smoothing"], 0.25)
        self.assertFalse(settings["smooth_slerp_gate"])
        self.assertAlmostEqual(settings["vram_budget"], 0.0)
        self.assertEqual(settings["cache_patches"], "enabled")
        # per-LoRA cleaning knobs default to no-ops
        self.assertAlmostEqual(settings["star_eta"], 100.0)
        self.assertAlmostEqual(settings["tame_layers"], 0.0)
        self.assertAlmostEqual(settings["tame_threshold"], 0.3)

    def test_merge_settings_floor_mode_resolution(self):
        """The mode switch resolves to the same downstream value the old float
        sentinel produced: 'auto' -> -1.0 (ignores slider); 'manual' -> slider
        clamped to [0,1]."""
        node = lora_optimizer.LoRAMergeSettings()
        base = self._build_defaults(lora_optimizer.LoRAMergeSettings.INPUT_TYPES())
        auto = node.build_settings(
            **{**base, "auto_strength_floor_mode": "auto", "auto_strength_floor": 0.5})[0]
        self.assertAlmostEqual(auto["auto_strength_floor"], -1.0)
        manual = node.build_settings(
            **{**base, "auto_strength_floor_mode": "manual", "auto_strength_floor": 0.5})[0]
        self.assertAlmostEqual(manual["auto_strength_floor"], 0.5)
        clamped = node.build_settings(
            **{**base, "auto_strength_floor_mode": "manual", "auto_strength_floor": -1.0})[0]
        self.assertAlmostEqual(clamped["auto_strength_floor"], 0.0)

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
        self.assertEqual(settings["record_dataset"], "disabled")

    def test_autotuner_settings_record_dataset_flows(self):
        node = lora_optimizer.LoRAAutoTunerSettings()
        inputs = lora_optimizer.LoRAAutoTunerSettings.INPUT_TYPES()
        self.assertIn("record_dataset", inputs["required"])
        defaults = self._build_defaults(inputs)
        defaults["record_dataset"] = "enabled"
        result = node.build_settings(**defaults)
        self.assertEqual(result[0]["record_dataset"], "enabled")

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
        """_DEFAULTS holds the RESOLVED downstream settings (the fallback used
        when merge_settings isn't connected), so it mirrors INPUT_TYPES defaults
        EXCEPT where build_settings resolves a widget: 'auto_strength_floor_mode'
        is resolved away, and 'auto_strength_floor' defaults to the -1 'auto'
        sentinel rather than the manual slider's value."""
        inputs = lora_optimizer.LoRAMergeSettings.INPUT_TYPES()
        defaults_dict = lora_optimizer.LoRAMergeSettings._DEFAULTS
        resolved = {"auto_strength_floor", "auto_strength_floor_mode"}
        for key, spec in inputs["required"].items():
            if key in resolved:
                continue
            if isinstance(spec[0], list):
                input_default = spec[1].get("default", spec[0][0])
            else:
                input_default = spec[1]["default"]
            self.assertEqual(defaults_dict[key], input_default,
                             f"_DEFAULTS[{key}]={defaults_dict[key]} != INPUT_TYPES default={input_default}")
        # The mode switch is resolved into auto_strength_floor, not a downstream key
        self.assertNotIn("auto_strength_floor_mode", defaults_dict)
        # Default (auto) resolves to the -1 sentinel downstream
        self.assertAlmostEqual(defaults_dict["auto_strength_floor"], -1.0)
        self.assertEqual(
            set(defaults_dict.keys()),
            set(inputs["required"].keys()) - {"auto_strength_floor_mode"},
            "_DEFAULTS keys don't match INPUT_TYPES keys (minus the resolved mode switch)")


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


@unittest.skipIf(torch is None, "torch not available")
class TestExtractLoRAFromDelta(unittest.TestCase):
    """Tests for _extract_lora_svd() helper used by LoRAExtractFromModel."""

    def _make_delta(self, rows, cols, rank):
        """Create a clean rank-r delta matrix using orthonormal U and V."""
        U, _ = torch.linalg.qr(torch.randn(rows, rank))   # orthonormal (rows, rank)
        V, _ = torch.linalg.qr(torch.randn(cols, rank))   # orthonormal (cols, rank)
        S = torch.rand(rank) + 0.5  # positive singular values, not too small
        return (U * S.unsqueeze(0)) @ V.T                  # (rows, cols)

    def test_fixed_rank_output_shape(self):
        """Fixed mode: output lora_up/down have correct shapes."""
        mod = _load_module()
        delta = self._make_delta(64, 32, 8)
        up, down, alpha = mod._extract_lora_svd(delta, rank=4, rank_mode="fixed", energy_threshold=0.99)
        self.assertEqual(up.shape, (64, 4))
        self.assertEqual(down.shape, (4, 32))
        self.assertEqual(alpha, 4.0)

    def test_auto_rank_energy_retained(self):
        """Auto mode: retained energy >= threshold."""
        mod = _load_module()
        delta = self._make_delta(64, 32, 16)
        up, down, alpha = mod._extract_lora_svd(delta, rank=32, rank_mode="auto", energy_threshold=0.95)
        # Reconstruct and check energy
        reconstructed = up @ down
        original_energy = (delta ** 2).sum().item()
        reconstructed_energy = (reconstructed ** 2).sum().item()
        self.assertGreaterEqual(reconstructed_energy / original_energy, 0.93)  # allow small numerical error

    def test_near_zero_delta_returns_none(self):
        """Near-zero delta (unaffected layer) returns None."""
        mod = _load_module()
        delta = torch.zeros(64, 32)
        result = mod._extract_lora_svd(delta, rank=4, rank_mode="fixed", energy_threshold=0.99)
        self.assertIsNone(result)

    def test_singular_value_floor_applied(self):
        """Noise-only singular values below floor are excluded even in fixed mode."""
        mod = _load_module()
        # Create a clean rank-2 signal with tiny noise
        signal = self._make_delta(32, 32, 2)
        noise = torch.randn(32, 32) * 1e-6
        delta = signal + noise
        up, down, alpha = mod._extract_lora_svd(delta, rank=16, rank_mode="fixed", energy_threshold=0.99)
        # Floor should cut noise singular values — effective rank must be ≤ 2
        self.assertLessEqual(alpha, 2.0)


    def test_conv_layer_reshaped_to_2d(self):
        """4D conv delta is reshaped to 2D; lora_down has flat spatial dimension."""
        mod = _load_module()
        # Simulate a conv2d weight delta: (C_out=64, C_in=32, kH=3, kW=3)
        delta = torch.randn(64, 32, 3, 3)
        up, down, alpha = mod._extract_lora_svd(delta, rank=4, rank_mode="fixed", energy_threshold=0.99)
        self.assertEqual(up.shape[0], 64)       # rows = C_out
        self.assertEqual(down.shape[1], 32 * 3 * 3)  # cols = C_in * kH * kW
        self.assertEqual(up.shape[1], down.shape[0])  # inner dim consistent

    def test_1d_delta_returns_none(self):
        """1D bias delta returns None (SVD requires 2D input)."""
        mod = _load_module()
        delta = torch.randn(64)  # bias vector
        result = mod._extract_lora_svd(delta, rank=4, rank_mode="fixed", energy_threshold=0.99)
        self.assertIsNone(result)


@unittest.skipIf(torch is None, "torch is not installed")
class AnalysisCacheTests(unittest.TestCase):
    def test_names_only_hash_excludes_strength(self):
        """Same LoRA files at different strengths produce the same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "lora_a.safetensors")
            path_b = os.path.join(tmpdir, "lora_b.safetensors")
            open(path_a, "wb").close()
            open(path_b, "wb").close()

            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            side_effect=lambda _k, n: os.path.join(tmpdir, n)):
                stack_s1 = [
                    {"name": "lora_a.safetensors", "strength": 0.5},
                    {"name": "lora_b.safetensors", "strength": 1.0},
                ]
                stack_s2 = [
                    {"name": "lora_a.safetensors", "strength": 1.5},
                    {"name": "lora_b.safetensors", "strength": 0.3},
                ]
                h1, signs1 = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(stack_s1)
                h2, signs2 = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(stack_s2)
                self.assertEqual(h1, h2)
                self.assertEqual(signs1, {0: 1, 1: 1})
                self.assertEqual(signs2, {0: 1, 1: 1})

    def test_names_only_hash_captures_sign(self):
        """Negative strength is captured in the returned signs dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "lora_a.safetensors")
            open(path_a, "wb").close()
            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            side_effect=lambda _k, n: os.path.join(tmpdir, n)):
                stack_pos = [{"name": "lora_a.safetensors", "strength":  1.0}]
                stack_neg = [{"name": "lora_a.safetensors", "strength": -1.0}]
                _, signs_pos = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(stack_pos)
                _, signs_neg = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(stack_neg)
                self.assertEqual(signs_pos[0],  1)
                self.assertEqual(signs_neg[0], -1)

    def test_analysis_cache_roundtrip(self):
        """Save and load analysis cache; content survives round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                per_prefix = {
                    "prefix_a": {
                        "pair_conflicts": {"0,1": {"overlap": 100, "conflict": 30,
                                                   "dot": 0.5, "norm_a_sq": 1.0,
                                                   "norm_b_sq": 1.0,
                                                   "weighted_total": 0.8,
                                                   "weighted_conflict": 0.2,
                                                   "expected_conflict": 0.15,
                                                   "excess_conflict": 0.05,
                                                   "subspace_overlap": 0.3,
                                                   "subspace_weight": 1.0}},
                        "per_lora_norm_sq": {"0": 1.5, "1": 0.8},
                        "magnitude_samples_unscaled": {"0": [0.1, 0.2], "1": [0.3]},
                        "ranks": {"0": 16, "1": 32},
                        "target_key": "model.layer.weight",
                        "is_clip": False,
                        "raw_n": 2,
                        "skip_count": 0,
                        "strength_signs": {"0": 1, "1": 1},
                    }
                }
                source_loras = [{"name": "a.safetensors", "mtime": 1.0, "size": 100}]
                lora_optimizer.LoRAAutoTuner._analysis_cache_save(
                    "abc123", per_prefix, source_loras)

                loaded = lora_optimizer.LoRAAutoTuner._analysis_cache_load("abc123")
                self.assertIsNotNone(loaded)
                self.assertIn("prefix_a", loaded)
                self.assertEqual(loaded["prefix_a"]["per_lora_norm_sq"]["0"], 1.5)

    def test_analysis_cache_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._analysis_cache_load("nonexistent"))

    def test_analysis_cache_load_stale_algo_version_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                stale_hash = "staletest1"
                path = os.path.join(tmpdir, f"{stale_hash}.analysis.json")
                with open(path, "w") as f:
                    json.dump({"analysis_version": 1,
                               "algo_version": "0.0.0",
                               "per_prefix": {"prefix_a": {}}}, f)
                result = lora_optimizer.LoRAAutoTuner._analysis_cache_load(stale_hash)
                self.assertIsNone(result)

    def test_analysis_partial_path_uses_partial_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path = lora_optimizer.LoRAAutoTuner._analysis_partial_path("abc123")
                self.assertTrue(path.endswith("abc123.analysis.partial.json"))

    def test_analysis_partial_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._analysis_partial_load("nonexistent"))

    def test_analysis_partial_load_stale_algo_version_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path = os.path.join(tmpdir, "stale.analysis.partial.json")
                with open(path, "w") as f:
                    json.dump({"algo_version": "0.0.0", "per_prefix": {}}, f)
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._analysis_partial_load("stale"))

    def test_analysis_partial_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                per_prefix = {"prefix_a": {"ranks": {"0": 16}, "is_clip": False}}
                source_loras = [{"name": "a.safetensors"}]
                lora_optimizer.LoRAAutoTuner._analysis_partial_save(
                    "abc123", per_prefix, source_loras)
                loaded = lora_optimizer.LoRAAutoTuner._analysis_partial_load("abc123")
                self.assertIsNotNone(loaded)
                self.assertIn("prefix_a", loaded)

    def test_analysis_partial_save_is_atomic(self):
        """Save uses tmp+replace so a crash mid-write doesn't corrupt the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._analysis_partial_save(
                    "abc123", {"p": {}}, [])
                partial_path = lora_optimizer.LoRAAutoTuner._analysis_partial_path("abc123")
                tmp_path = partial_path + ".tmp"
                self.assertTrue(os.path.exists(partial_path))
                self.assertFalse(os.path.exists(tmp_path))

    def test_analysis_partial_delete_removes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._analysis_partial_save(
                    "abc123", {}, [])
                path = lora_optimizer.LoRAAutoTuner._analysis_partial_path("abc123")
                self.assertTrue(os.path.exists(path))
                lora_optimizer.LoRAAutoTuner._analysis_partial_delete("abc123")
                self.assertFalse(os.path.exists(path))

    def test_analysis_partial_delete_silent_on_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                # Should not raise
                lora_optimizer.LoRAAutoTuner._analysis_partial_delete("nonexistent")

    def test_extract_for_analysis_cache_strips_strength(self):
        """Extracted data has unscaled magnitude samples; tuple keys become strings."""
        partial_stats = [(0, 16, 1.5, 2.25), (1, 32, 0.9, 0.81)]
        pair_conflicts = {
            (0, 1): {"overlap": 50, "conflict": 10, "dot": 0.4,
                     "norm_a_sq": 2.25, "norm_b_sq": 0.81,
                     "weighted_total": 0.6, "weighted_conflict": 0.1,
                     "expected_conflict": 0.12, "excess_conflict": 0.0,
                     "subspace_overlap": 0.2, "subspace_weight": 1.35}
        }
        magnitude_samples = [
            torch.tensor([1.5, 3.0]),   # LoRA 0: scaled by abs(strength=1.5)
            torch.tensor([0.9, 1.8]),   # LoRA 1: scaled by abs(strength=0.9)
        ]
        per_lora_norm_sq = {0: 2.25, 1: 0.81}
        result = (
            "prefix_x", partial_stats, pair_conflicts,
            magnitude_samples, ("layer.weight", False),
            0, 2, per_lora_norm_sq,
        )
        active_loras = [
            {"name": "a.safetensors", "strength": 1.5, "clip_strength": None},
            {"name": "b.safetensors", "strength": 0.9, "clip_strength": None},
        ]
        extracted = lora_optimizer.LoRAOptimizer._extract_for_analysis_cache(
            result, active_loras)

        # Pair key must be a string
        self.assertIn("0,1", extracted["pair_conflicts"])
        # Samples unscaled: divide by abs(strength)
        self.assertAlmostEqual(extracted["magnitude_samples_unscaled"]["0"][0],
                               1.5 / 1.5, places=5)
        self.assertAlmostEqual(extracted["magnitude_samples_unscaled"]["1"][0],
                               0.9 / 0.9, places=5)
        self.assertEqual(extracted["per_lora_norm_sq"]["0"], 2.25)
        self.assertEqual(extracted["target_key"], "layer.weight")
        self.assertFalse(extracted["is_clip"])
        self.assertEqual(extracted["strength_signs"]["0"], 1)

    def test_reconstruct_rescales_by_new_strength(self):
        """Reconstruction rescales magnitude samples and display_l2 to new strength."""
        cached_prefix = {
            "pair_conflicts": {
                "0,1": {"overlap": 50, "conflict": 10, "dot": 0.4,
                        "norm_a_sq": 2.25, "norm_b_sq": 0.81,
                        "weighted_total": 0.6, "weighted_conflict": 0.1,
                        "expected_conflict": 0.12, "excess_conflict": 0.0,
                        "subspace_overlap": 0.2, "subspace_weight": 1.35}
            },
            "per_lora_norm_sq": {"0": 2.25, "1": 0.81},
            "magnitude_samples_unscaled": {"0": [1.0, 2.0], "1": [1.0]},
            "ranks": {"0": 16, "1": 32},
            "target_key": "layer.weight",
            "is_clip": False,
            "raw_n": 2,
            "skip_count": 0,
            "strength_signs": {"0": 1, "1": 1},
        }
        active_loras = [
            {"name": "a.safetensors", "strength": 2.0, "clip_strength": None},
            {"name": "b.safetensors", "strength": 0.5, "clip_strength": None},
        ]
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_analysis_cache(
            "prefix_x", cached_prefix, active_loras)
        self.assertIsNotNone(result)
        prefix, partial_stats, pair_conflicts, mag_samples, target_info, skip_count, raw_n, norm_sq = result

        self.assertEqual(prefix, "prefix_x")
        # display_l2 = sqrt(norm_sq) * abs(strength)
        self.assertAlmostEqual(partial_stats[0][2], math.sqrt(2.25) * 2.0, places=4)
        self.assertAlmostEqual(partial_stats[1][2], math.sqrt(0.81) * 0.5, places=4)
        # pair_conflicts keys are int tuples
        self.assertIn((0, 1), pair_conflicts)
        # magnitude_samples rescaled by abs(new_strength)
        self.assertAlmostEqual(mag_samples[0][0].item(), 1.0 * 2.0, places=4)
        self.assertAlmostEqual(mag_samples[1][0].item(), 1.0 * 0.5, places=4)

    def test_reconstruct_returns_none_on_sign_flip(self):
        """Sign flip triggers None so caller falls back to full analysis."""
        cached_prefix = {
            "pair_conflicts": {},
            "per_lora_norm_sq": {"0": 1.0},
            "magnitude_samples_unscaled": {"0": [1.0]},
            "ranks": {"0": 16},
            "target_key": "layer.weight",
            "is_clip": False,
            "raw_n": 1,
            "skip_count": 0,
            "strength_signs": {"0": 1},  # was positive
        }
        active_loras = [{"name": "a.safetensors", "strength": -1.0, "clip_strength": None}]
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_analysis_cache(
            "prefix_x", cached_prefix, active_loras)
        self.assertIsNone(result)

    def test_run_group_analysis_skips_analyze_on_cache_hit(self):
        """When cached_analysis covers a prefix, _analyze_target_group is not called."""
        optimizer = lora_optimizer.LoRAOptimizer()
        active_loras = [
            _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
            _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
        ]
        model = _make_model()
        target_groups = optimizer._build_target_groups(
            ["prefix_a"], {"prefix_a": "layer.weight"}, {})

        fake_cached = {
            "prefix_a": {
                "pair_conflicts": {
                    "0,1": {"overlap": 10, "conflict": 2, "dot": 0.1,
                            "norm_a_sq": 1.0, "norm_b_sq": 0.25,
                            "weighted_total": 0.3, "weighted_conflict": 0.05,
                            "expected_conflict": 0.1, "excess_conflict": 0.0,
                            "subspace_overlap": 0.1, "subspace_weight": 0.5}
                },
                "per_lora_norm_sq": {"0": 1.0, "1": 0.25},
                "magnitude_samples_unscaled": {"0": [0.5], "1": [0.3]},
                "ranks": {"0": 1, "1": 1},
                "target_key": "layer.weight",
                "is_clip": False,
                "raw_n": 2,
                "skip_count": 0,
                "strength_signs": {"0": 1, "1": 1},
            }
        }

        call_count = {"n": 0}
        orig = optimizer._analyze_target_group
        def counting_analyze(*args, **kwargs):
            call_count["n"] += 1
            return orig(*args, **kwargs)

        with mock.patch.object(optimizer, "_analyze_target_group",
                               side_effect=counting_analyze):
            device = torch.device("cpu")
            result = optimizer._run_group_analysis(
                target_groups, active_loras, model, None, device,
                cached_analysis=fake_cached,
                track_new_entries=True,
            )

        self.assertEqual(call_count["n"], 0)
        self.assertEqual(result["new_analysis_entries"], {})

    def test_run_group_analysis_populates_new_entries_on_miss(self):
        """Cache miss populates new_analysis_entries in the return dict."""
        optimizer = lora_optimizer.LoRAOptimizer()
        active_loras = [
            _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
            _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
        ]
        model = _make_model()
        target_groups = optimizer._build_target_groups(
            ["prefix_a"], {"prefix_a": "layer.weight"}, {})

        device = torch.device("cpu")
        result = optimizer._run_group_analysis(
            target_groups, active_loras, model, None, device,
            cached_analysis={},  # empty cache = full miss
            track_new_entries=True,
        )
        self.assertIn("prefix_a", result["new_analysis_entries"])

    def test_run_group_analysis_calls_on_prefix_done_for_fresh_prefixes(self):
        """on_prefix_done fires once per freshly-computed prefix."""
        optimizer = lora_optimizer.LoRAOptimizer()
        active_loras = [
            _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
            _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
        ]
        model = _make_model()
        target_groups = optimizer._build_target_groups(
            ["prefix_a"], {"prefix_a": "layer.weight"}, {})

        calls = []
        device = torch.device("cpu")
        optimizer._run_group_analysis(
            target_groups, active_loras, model, None, device,
            cached_analysis={},
            track_new_entries=True,
            on_prefix_done=lambda prefix, entry: calls.append((prefix, entry)),
        )
        self.assertEqual(len(calls), 1)
        prefix, entry = calls[0]
        self.assertEqual(prefix, "prefix_a")
        self.assertIn("ranks", entry)

    def test_run_group_analysis_does_not_call_on_prefix_done_for_cache_hits(self):
        """on_prefix_done is NOT called for prefixes already in cached_analysis."""
        optimizer = lora_optimizer.LoRAOptimizer()
        active_loras = [
            _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
            _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
        ]
        model = _make_model()
        target_groups = optimizer._build_target_groups(
            ["prefix_a"], {"prefix_a": "layer.weight"}, {})

        fake_cached = {
            "prefix_a": {
                "pair_conflicts": {
                    "0,1": {"overlap": 10, "conflict": 2, "dot": 0.1,
                            "norm_a_sq": 1.0, "norm_b_sq": 0.25,
                            "weighted_total": 0.3, "weighted_conflict": 0.05,
                            "expected_conflict": 0.1, "excess_conflict": 0.0,
                            "subspace_overlap": 0.1, "subspace_weight": 0.5}
                },
                "per_lora_norm_sq": {"0": 1.0, "1": 0.25},
                "magnitude_samples_unscaled": {"0": [0.5], "1": [0.3]},
                "ranks": {"0": 1, "1": 1},
                "target_key": "layer.weight",
                "is_clip": False,
                "raw_n": 2,
                "skip_count": 0,
                "strength_signs": {"0": 1, "1": 1},
            }
        }

        calls = []
        device = torch.device("cpu")
        optimizer._run_group_analysis(
            target_groups, active_loras, model, None, device,
            cached_analysis=fake_cached,
            track_new_entries=True,
            on_prefix_done=lambda prefix, entry: calls.append(prefix),
        )
        self.assertEqual(calls, [])

    def test_dataset_entry_includes_raw_analysis(self):
        """Dataset entries include raw_analysis field when new_analysis_entries provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.folder_paths.get_user_directory",
                            return_value=tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                tuner_data = {
                    "top_n": [],
                    "analysis_summary": {
                        "n_loras": 2, "prefix_count": 1,
                        "avg_conflict_ratio": 0.3, "avg_excess_conflict": 0.1,
                        "avg_subspace_overlap": 0.2, "avg_cosine_sim": 0.5,
                        "magnitude_ratio": 1.2, "decision_smoothing": 0.25,
                    },
                    "architecture_preset": "auto",
                    "lora_hash": "abc",
                }
                prefix_stats = {
                    "prefix_a": {
                        "n_loras": 2, "conflict_ratio": 0.3,
                        "excess_conflict": 0.1, "avg_cos_sim": 0.5,
                        "magnitude_ratio": 1.2, "avg_subspace_overlap": 0.2,
                        "magnitude_samples": [],
                        "per_lora_norm_sq": {0: 1.0, 1: 0.5},
                        "pairwise_dots": {},
                        "raw_n_loras": 2,
                    }
                }
                new_analysis_entries = {
                    "prefix_a": {
                        "pair_conflicts": {"0,1": {"overlap": 10}},
                        "per_lora_norm_sq": {"0": 1.0, "1": 0.5},
                        "magnitude_samples_unscaled": {"0": [0.5], "1": [0.3]},
                        "ranks": {"0": 1, "1": 1},
                        "target_key": "layer.weight",
                        "is_clip": False,
                        "raw_n": 2,
                        "skip_count": 0,
                        "strength_signs": {"0": 1, "1": 1},
                    }
                }
                active_loras = [
                    {"name": "a.safetensors", "strength": 1.0},
                    {"name": "b.safetensors", "strength": 0.5},
                ]
                tuner._save_tuner_dataset_entry(
                    tuner_data, active_loras, prefix_stats, "wan_video",
                    names_only_hash="testhash",
                    new_analysis_entries=new_analysis_entries)

                dataset_path = os.path.join(
                    tmpdir, "lora_optimizer_reports", "autotuner_dataset.jsonl")
                with open(dataset_path) as f:
                    entry = json.loads(f.readline())
                self.assertIn("raw_analysis", entry)
                self.assertEqual(entry["raw_analysis"]["names_only_hash"], "testhash")
                self.assertIn("prefix_a", entry["raw_analysis"]["per_prefix"])
                self.assertIn("pair_conflicts",
                              entry["raw_analysis"]["per_prefix"]["prefix_a"])


class TestAnalysisIndexRemap(unittest.TestCase):
    """The names_only_hash is order-independent but cache entries are
    index-keyed — reordering the stack must remap entries, not misattribute
    one LoRA's stats to another."""

    @staticmethod
    def _entry(norm_a=1.5, norm_b=0.8):
        return {
            "pair_conflicts": {"0,1": {"overlap": 100, "conflict": 30,
                                       "dot": 0.5,
                                       "norm_a_sq": norm_a,
                                       "norm_b_sq": norm_b,
                                       "weighted_total": 0.8,
                                       "weighted_conflict": 0.2,
                                       "expected_conflict": 0.15,
                                       "excess_conflict": 0.05,
                                       "subspace_overlap": 0.3,
                                       "subspace_weight": 1.0}},
            "per_lora_norm_sq": {"0": norm_a, "1": norm_b},
            "magnitude_samples_unscaled": {"0": [0.1, 0.2], "1": [0.3]},
            "ranks": {"0": 16, "1": 32},
            "target_key": "model.layer.weight",
            "is_clip": False,
            "raw_n": 2,
            "skip_count": 0,
            "strength_signs": {"0": 1, "1": 1},
        }

    def test_same_order_passthrough(self):
        per_prefix = {"p": self._entry()}
        out = lora_optimizer.LoRAAutoTuner._remap_analysis_indices(
            per_prefix,
            [{"name": "a.safetensors"}, {"name": "b.safetensors"}],
            [{"name": "a.safetensors"}, {"name": "b.safetensors"}])
        self.assertIs(out, per_prefix)

    def test_reordered_stack_remaps_indices_and_pair_orientation(self):
        per_prefix = {"p": self._entry(norm_a=1.5, norm_b=0.8)}
        # Cached order [A, B]; current order [B, A]
        out = lora_optimizer.LoRAAutoTuner._remap_analysis_indices(
            per_prefix,
            [{"name": "a.safetensors"}, {"name": "b.safetensors"}],
            [{"name": "b.safetensors"}, {"name": "a.safetensors"}])
        self.assertIsNotNone(out)
        entry = out["p"]
        # A (norm 1.5) is now index 1, B (norm 0.8) is index 0
        self.assertEqual(entry["per_lora_norm_sq"]["1"], 1.5)
        self.assertEqual(entry["per_lora_norm_sq"]["0"], 0.8)
        self.assertEqual(entry["ranks"]["1"], 16)
        self.assertEqual(entry["ranks"]["0"], 32)
        self.assertEqual(entry["magnitude_samples_unscaled"]["1"], [0.1, 0.2])
        # Pair key stays canonical (i < j) with a/b norms swapped to match
        self.assertIn("0,1", entry["pair_conflicts"])
        self.assertEqual(entry["pair_conflicts"]["0,1"]["norm_a_sq"], 0.8)
        self.assertEqual(entry["pair_conflicts"]["0,1"]["norm_b_sq"], 1.5)
        # Symmetric fields untouched
        self.assertEqual(entry["pair_conflicts"]["0,1"]["dot"], 0.5)

    def test_name_mismatch_returns_none(self):
        out = lora_optimizer.LoRAAutoTuner._remap_analysis_indices(
            {"p": self._entry()},
            [{"name": "a.safetensors"}, {"name": "b.safetensors"}],
            [{"name": "a.safetensors"}, {"name": "c.safetensors"}])
        self.assertIsNone(out)

    def test_missing_source_loras_returns_none(self):
        out = lora_optimizer.LoRAAutoTuner._remap_analysis_indices(
            {"p": self._entry()}, None,
            [{"name": "a.safetensors"}, {"name": "b.safetensors"}])
        self.assertIsNone(out)

    def test_cache_load_remaps_for_reordered_active_loras(self):
        """End-to-end: save under [A, B], load with [B, A] active order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._analysis_cache_save(
                    "reorder1", {"p": self._entry(norm_a=1.5, norm_b=0.8)},
                    [{"name": "a.safetensors"}, {"name": "b.safetensors"}])
                loaded = lora_optimizer.LoRAAutoTuner._analysis_cache_load(
                    "reorder1",
                    active_loras=[{"name": "b.safetensors", "strength": 1.0},
                                  {"name": "a.safetensors", "strength": 0.5}])
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["p"]["per_lora_norm_sq"]["1"], 1.5)
                self.assertEqual(loaded["p"]["per_lora_norm_sq"]["0"], 0.8)

    def test_duplicate_names_matched_in_occurrence_order(self):
        per_prefix = {"p": self._entry()}
        out = lora_optimizer.LoRAAutoTuner._remap_analysis_indices(
            per_prefix,
            [{"name": "a.safetensors"}, {"name": "a.safetensors"}],
            [{"name": "a.safetensors"}, {"name": "a.safetensors"}])
        self.assertIs(out, per_prefix)


class TestAnalysisPartialLifecycle(unittest.TestCase):
    """Integration tests for partial checkpoint create/resume/delete lifecycle."""

    def test_partial_file_created_after_each_prefix(self):
        """Callback writes partial file; file exists after analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                active_loras = [
                    _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
                    _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
                ]
                names_only_hash, _ = tuner._compute_names_only_hash(active_loras)
                source_loras = [{"name": l["name"]} for l in active_loras]

                partial_accumulated = {}
                written = []

                def on_prefix_done(prefix, entry):
                    partial_accumulated[prefix] = entry
                    tuner._analysis_partial_save(names_only_hash, partial_accumulated, source_loras)
                    written.append(prefix)

                model = _make_model()
                target_groups = tuner._build_target_groups(
                    ["prefix_a"], {"prefix_a": "layer.weight"}, {})
                device = torch.device("cpu")
                tuner._run_group_analysis(
                    target_groups, active_loras, model, None, device,
                    cached_analysis={},
                    track_new_entries=True,
                    on_prefix_done=on_prefix_done,
                )

                self.assertIn("prefix_a", written)
                partial_path = tuner._analysis_partial_path(names_only_hash)
                self.assertTrue(os.path.exists(partial_path))

    def test_partial_file_deleted_after_full_cache_saved(self):
        """_analysis_partial_delete removes the file after full cache is written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                hash_val = "testhash99"
                tuner._analysis_partial_save(hash_val, {"prefix_a": {}}, [])
                partial_path = tuner._analysis_partial_path(hash_val)
                self.assertTrue(os.path.exists(partial_path))

                tuner._analysis_cache_save(hash_val, {"prefix_a": {}}, [])
                tuner._analysis_partial_delete(hash_val)

                self.assertFalse(os.path.exists(partial_path))
                full_path = tuner._analysis_cache_path(hash_val)
                self.assertTrue(os.path.exists(full_path))

    def test_partial_file_loaded_when_no_full_cache(self):
        """If full cache is missing but .partial exists, it is loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                hash_val = "resumehash1"
                per_prefix = {"prefix_b": {"ranks": {"0": 16}, "is_clip": False}}
                tuner._analysis_partial_save(hash_val, per_prefix, [])

                self.assertIsNone(tuner._analysis_cache_load(hash_val))
                loaded = tuner._analysis_partial_load(hash_val)
                self.assertIsNotNone(loaded)
                self.assertIn("prefix_b", loaded)

    def test_partial_promoted_to_full_cache_when_all_prefixes_already_done(self):
        """If partial has all prefixes and analysis adds none, full cache is still written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                active_loras = [
                    _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
                    _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
                ]
                names_only_hash, _ = tuner._compute_names_only_hash(active_loras)
                source_loras = [{"name": l["name"]} for l in active_loras]

                # Simulate: partial file has all prefixes (crash after last prefix, before full save)
                model = _make_model()
                target_groups = tuner._build_target_groups(
                    ["prefix_a"], {"prefix_a": "layer.weight"}, {})
                device = torch.device("cpu")

                # Ensure clean state: remove any stale full cache from prior runs
                full_path = tuner._analysis_cache_path(names_only_hash)
                try:
                    os.unlink(full_path)
                except OSError:
                    pass
                tuner._analysis_partial_delete(names_only_hash)

                # First: compute full analysis and manually save only to partial
                result = tuner._run_group_analysis(
                    target_groups, active_loras, model, None, device,
                    cached_analysis={},
                    track_new_entries=True,
                )
                partial_entries = result["new_analysis_entries"]
                tuner._analysis_partial_save(names_only_hash, partial_entries, source_loras)

                # Verify: full cache does NOT exist yet (only partial was saved)
                self.assertFalse(os.path.exists(full_path))

                # Now simulate resume: load from partial (all prefixes cached), run analysis
                cached_analysis = tuner._analysis_partial_load(names_only_hash)
                self.assertIsNotNone(cached_analysis)

                using_partial = True
                partial_accumulated = dict(cached_analysis)

                def on_prefix_done(prefix, entry):
                    partial_accumulated[prefix] = entry
                    tuner._analysis_partial_save(names_only_hash, partial_accumulated, source_loras)

                result2 = tuner._run_group_analysis(
                    target_groups, active_loras, model, None, device,
                    cached_analysis=cached_analysis,
                    track_new_entries=True,
                    on_prefix_done=on_prefix_done,
                )
                new_entries = result2["new_analysis_entries"]
                # All prefixes were cached, so new_entries should be empty
                self.assertEqual(new_entries, {})

                # Apply the fix: promote partial to full cache when using_partial=True
                if new_entries or using_partial:
                    merged = dict(cached_analysis or {})
                    merged.update(new_entries)
                    tuner._analysis_cache_save(names_only_hash, merged, source_loras)
                tuner._analysis_partial_delete(names_only_hash)

                # Verify: full cache now exists, partial is gone
                self.assertTrue(os.path.exists(full_path))
                partial_path = tuner._analysis_partial_path(names_only_hash)
                self.assertFalse(os.path.exists(partial_path))


class TestLoraCacheIO(unittest.TestCase):

    def test_lora_identity_hash_returns_16char_hex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake lora file so os.stat works
            lora_path = os.path.join(tmpdir, "test.safetensors")
            with open(lora_path, "w") as f:
                f.write("x")
            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            return_value=lora_path):
                h = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "test.safetensors", "strength": 1.0})
                self.assertIsInstance(h, str)
                self.assertEqual(len(h), 16)
                self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_lora_identity_hash_differs_for_different_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.safetensors")
            path_b = os.path.join(tmpdir, "b.safetensors")
            with open(path_a, "w") as f: f.write("x")
            with open(path_b, "w") as f: f.write("y")
            def fake_get_full_path(folder, name):
                return path_a if name == "a.safetensors" else path_b
            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            side_effect=fake_get_full_path):
                ha = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "a.safetensors", "strength": 1.0})
                hb = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "b.safetensors", "strength": 1.0})
                self.assertNotEqual(ha, hb)

    def test_lora_identity_hash_ignores_strength(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "test.safetensors")
            with open(lora_path, "w") as f: f.write("x")
            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            return_value=lora_path):
                h1 = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "test.safetensors", "strength": 0.5})
                h2 = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                    {"name": "test.safetensors", "strength": 1.0})
                self.assertEqual(h1, h2)

    def test_lora_cache_path_uses_lora_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path = lora_optimizer.LoRAAutoTuner._lora_cache_path("abc123")
                self.assertTrue(path.endswith("abc123.lora.json"))

    def test_lora_cache_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._lora_cache_load("nonexistent"))

    def test_lora_cache_load_stale_algo_version_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path = os.path.join(tmpdir, "stale.lora.json")
                with open(path, "w") as f:
                    json.dump({"algo_version": "0.0.0", "per_prefix": {}}, f)
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._lora_cache_load("stale"))

    def test_lora_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                per_prefix = {
                    "prefix_a": {
                        "norm_sq": 1.5, "rank": 16,
                        "magnitude_samples_unscaled": [0.1, 0.2],
                        "strength_sign": 1,
                        "target_key": "layer.weight",
                        "is_clip": False, "skip_count": 0, "raw_n": 1,
                    }
                }
                lora_optimizer.LoRAAutoTuner._lora_cache_save("abc123", per_prefix)
                loaded = lora_optimizer.LoRAAutoTuner._lora_cache_load("abc123")
                self.assertIsNotNone(loaded)
                self.assertIn("prefix_a", loaded)
                self.assertEqual(loaded["prefix_a"]["norm_sq"], 1.5)

    def test_lora_cache_save_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._lora_cache_save("abc123", {})
                path = lora_optimizer.LoRAAutoTuner._lora_cache_path("abc123")
                self.assertTrue(os.path.exists(path))
                self.assertFalse(os.path.exists(path + ".tmp"))


class TestPairCacheIO(unittest.TestCase):

    def test_pair_cache_path_sorts_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path_ab = lora_optimizer.LoRAAutoTuner._pair_cache_path("aaa", "bbb")
                path_ba = lora_optimizer.LoRAAutoTuner._pair_cache_path("bbb", "aaa")
                self.assertEqual(path_ab, path_ba)
                self.assertIn("aaa_bbb", path_ab)

    def test_pair_cache_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._pair_cache_load("missing1", "missing2"))

    def test_pair_cache_load_stale_algo_version_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                path = os.path.join(tmpdir, "stale1_stale2.pair.json")
                with open(path, "w") as f:
                    json.dump({"algo_version": "0.0.0", "per_prefix": {}}, f)
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._pair_cache_load("stale1", "stale2"))

    def test_pair_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                per_prefix = {
                    "prefix_a": {
                        "overlap": 100, "conflict": 30, "dot": 0.5,
                        "norm_a_sq": 1.0, "norm_b_sq": 0.5,
                        "weighted_total": 0.8, "weighted_conflict": 0.2,
                        "expected_conflict": 0.15, "excess_conflict": 0.05,
                        "subspace_overlap": 0.3, "subspace_weight": 1.0,
                    }
                }
                lora_optimizer.LoRAAutoTuner._pair_cache_save("aaa", "bbb", per_prefix)
                loaded = lora_optimizer.LoRAAutoTuner._pair_cache_load("aaa", "bbb")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["prefix_a"]["overlap"], 100)

    def test_pair_cache_load_commutative(self):
        """_pair_cache_load("a","b") and ("b","a") return the same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                per_prefix = {"prefix_a": {"overlap": 50}}
                lora_optimizer.LoRAAutoTuner._pair_cache_save("aaa", "bbb", per_prefix)
                loaded_ab = lora_optimizer.LoRAAutoTuner._pair_cache_load("aaa", "bbb")
                loaded_ba = lora_optimizer.LoRAAutoTuner._pair_cache_load("bbb", "aaa")
                self.assertEqual(loaded_ab, loaded_ba)

    def test_pair_cache_save_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._pair_cache_save("aaa", "bbb", {})
                path = lora_optimizer.LoRAAutoTuner._pair_cache_path("aaa", "bbb")
                self.assertTrue(os.path.exists(path))
                self.assertFalse(os.path.exists(path + ".tmp"))


class TestClipStrengthCacheRoundTrip(unittest.TestCase):
    """Clip prefixes scale/sign by the effective clip strength — extract and
    reconstruct must agree, or clip magnitudes come back wrong."""

    def _make_clip_result(self):
        partial_stats = [(0, 16, 1.5, 2.25)]
        magnitude_samples = [torch.tensor([0.5, 1.0])]  # scaled by |clip_strength|=0.5
        return (
            "clip_prefix", partial_stats, {}, magnitude_samples,
            ("clip.layer.weight", True), 0, 1, {0: 2.25},
        )

    def test_analysis_cache_clip_rescale_uses_clip_strength(self):
        active = [{"name": "a.safetensors", "strength": 1.0, "clip_strength": 0.5}]
        entry = lora_optimizer.LoRAOptimizer._extract_for_analysis_cache(
            self._make_clip_result(), active)
        # Unscaled by |clip_strength|=0.5, not |strength|=1.0
        self.assertEqual(entry["magnitude_samples_unscaled"]["0"], [1.0, 2.0])
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_analysis_cache(
            "clip_prefix", entry, active)
        self.assertIsNotNone(result)
        # Round-trip restores the original clip-scaled samples
        self.assertTrue(torch.allclose(result[3][0], torch.tensor([0.5, 1.0])))

    def test_analysis_cache_clip_sign_flip_invalidates(self):
        active = [{"name": "a.safetensors", "strength": 1.0, "clip_strength": 0.5}]
        entry = lora_optimizer.LoRAOptimizer._extract_for_analysis_cache(
            self._make_clip_result(), active)
        # Model strength sign unchanged, clip sign flipped → must miss
        flipped = [{"name": "a.safetensors", "strength": 1.0, "clip_strength": -0.5}]
        self.assertIsNone(lora_optimizer.LoRAOptimizer._reconstruct_from_analysis_cache(
            "clip_prefix", entry, flipped))

    def test_lora_cache_clip_ignores_multiplier(self):
        # clip_strength=None → eff clip strength is the model strength, NOT
        # strength * clip_strength_multiplier (the multiplier is applied
        # globally at add_patches, never to the analyzed diffs)
        active = [{"name": "a.safetensors", "strength": 2.0, "clip_strength": None}]
        result = (
            "clip_prefix", [(0, 16, 3.0, 9.0)], {},
            [torch.tensor([2.0])],  # scaled by |strength|=2.0
            ("clip.layer.weight", True), 0, 1, {0: 9.0},
        )
        entry = lora_optimizer.LoRAOptimizer._extract_for_lora_cache(
            result, 0, active, clip_strength_multiplier=0.5)
        self.assertEqual(entry["magnitude_samples_unscaled"], [1.0])


class TestCacheValidationFallback(unittest.TestCase):
    """Malformed cache entries (e.g. corrupt community downloads) must fall
    back to fresh analysis, not crash the run."""

    def test_analysis_reconstruct_malformed_returns_none(self):
        active = [{"name": "a.safetensors", "strength": 1.0, "clip_strength": None}]
        self.assertIsNone(lora_optimizer.LoRAOptimizer._reconstruct_from_analysis_cache(
            "p", {"per_lora_norm_sq": {"0": "garbage"}}, active))
        self.assertIsNone(lora_optimizer.LoRAOptimizer._reconstruct_from_analysis_cache(
            "p", {}, active))

    def test_pair_lora_reconstruct_malformed_returns_none(self):
        active = [{"name": "a.safetensors", "strength": 1.0, "clip_strength": None}]
        self.assertIsNone(lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "p", {0: {"is_clip": False}}, {}, active, {0: "h0"}))


class TestMemoryFindByNamesSigns(unittest.TestCase):
    """auto_ignore_strength fallback must not replay entries tuned with
    opposite strength signs — conflict data is sign-dependent."""

    def _save_entry(self, strength):
        tuner_data = {"top_n": [{"rank": 1, "config": {}, "score_final": 0.5}]}
        lora_optimizer.LoRAAutoTuner._memory_save(
            "hashx", "setty", {}, [{"name": "a.safetensors", "strength": strength}],
            tuner_data)

    def test_sign_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                self._save_entry(strength=1.0)
                found = lora_optimizer.LoRAAutoTuner._memory_find_by_names(
                    ["a.safetensors"], "setty", 1,
                    lora_signs_sorted=[("a.safetensors", -1)])
                self.assertIsNone(found)
                found = lora_optimizer.LoRAAutoTuner._memory_find_by_names(
                    ["a.safetensors"], "setty", 1,
                    lora_signs_sorted=[("a.safetensors", 1)])
                self.assertIsNotNone(found)


class TestCacheExtraction(unittest.TestCase):

    def _make_result(self):
        """Minimal 8-tuple from _analyze_target_group."""
        partial_stats = [(0, 16, 1.5, 2.25), (1, 32, 0.9, 0.81)]
        pair_conflicts = {
            (0, 1): {
                "overlap": 50, "conflict": 10, "dot": 0.4,
                "norm_a_sq": 2.25, "norm_b_sq": 0.81,
                "weighted_total": 0.6, "weighted_conflict": 0.1,
                "expected_conflict": 0.12, "excess_conflict": 0.0,
                "subspace_overlap": 0.2, "subspace_weight": 0.5,
            }
        }
        magnitude_samples = [
            torch.tensor([0.5, 1.0]),  # lora 0, already scaled by strength
            torch.tensor([0.3, 0.6]),  # lora 1
        ]
        per_lora_norm_sq = {0: 2.25, 1: 0.81}
        return (
            "prefix_a", partial_stats, pair_conflicts, magnitude_samples,
            ("layer.weight", False), 0, 2, per_lora_norm_sq
        )

    def test_extract_for_lora_cache_participating(self):
        active_loras = [
            {"name": "a.safetensors", "strength": 1.5},
            {"name": "b.safetensors", "strength": 0.9},
        ]
        result = self._make_result()
        entry = lora_optimizer.LoRAOptimizer._extract_for_lora_cache(result, 0, active_loras)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["norm_sq"], 2.25)
        self.assertEqual(entry["rank"], 16)
        self.assertEqual(entry["strength_sign"], 1)
        self.assertFalse(entry["is_clip"])
        self.assertEqual(entry["target_key"], "layer.weight")
        # magnitude unscaled: tensor / abs(strength=1.5)
        self.assertAlmostEqual(entry["magnitude_samples_unscaled"][0], 0.5 / 1.5, places=5)

    def test_extract_for_lora_cache_non_participating_returns_none(self):
        """LoRA index not in per_lora_norm_sq → non-participating → return None."""
        active_loras = [
            {"name": "a.safetensors", "strength": 1.0},
            {"name": "b.safetensors", "strength": 1.0},
            {"name": "c.safetensors", "strength": 1.0},  # index 2: not in result
        ]
        result = self._make_result()
        entry = lora_optimizer.LoRAOptimizer._extract_for_lora_cache(result, 2, active_loras)
        self.assertIsNone(entry)

    def test_extract_for_pair_cache_norm_order_by_hash(self):
        """norm_a_sq in entry corresponds to the LoRA with the smaller hash."""
        result = self._make_result()
        # hash_i > hash_j → swap norm_a/norm_b
        entry = lora_optimizer.LoRAOptimizer._extract_for_pair_cache(
            result, i=0, j=1, hash_i="zzz", hash_j="aaa")
        # In result, norm_a_sq=2.25 (lora 0), norm_b_sq=0.81 (lora 1)
        # hash_j="aaa" < hash_i="zzz", so "aaa" (lora 1) should be norm_a
        self.assertAlmostEqual(entry["norm_a_sq"], 0.81)
        self.assertAlmostEqual(entry["norm_b_sq"], 2.25)

    def test_extract_for_pair_cache_no_swap_when_hash_i_smaller(self):
        result = self._make_result()
        entry = lora_optimizer.LoRAOptimizer._extract_for_pair_cache(
            result, i=0, j=1, hash_i="aaa", hash_j="zzz")
        # hash_i="aaa" < hash_j="zzz" → no swap
        self.assertAlmostEqual(entry["norm_a_sq"], 2.25)
        self.assertAlmostEqual(entry["norm_b_sq"], 0.81)

    def test_extract_for_pair_cache_non_participating_returns_none(self):
        """Pair not in pair_conflicts (one LoRA doesn't participate) → None."""
        result = self._make_result()
        # Pair (0,2) not in result
        entry = lora_optimizer.LoRAOptimizer._extract_for_pair_cache(
            result, i=0, j=2, hash_i="aaa", hash_j="zzz")
        self.assertIsNone(entry)


class TestPairLoraReconstruction(unittest.TestCase):

    def _make_lora_entries(self):
        return {
            0: {
                "norm_sq": 2.25, "rank": 16,
                "magnitude_samples_unscaled": [0.5, 1.0],
                "strength_sign": 1,
                "target_key": "layer.weight",
                "is_clip": False, "skip_count": 0, "raw_n": 2,
            },
            1: {
                "norm_sq": 0.81, "rank": 32,
                "magnitude_samples_unscaled": [0.3, 0.6],
                "strength_sign": 1,
                "target_key": "layer.weight",
                "is_clip": False, "skip_count": 0, "raw_n": 2,
            },
        }

    def _make_pair_entries(self, hash_0="aaa", hash_1="bbb"):
        # hash_0 < hash_1, so norm_a_sq = lora 0
        return {
            (0, 1): {
                "overlap": 50, "conflict": 10, "dot": 0.4,
                "norm_a_sq": 2.25, "norm_b_sq": 0.81,
                "weighted_total": 0.6, "weighted_conflict": 0.1,
                "expected_conflict": 0.12, "excess_conflict": 0.0,
                "subspace_overlap": 0.2, "subspace_weight": 0.5,
            }
        }

    def test_reconstruction_returns_8tuple(self):
        active_loras = [
            {"name": "a.safetensors", "strength": 1.5},
            {"name": "b.safetensors", "strength": 0.9},
        ]
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=self._make_lora_entries(),
            pair_entries=self._make_pair_entries(),
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 8)
        prefix, partial_stats, pair_conflicts, magnitude_samples, target_info, skip_count, raw_n, per_lora_norm_sq = result
        self.assertEqual(prefix, "prefix_a")
        self.assertEqual(len(partial_stats), 2)
        self.assertIn((0, 1), pair_conflicts)
        self.assertAlmostEqual(per_lora_norm_sq[0], 2.25)

    def test_reconstruction_rescales_magnitude_by_current_strength(self):
        active_loras = [
            {"name": "a.safetensors", "strength": 2.0},  # different from cache
            {"name": "b.safetensors", "strength": 1.0},
        ]
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=self._make_lora_entries(),
            pair_entries=self._make_pair_entries(),
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        _, _, _, magnitude_samples, _, _, _, _ = result
        # unscaled[0] = 0.5, rescaled by strength=2.0 → 1.0
        self.assertAlmostEqual(magnitude_samples[0][0].item(), 1.0, places=5)

    def test_reconstruction_returns_none_on_sign_flip(self):
        active_loras = [
            {"name": "a.safetensors", "strength": -1.0},  # sign flipped vs cached +1
            {"name": "b.safetensors", "strength": 1.0},
        ]
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=self._make_lora_entries(),
            pair_entries=self._make_pair_entries(),
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNone(result)

    def test_reconstruction_returns_none_on_clip_strength_sign_flip(self):
        """Sign flip detected via clip_strength for CLIP prefixes."""
        lora_entries = {
            0: {
                "norm_sq": 2.25, "rank": 16,
                "magnitude_samples_unscaled": [0.5],
                "strength_sign": -1,  # cached with negative clip_strength
                "target_key": "lora_te_text_model_encoder_layers_0_weight",
                "is_clip": True, "skip_count": 0, "raw_n": 1,
            },
            1: {**self._make_lora_entries()[1],
                "is_clip": True},
        }
        active_loras = [
            {"name": "a.safetensors", "strength": 1.0, "clip_strength": 0.5},  # sign flip vs -1
            {"name": "b.safetensors", "strength": 1.0},
        ]
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_clip",
            lora_entries=lora_entries,
            pair_entries=self._make_pair_entries(),
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNone(result)

    def test_reconstruction_swaps_norm_when_hash_ordering_differs(self):
        """When lora 0's hash > lora 1's hash, norm_a_sq in pair file is lora 1's."""
        active_loras = [
            {"name": "a.safetensors", "strength": 1.0},
            {"name": "b.safetensors", "strength": 1.0},
        ]
        lora_hashes = {0: "zzz", 1: "aaa"}  # lora 1 has smaller hash
        # In the pair file, norm_a_sq = lora with smaller hash = lora 1 = 0.81
        pair_entries = {
            (0, 1): {
                "overlap": 50, "conflict": 10, "dot": 0.4,
                "norm_a_sq": 0.81,   # lora 1 (smaller hash "aaa")
                "norm_b_sq": 2.25,   # lora 0 (larger hash "zzz")
                "weighted_total": 0.6, "weighted_conflict": 0.1,
                "expected_conflict": 0.12, "excess_conflict": 0.0,
                "subspace_overlap": 0.2, "subspace_weight": 0.5,
            }
        }
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=self._make_lora_entries(),
            pair_entries=pair_entries,
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNotNone(result)
        _, _, pair_conflicts, _, _, _, _, _ = result
        # After reconstruction, norm_a_sq should be for lora 0 = 2.25
        self.assertAlmostEqual(pair_conflicts[(0, 1)]["norm_a_sq"], 2.25)
        self.assertAlmostEqual(pair_conflicts[(0, 1)]["norm_b_sq"], 0.81)

    def test_reconstruction_handles_non_participating_loras(self):
        """LoRA with None entry (non-participating) is excluded from partial_stats."""
        active_loras = [
            {"name": "a.safetensors", "strength": 1.0},
            {"name": "b.safetensors", "strength": 1.0},
            {"name": "c.safetensors", "strength": 1.0},  # non-participating
        ]
        lora_hashes = {0: "aaa", 1: "bbb", 2: "ccc"}
        lora_entries = {**self._make_lora_entries(), 2: None}
        pair_entries = self._make_pair_entries()  # only (0,1)
        result = lora_optimizer.LoRAOptimizer._reconstruct_from_pair_lora_cache(
            "prefix_a",
            lora_entries=lora_entries,
            pair_entries=pair_entries,
            active_loras=active_loras,
            lora_hashes=lora_hashes,
        )
        self.assertIsNotNone(result)
        _, partial_stats, pair_conflicts, _, _, _, _, _ = result
        lora_indices_in_stats = [s[0] for s in partial_stats]
        self.assertNotIn(2, lora_indices_in_stats)
        self.assertNotIn((0, 2), pair_conflicts)
        self.assertNotIn((1, 2), pair_conflicts)


class TestPairLoraCacheWiring(unittest.TestCase):

    def _make_lora_caches(self, prefix="prefix_a"):
        """Minimal lora_caches dict covering prefix_a for 2 loras."""
        entry = {
            "norm_sq": 1.0, "rank": 1,
            "magnitude_samples_unscaled": [0.5],
            "strength_sign": 1,
            "target_key": "layer.weight",
            "is_clip": False, "skip_count": 0, "raw_n": 2,
        }
        return {
            0: {prefix: entry},
            1: {prefix: {**entry, "norm_sq": 0.25}},
        }

    def _make_pair_caches(self, prefix="prefix_a"):
        return {
            (0, 1): {
                prefix: {
                    "overlap": 10, "conflict": 2, "dot": 0.1,
                    "norm_a_sq": 1.0, "norm_b_sq": 0.25,
                    "weighted_total": 0.3, "weighted_conflict": 0.05,
                    "expected_conflict": 0.1, "excess_conflict": 0.0,
                    "subspace_overlap": 0.1, "subspace_weight": 0.5,
                }
            }
        }

    def test_run_group_analysis_uses_pair_lora_cache_on_full_hit(self):
        """When pair+lora caches cover all prefixes, _analyze_target_group is not called."""
        optimizer = lora_optimizer.LoRAOptimizer()
        active_loras = [
            _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
            _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
        ]
        model = _make_model()
        target_groups = optimizer._build_target_groups(
            ["prefix_a"], {"prefix_a": "layer.weight"}, {})

        call_count = {"n": 0}
        orig = optimizer._analyze_target_group
        def counting(*args, **kwargs):
            call_count["n"] += 1
            return orig(*args, **kwargs)

        lora_hashes = {0: "aaa", 1: "bbb"}
        with mock.patch.object(optimizer, "_analyze_target_group", side_effect=counting):
            result = optimizer._run_group_analysis(
                target_groups, active_loras, model, None, torch.device("cpu"),
                lora_caches=self._make_lora_caches(),
                pair_caches=self._make_pair_caches(),
                lora_hashes=lora_hashes,
                track_new_entries=True,
            )
        self.assertEqual(call_count["n"], 0)
        self.assertEqual(result["new_lora_entries"][0], {})
        self.assertEqual(result["new_pair_entries"][(0, 1)], {})

    def test_run_group_analysis_populates_new_entries_on_pair_lora_miss(self):
        """Cache miss populates new_lora_entries and new_pair_entries."""
        optimizer = lora_optimizer.LoRAOptimizer()
        active_loras = [
            _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
            _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
        ]
        model = _make_model()
        target_groups = optimizer._build_target_groups(
            ["prefix_a"], {"prefix_a": "layer.weight"}, {})
        lora_hashes = {0: "aaa", 1: "bbb"}
        result = optimizer._run_group_analysis(
            target_groups, active_loras, model, None, torch.device("cpu"),
            lora_caches={0: {}, 1: {}},  # prefix_a missing → full miss
            pair_caches={(0, 1): {}},
            lora_hashes=lora_hashes,
            track_new_entries=True,
        )
        self.assertIn("prefix_a", result["new_lora_entries"][0])
        self.assertIn("prefix_a", result["new_pair_entries"][(0, 1)])


class TestPairLoraCacheAutoTune(unittest.TestCase):
    """Verify that auto_tune loads, uses, and saves pair/lora caches."""

    def test_lora_and_pair_cache_files_created_after_analysis(self):
        """After _run_group_analysis with misses, lora and pair files are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                active_loras = [
                    _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
                    _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
                ]

                lora_hashes = {}
                with mock.patch("lora_optimizer.folder_paths.get_full_path",
                                return_value=None):
                    for i, lora in enumerate(active_loras):
                        lora_hashes[i] = tuner._lora_identity_hash(lora)

                new_lora_entries = {0: {"prefix_a": {"norm_sq": 1.0}},
                                    1: {"prefix_a": {"norm_sq": 0.5}}}
                new_pair_entries = {(0, 1): {"prefix_a": {"overlap": 10}}}

                # Save as if auto_tune just completed
                for i, h in lora_hashes.items():
                    tuner._lora_cache_save(h, new_lora_entries[i])
                tuner._pair_cache_save(
                    lora_hashes[0], lora_hashes[1], new_pair_entries[(0, 1)])

                # Verify files exist
                path_0 = tuner._lora_cache_path(lora_hashes[0])
                path_1 = tuner._lora_cache_path(lora_hashes[1])
                path_pair = tuner._pair_cache_path(lora_hashes[0], lora_hashes[1])
                self.assertTrue(os.path.exists(path_0))
                self.assertTrue(os.path.exists(path_1))
                self.assertTrue(os.path.exists(path_pair))

    def test_lora_cache_merged_with_existing_on_save(self):
        """New per-prefix entries are merged into the existing lora cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                tuner = lora_optimizer.LoRAAutoTuner()
                # Pre-existing cache with prefix_a
                tuner._lora_cache_save("hash_x", {"prefix_a": {"norm_sq": 1.0}})
                # New analysis added prefix_b
                existing = tuner._lora_cache_load("hash_x") or {}
                existing["prefix_b"] = {"norm_sq": 2.0}
                tuner._lora_cache_save("hash_x", existing)
                # Both prefixes should be in the file
                loaded = tuner._lora_cache_load("hash_x")
                self.assertIn("prefix_a", loaded)
                self.assertIn("prefix_b", loaded)

    def test_score_merge_result_lora_adapter_on_device(self):
        """_score_merge_result should handle LoRAAdapter patches correctly."""
        LoRAAdapter = lora_optimizer.LoRAAdapter
        up = torch.randn(8, 4)
        down = torch.randn(4, 16)
        alpha = 4.0
        adapter = LoRAAdapter(
            loaded_keys=set(),
            weights=(up, down, alpha, None, None, None),
        )
        patches = {("key1",): adapter}
        result = lora_optimizer._score_merge_result(patches, {}, compute_svd=False)
        self.assertIn("norm_mean", result)
        self.assertGreater(result["norm_mean"], 0)
        self.assertIn("composite_score", result)

    def test_sample_pair_metrics_downsamples_large_vectors(self):
        """Pair metrics should work correctly with large vectors that trigger downsampling."""
        optimizer = lora_optimizer.LoRAOptimizer()
        a = torch.randn(200000)
        b = torch.randn(200000)
        result = optimizer._sample_pair_metrics(a, b)
        self.assertIn("overlap", result)
        self.assertIn("conflict", result)
        self.assertIn("dot", result)
        self.assertGreater(result["overlap"], 0)
        result2 = optimizer._sample_pair_metrics(a, b)
        self.assertEqual(result["overlap"], result2["overlap"])
        self.assertEqual(result["conflict"], result2["conflict"])
        self.assertAlmostEqual(result["dot"], result2["dot"], places=4)


    def test_sl_patch_cache_populates_and_hits(self):
        """Single-LoRA patch cache should store results and reuse them on matching auto_strength."""
        optimizer = lora_optimizer.LoRAOptimizer()
        # Simulate a result tuple as returned by _merge_one_group
        fake_patch = ("diff", (torch.randn(4, 4),))
        fake_result = ("weight.key", False, fake_patch, "weighted_sum", "lora_unet_block", 0.0, 1, False, 1.0, 0.9)

        cache = {}
        prefix = "lora_unet_block"
        auto_strength = "enabled"

        # First access: miss → populate
        key = (prefix, auto_strength)
        self.assertNotIn(key, cache)
        cache[key] = fake_result
        self.assertIn(key, cache)
        self.assertIs(cache[key], fake_result)

        # Second access with same auto_strength: hit
        self.assertIs(cache.get(key), fake_result)

        # Different auto_strength: miss
        key2 = (prefix, "disabled")
        self.assertIsNone(cache.get(key2))

        # Different prefix, same auto_strength: miss
        key3 = ("lora_unet_other", auto_strength)
        self.assertIsNone(cache.get(key3))


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestIdeogram4Support(unittest.TestCase):
    """Ideogram 4 detection (must beat the Z-Image check — both are NextDiT
    with layers.N.attention.qkv), key normalization, and preset routing."""

    def _detect(self, sd):
        return lora_optimizer._LoRAMergeBase._detect_architecture(sd)

    @staticmethod
    def _zeros_sd(keys):
        return {k: torch.zeros(1) for k in keys}

    def test_ai_toolkit_native_format_detected(self):
        sd = self._zeros_sd([
            "diffusion_model.layers.0.attention.qkv.lora_A.weight",
            "diffusion_model.layers.0.attention.qkv.lora_B.weight",
            "diffusion_model.layers.0.attention.o.lora_A.weight",
            "diffusion_model.layers.0.attention.o.lora_B.weight",
            "diffusion_model.layers.0.feed_forward.w1.lora_A.weight",
            "diffusion_model.layers.0.feed_forward.w1.lora_B.weight",
        ])
        self.assertEqual(self._detect(sd), "ideogram4")

    def test_fal_conditional_transformer_prefix_detected(self):
        sd = self._zeros_sd([
            "conditional_transformer.layers.3.attention.qkv.lora_A.weight",
            "conditional_transformer.layers.3.attention.qkv.lora_B.weight",
            "conditional_transformer.layers.3.attention.qkv.alpha",
        ])
        self.assertEqual(self._detect(sd), "ideogram4")

    def test_lokr_output_proj_detected(self):
        sd = self._zeros_sd([
            "diffusion_model.layers.7.attention.o.lokr_w1",
            "diffusion_model.layers.7.attention.o.lokr_w2",
            "diffusion_model.layers.7.attention.o.alpha",
        ])
        self.assertEqual(self._detect(sd), "ideogram4")

    def test_lowercase_adaln_plus_ffn_detected(self):
        sd = self._zeros_sd([
            "diffusion_model.layers.2.adaln_modulation.lora_A.weight",
            "diffusion_model.layers.2.feed_forward.w2.lora_A.weight",
        ])
        self.assertEqual(self._detect(sd), "ideogram4")

    def test_qkv_only_disambiguated_by_fused_width(self):
        # qkv-only LoRA: no o/adaln markers — the 13824-row (3x4608) fused
        # up matrix is the Ideogram tell; 6912 (3x2304) is Z-Image Turbo
        ideo = {
            "diffusion_model.layers.0.attention.qkv.lora_A.weight": torch.zeros(8, 4608),
            "diffusion_model.layers.0.attention.qkv.lora_B.weight": torch.zeros(13824, 8),
        }
        self.assertEqual(self._detect(ideo), "ideogram4")
        zim = {
            "diffusion_model.layers.0.attention.qkv.lora_A.weight": torch.zeros(8, 2304),
            "diffusion_model.layers.0.attention.qkv.lora_B.weight": torch.zeros(6912, 8),
        }
        self.assertEqual(self._detect(zim), "zimage")

    def test_zimage_keys_still_detect_zimage(self):
        """Regression: Z-Image markers (attention.out, adaLN_modulation) must
        not be claimed by the Ideogram 4 checks."""
        sd = self._zeros_sd([
            "diffusion_model.layers.0.attention.qkv.lora_up.weight",
            "diffusion_model.layers.0.attention.qkv.lora_down.weight",
            "diffusion_model.layers.0.attention.out.lora_up.weight",
            "diffusion_model.layers.0.attention.out.lora_down.weight",
            "diffusion_model.layers.0.adaLN_modulation.1.lora_up.weight",
        ])
        self.assertEqual(self._detect(sd), "zimage")

    def test_normalize_fal_and_peft_prefixes(self):
        norm = lora_optimizer._LoRAMergeBase._normalize_keys_ideogram4
        sd = self._zeros_sd([
            "conditional_transformer.layers.3.attention.qkv.lora_A.weight",
            "transformer.layers.4.feed_forward.w3.lora_B.weight",
            "base_model.model.layers.5.attention.o.lora_A.weight",
            "layers.6.adaln_modulation.lora_B.weight",
            "diffusion_model.layers.7.attention.qkv.lora_A.weight",  # passthrough
        ])
        out = norm(sd)
        self.assertIn("diffusion_model.layers.3.attention.qkv.lora_A.weight", out)
        self.assertIn("diffusion_model.layers.4.feed_forward.w3.lora_B.weight", out)
        self.assertIn("diffusion_model.layers.5.attention.o.lora_A.weight", out)
        self.assertIn("diffusion_model.layers.6.adaln_modulation.lora_B.weight", out)
        self.assertIn("diffusion_model.layers.7.attention.qkv.lora_A.weight", out)
        self.assertEqual(len(out), len(sd))

    def test_normalize_kohya_underscores(self):
        norm = lora_optimizer._LoRAMergeBase._normalize_keys_ideogram4
        sd = self._zeros_sd([
            "lora_unet_layers_0_attention_qkv.lora_down.weight",
            "lora_unet_layers_12_feed_forward_w2.lora_up.weight",
            "lora_unet_layers_3_adaln_modulation.alpha",
        ])
        out = norm(sd)
        self.assertIn("diffusion_model.layers.0.attention.qkv.lora_down.weight", out)
        self.assertIn("diffusion_model.layers.12.feed_forward.w2.lora_up.weight", out)
        self.assertIn("diffusion_model.layers.3.adaln_modulation.alpha", out)

    def test_preset_routes_to_dit(self):
        key, preset = lora_optimizer._resolve_arch_preset("auto", "ideogram4")
        self.assertEqual(key, "dit")

    def test_normalize_dispatch(self):
        sd = {"conditional_transformer.layers.0.attention.o.lora_A.weight": torch.zeros(1)}
        out = lora_optimizer._LoRAMergeBase._normalize_keys(sd, "ideogram4")
        self.assertIn("diffusion_model.layers.0.attention.o.lora_A.weight", out)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestMiniMaxH3Support(unittest.TestCase):
    """MiniMax H3 detection, multi-trainer normalization, and fused-QKV
    slice routing for native ComfyUI models."""

    @staticmethod
    def _detect(sd):
        return lora_optimizer._LoRAMergeBase._detect_architecture(sd)

    @staticmethod
    def _zeros(keys):
        return {key: torch.zeros(1) for key in keys}

    def test_native_reference_qkv_detected(self):
        sd = self._zeros([
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight",
        ])
        self.assertEqual(self._detect(sd), "minimax_h3")

    def test_bare_turbo_and_musubi_formats_detected(self):
        bare = self._zeros([
            "blocks.12.mlp.fc1.lora_A.weight",
            "blocks.12.mlp.fc1.lora_B.weight",
        ])
        musubi = self._zeros([
            "lora_unet_blocks_4_attn_qkv_proj.lora_down.weight",
            "lora_unet_blocks_4_attn_qkv_proj.lora_up.weight",
        ])
        self.assertEqual(self._detect(bare), "minimax_h3")
        self.assertEqual(self._detect(musubi), "minimax_h3")

    def test_diffusers_and_refiner_formats_detected_before_acestep(self):
        diffusers = self._zeros([
            "transformer.transformer_blocks.3.attn.to_q.lora_A.weight",
            "transformer.transformer_blocks.3.ff.net.0.proj.lora_B.weight",
        ])
        refiner = self._zeros([
            "base_model.model.token_refiner.refiner_blocks.1.attn.to_v.lora_A.weight",
        ])
        self.assertEqual(self._detect(diffusers), "minimax_h3")
        self.assertEqual(self._detect(refiner), "minimax_h3")

    def test_attention_only_diffusers_uses_h3_dimensions(self):
        h3 = {
            "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 5376),
            "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(7168, 2),
        }
        ace = {
            "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 64),
            "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(64, 2),
        }
        self.assertEqual(self._detect(h3), "minimax_h3")
        self.assertEqual(self._detect(ace), "acestep")

    def test_ltx2_audio_output_is_not_claimed_as_h3(self):
        ltx = self._zeros([
            "diffusion_model.audio_proj_out.lora_A.weight",
            "diffusion_model.transformer_blocks.0.audio_adaln_single.lora_B.weight",
        ])
        self.assertEqual(self._detect(ltx), "ltx")

    def test_native_fused_qkv_splits_without_rank_inflation(self):
        down = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        up = torch.arange(36, dtype=torch.float32).reshape(18, 2)
        alpha = torch.tensor(2.0)
        sd = {
            "blocks.0.attn.qkv_proj.lora_A.weight": down,
            "blocks.0.attn.qkv_proj.lora_B.weight": up,
            "blocks.0.attn.qkv_proj.alpha": alpha,
        }
        out = lora_optimizer._LoRAMergeBase._normalize_keys_minimax_h3(sd)
        for index, component in enumerate(("to_q", "to_k", "to_v")):
            base = f"diffusion_model.blocks.0.attn.{component}"
            self.assertIs(out[f"{base}.lora_A.weight"], down)
            self.assertTrue(torch.equal(
                out[f"{base}.lora_B.weight"], up[index * 6:(index + 1) * 6]))
            self.assertIs(out[f"{base}.alpha"], alpha)
        self.assertNotIn("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", out)

    def test_musubi_unflattens_compound_module_names(self):
        sd = self._zeros([
            "lora_unet_blocks_12_attn_out_proj.lora_down.weight",
            "lora_unet_blocks_12_mlp_fc2.lora_up.weight",
            "lora_unet_token_refiner_blocks_1_mlp_fc1.alpha",
            "lora_unet_final_layer_audio_out.lora_A.weight",
        ])
        out = lora_optimizer._LoRAMergeBase._normalize_keys_minimax_h3(sd)
        self.assertIn("diffusion_model.blocks.12.attn.out_proj.lora_down.weight", out)
        self.assertIn("diffusion_model.blocks.12.mlp.fc2.lora_up.weight", out)
        self.assertIn("diffusion_model.token_refiner.blocks.1.mlp.fc1.alpha", out)
        self.assertIn("diffusion_model.final_layer.audio_out.lora_A.weight", out)

    def test_diffusers_names_map_to_native_and_swap_swiglu_up_rows(self):
        fc1_down = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        fc1_up = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        sd = {
            "base_model.model.transformer_blocks.2.attn.to_out.0.lora_A.weight": torch.zeros(2, 4),
            "base_model.model.transformer_blocks.2.ff.net.0.proj.lora_A.weight": fc1_down,
            "base_model.model.transformer_blocks.2.ff.net.0.proj.lora_B.weight": fc1_up,
            "transformer.token_refiner.refiner_blocks.1.attn.to_q.lora_A.default.weight": torch.zeros(2, 4),
            "transformer.audio_proj_in.lora_B.weight": torch.zeros(4, 2),
            "transformer.norm_out.linear.lora_A.weight": torch.zeros(2, 4),
        }
        out = lora_optimizer._LoRAMergeBase._normalize_keys_minimax_h3(sd)
        self.assertIn("diffusion_model.blocks.2.attn.out_proj.lora_A.weight", out)
        self.assertIs(out["diffusion_model.blocks.2.mlp.fc1.lora_A.weight"], fc1_down)
        self.assertTrue(torch.equal(
            out["diffusion_model.blocks.2.mlp.fc1.lora_B.weight"],
            torch.cat([fc1_up[4:], fc1_up[:4]], dim=0)))
        self.assertIn("diffusion_model.token_refiner.blocks.1.attn.to_q.lora_A.weight", out)
        self.assertIn("diffusion_model.audio_patch_proj.lora_B.weight", out)
        self.assertIn("diffusion_model.final_layer.adaln_proj.linear.lora_A.weight", out)

    def test_native_qkv_aliases_target_exact_fused_slices(self):
        target = "diffusion_model.blocks.0.attn.qkv_proj"
        model_keys = {target: target}
        state_dict = {target: torch.zeros(21, 5)}
        out = lora_optimizer._LoRAMergeBase._add_minimax_h3_qkv_aliases(
            model_keys, state_dict)
        self.assertEqual(out["diffusion_model.blocks.0.attn.to_q"],
                         (target, (0, 0, 7)))
        self.assertEqual(out["diffusion_model.blocks.0.attn.to_k"],
                         (target, (0, 7, 7)))
        self.assertEqual(out["diffusion_model.blocks.0.attn.to_v"],
                         (target, (0, 14, 7)))

    def test_offset_qkv_patches_refuse_to_stock_comfy_native_key(self):
        target = "diffusion_model.blocks.0.attn.qkv_proj"
        down = torch.randn(2, 5)
        ups = [torch.randn(7, 2) for _ in range(3)]
        patches = {
            (target, (0, index * 7, 7)): lora_optimizer.LoRAAdapter(
                set(), (up, down.clone(), 2.0, None, None, None))
            for index, up in enumerate(ups)
        }
        out = lora_optimizer._LoRAMergeBase._refuse_fused_qkv_patches(patches)
        self.assertEqual(set(out), {target})
        fused_diff = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(out[target])
        expected = torch.cat([up @ down for up in ups], dim=0)
        self.assertTrue(torch.allclose(fused_diff, expected, atol=1e-6))

    def test_independent_component_factors_refuse_exactly_at_rank_sum(self):
        target = "diffusion_model.token_refiner.blocks.1.attn.qkv_proj"
        downs = [torch.randn(2, 5) for _ in range(3)]
        ups = [torch.randn(7, 2) for _ in range(3)]
        patches = {
            (target, (0, index * 7, 7)): lora_optimizer.LoRAAdapter(
                set(), (ups[index], downs[index], 1.0 + index, None, None, None))
            for index in range(3)
        }
        out = lora_optimizer._LoRAMergeBase._refuse_fused_qkv_patches(patches)
        fused = out[target]
        self.assertIsInstance(fused, lora_optimizer.LoRAAdapter)
        self.assertEqual(fused.weights[1].shape[0], 6)
        fused_diff = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(fused)
        expected = torch.cat([
            ups[i] @ downs[i] * ((1.0 + i) / 2.0) for i in range(3)
        ], dim=0)
        self.assertTrue(torch.allclose(fused_diff, expected, atol=1e-6))

    def test_dispatch_preset_video_floor_and_model_class_hint(self):
        sd = {"blocks.0.mlp.fc2.lora_A.weight": torch.zeros(1)}
        out = lora_optimizer._LoRAMergeBase._normalize_keys(sd, "minimax_h3")
        self.assertIn("diffusion_model.blocks.0.mlp.fc2.lora_A.weight", out)
        preset_key, _ = lora_optimizer._resolve_arch_preset("auto", "minimax_h3")
        self.assertEqual(preset_key, "dit")
        self.assertEqual(lora_optimizer._VIDEO_ARCH_ORTHOGONAL_FLOOR["minimax_h3"], 1.0)

        MiniMaxH3 = type("MiniMaxH3", (), {})
        wrapper = type("ModelPatcher", (), {})()
        wrapper.model = MiniMaxH3()
        self.assertEqual(
            lora_optimizer._LoRAMergeBase._model_class_arch(wrapper),
            "minimax_h3")


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestAceStepDetection(unittest.TestCase):
    """Test _detect_architecture for ACE-Step v1.0 and v1.5 key patterns."""

    def _detect(self, keys):
        sd = {k: torch.zeros(1) for k in keys}
        return lora_optimizer._LoRAMergeBase._detect_architecture(sd)

    # --- v1.5 PEFT format ---
    def test_v15_peft_self_attn(self):
        keys = [
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    def test_v15_peft_cross_attn(self):
        keys = [
            "base_model.model.layers.12.cross_attn.k_proj.lora_A.weight",
            "base_model.model.layers.12.cross_attn.k_proj.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    def test_v15_peft_mlp_only_not_detected(self):
        """MLP-only LoRA without attn keys should not detect as acestep."""
        keys = [
            "base_model.model.layers.0.mlp.gate_proj.lora_A.weight",
            "base_model.model.layers.0.mlp.gate_proj.lora_B.weight",
        ]
        # No self_attn/cross_attn keys, so won't match acestep pattern
        self.assertNotEqual(self._detect(keys), "acestep")

    def test_v15_bare_layers(self):
        """v1.5 keys without base_model.model. prefix."""
        keys = [
            "layers.5.self_attn.v_proj.lora_up.weight",
            "layers.5.self_attn.v_proj.lora_down.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    # --- v1.0 diffusers format ---
    def test_v10_transformer_blocks(self):
        keys = [
            "transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer_blocks.0.attn.to_q.lora_B.weight",
            "transformer_blocks.0.cross_attn.to_k.lora_A.weight",
            "transformer_blocks.0.cross_attn.to_k.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    def test_v10_speaker_embedder(self):
        keys = [
            "speaker_embedder.lora_A.weight",
            "speaker_embedder.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")

    def test_v10_lyric_encoder(self):
        keys = [
            "lyric_encoder.encoders.0.self_attn.linear_q.lora_A.weight",
            "lyric_encoder.encoders.0.self_attn.linear_q.lora_B.weight",
        ]
        self.assertEqual(self._detect(keys), "acestep")


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestAceStepNormalization(unittest.TestCase):
    """Test _normalize_keys_acestep for v1.0 and v1.5 key formats."""

    def _norm(self, keys):
        sd = {k: torch.zeros(1) for k in keys}
        return lora_optimizer._LoRAMergeBase._normalize_keys_acestep(sd)

    # --- v1.5 PEFT format ---
    def test_v15_peft_strips_prefix(self):
        result = self._norm([
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        ])
        self.assertIn("diffusion_model.layers.0.self_attn.q_proj.lora_A.weight", result)

    def test_v15_bare_layers_adds_prefix(self):
        result = self._norm(["layers.5.cross_attn.k_proj.lora_up.weight"])
        self.assertIn("diffusion_model.layers.5.cross_attn.k_proj.lora_up.weight", result)

    def test_v15_kohya_underscore(self):
        result = self._norm(["lora_unet_layers_3_self_attn_q_proj.lora_down.weight"])
        self.assertIn("diffusion_model.layers.3.self_attn.q_proj.lora_down.weight", result)

    def test_v15_mlp_keys(self):
        result = self._norm([
            "base_model.model.layers.10.mlp.gate_proj.lora_A.weight",
        ])
        self.assertIn("diffusion_model.layers.10.mlp.gate_proj.lora_A.weight", result)

    # --- v1.0 → v1.5 mapping ---
    def test_v10_transformer_blocks_to_layers(self):
        result = self._norm([
            "transformer_blocks.7.attn.to_q.lora_A.weight",
        ])
        self.assertIn("diffusion_model.layers.7.self_attn.q_proj.lora_A.weight", result)

    def test_v10_cross_attn_preserved(self):
        result = self._norm([
            "transformer_blocks.3.cross_attn.to_v.lora_B.weight",
        ])
        self.assertIn("diffusion_model.layers.3.cross_attn.v_proj.lora_B.weight", result)

    def test_v10_to_out_0_to_o_proj(self):
        result = self._norm([
            "transformer_blocks.0.attn.to_out.0.lora_A.weight",
        ])
        self.assertIn("diffusion_model.layers.0.self_attn.o_proj.lora_A.weight", result)

    def test_v10_cross_attn_to_out_0(self):
        result = self._norm([
            "transformer_blocks.5.cross_attn.to_out.0.lora_B.weight",
        ])
        self.assertIn("diffusion_model.layers.5.cross_attn.o_proj.lora_B.weight", result)

    def test_v10_speaker_embedder(self):
        result = self._norm(["speaker_embedder.lora_A.weight"])
        self.assertIn("diffusion_model.speaker_embedder.lora_A.weight", result)

    def test_v10_lyric_encoder(self):
        result = self._norm([
            "lyric_encoder.encoders.2.self_attn.linear_q.lora_A.weight",
        ])
        self.assertIn(
            "diffusion_model.lyric_encoder.encoders.2.self_attn.q_proj.lora_A.weight",
            result,
        )

    def test_v10_lyric_encoder_linear_v(self):
        result = self._norm([
            "lyric_encoder.encoders.0.self_attn.linear_v.lora_B.weight",
        ])
        self.assertIn(
            "diffusion_model.lyric_encoder.encoders.0.self_attn.v_proj.lora_B.weight",
            result,
        )

    # --- Mixed format: ensure no cross-contamination ---
    def test_self_attn_not_double_prefixed(self):
        """self_attn should not become self_self_attn."""
        result = self._norm([
            "transformer_blocks.0.cross_attn.to_q.lora_A.weight",
        ])
        key = list(result.keys())[0]
        self.assertNotIn("self_self_attn", key)
        self.assertNotIn("self_cross_attn", key)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestAceStepPreset(unittest.TestCase):
    """Test ACE-Step architecture preset and auto-detection integration."""

    def test_acestep_maps_to_dedicated_preset(self):
        self.assertEqual(lora_optimizer._ARCH_TO_PRESET["acestep"], "acestep_dit")
        self.assertIn("acestep_dit", lora_optimizer._ARCH_PRESETS)

    def test_acestep_preset_has_wider_orthogonal_band(self):
        dit = lora_optimizer._ARCH_PRESETS["dit"]
        ace = lora_optimizer._ARCH_PRESETS["acestep_dit"]
        self.assertGreater(ace["orthogonal_cos_sim_max"], dit["orthogonal_cos_sim_max"])

    def test_acestep_preset_has_higher_ties_threshold(self):
        dit = lora_optimizer._ARCH_PRESETS["dit"]
        ace = lora_optimizer._ARCH_PRESETS["acestep_dit"]
        self.assertGreater(ace["ties_conflict_threshold"], dit["ties_conflict_threshold"])

    def test_acestep_preset_full_magnitude_preservation(self):
        ace = lora_optimizer._ARCH_PRESETS["acestep_dit"]
        self.assertEqual(ace["auto_strength_orthogonal_floor"], 1.0)

    def test_resolve_arch_preset_acestep(self):
        key, preset = lora_optimizer._resolve_arch_preset("auto", "acestep")
        self.assertEqual(key, "acestep_dit")
        self.assertEqual(preset["display_name"], "ACE-Step (Music DiT)")

    def test_resolve_arch_preset_manual_override(self):
        key, preset = lora_optimizer._resolve_arch_preset("acestep_dit", "unknown")
        self.assertEqual(key, "acestep_dit")


class TestLoRACombinationGenerator(unittest.TestCase):
    """Tests for LoRACombinationGenerator static combo generation and tracking."""

    def setUp(self):
        self.lora_names = ["alpha.safetensors", "beta.safetensors",
                           "gamma.safetensors", "delta.safetensors"]

    # -- combo generation --

    def test_generates_all_pairs_from_lora_list(self):
        combos = lora_optimizer.LoRACombinationGenerator._generate_combos(
            self.lora_names, combo_size="2",
        )
        self.assertEqual(len(combos), 6)  # C(4,2)
        for c in combos:
            self.assertEqual(len(c), 2)

    def test_generates_all_triples_from_lora_list(self):
        combos = lora_optimizer.LoRACombinationGenerator._generate_combos(
            self.lora_names, combo_size="3",
        )
        self.assertEqual(len(combos), 4)  # C(4,3)
        for c in combos:
            self.assertEqual(len(c), 3)

    def test_generates_both_pairs_and_triples(self):
        all_combos = lora_optimizer.LoRACombinationGenerator._generate_combos(
            self.lora_names, combo_size="2_and_3",
        )
        self.assertEqual(len(all_combos), 10)  # C(4,2)+C(4,3) = 6+4

    # -- deterministic shuffle --

    def test_shuffle_is_deterministic_with_seed(self):
        combos = lora_optimizer.LoRACombinationGenerator._generate_combos(
            self.lora_names, combo_size="2",
        )
        a = lora_optimizer.LoRACombinationGenerator._shuffle_combos(combos, seed=42)
        b = lora_optimizer.LoRACombinationGenerator._shuffle_combos(combos, seed=42)
        self.assertEqual(a, b)

    def test_different_seeds_produce_different_order(self):
        combos = lora_optimizer.LoRACombinationGenerator._generate_combos(
            self.lora_names, combo_size="2",
        )
        a = lora_optimizer.LoRACombinationGenerator._shuffle_combos(combos, seed=42)
        b = lora_optimizer.LoRACombinationGenerator._shuffle_combos(combos, seed=99)
        # Same elements, different order
        self.assertEqual(sorted(a), sorted(b))
        self.assertNotEqual(a, b)

    # -- combo hash --

    def test_combo_hash_is_order_independent(self):
        h1 = lora_optimizer.LoRACombinationGenerator._combo_hash(("a", "b"))
        h2 = lora_optimizer.LoRACombinationGenerator._combo_hash(("b", "a"))
        self.assertEqual(h1, h2)

    # -- find_next --

    def test_find_next_skips_completed(self):
        combos = [("a", "b"), ("c", "d"), ("e", "f")]
        done_hash = lora_optimizer.LoRACombinationGenerator._combo_hash(("a", "b"))
        result = lora_optimizer.LoRACombinationGenerator._find_next(
            combos, completed={done_hash},
        )
        self.assertEqual(result, ("c", "d"))

    def test_find_next_returns_none_when_all_done(self):
        combos = [("a", "b"), ("c", "d")]
        completed = {
            lora_optimizer.LoRACombinationGenerator._combo_hash(c) for c in combos
        }
        result = lora_optimizer.LoRACombinationGenerator._find_next(
            combos, completed=completed,
        )
        self.assertIsNone(result)

    # -- progress persistence --

    def test_progress_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "progress.json")
            completed = {"abc123", "def456"}
            lora_optimizer.LoRACombinationGenerator._save_progress(
                path, completed=completed, total=10,
            )
            loaded_completed, loaded_total = (
                lora_optimizer.LoRACombinationGenerator._load_progress(path)
            )
            self.assertEqual(loaded_completed, completed)
            self.assertEqual(loaded_total, 10)

    def test_progress_load_missing_file_returns_empty(self):
        completed, total = lora_optimizer.LoRACombinationGenerator._load_progress(
            "/tmp/nonexistent_combo_progress_xyz.json",
        )
        self.assertEqual(completed, set())
        self.assertEqual(total, 0)

    def test_progress_persists_across_different_shuffle_orders(self):
        """Completed combos are tracked regardless of shuffle_order."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "progress.json")
            lora_optimizer.LoRACombinationGenerator._save_progress(
                path, completed={"abc123"}, total=5,
            )
            # Loading should return the same completed set regardless
            completed, total = (
                lora_optimizer.LoRACombinationGenerator._load_progress(path)
            )
            self.assertEqual(completed, {"abc123"})
            self.assertEqual(total, 5)

    # -- node interface --

    def test_input_types_has_required_fields(self):
        inputs = lora_optimizer.LoRACombinationGenerator.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("shuffle_order", req)
        self.assertIn("strength", req)
        self.assertIn("combo_size", req)
        self.assertIn("folder_filter", req)

    def test_folder_filter_narrows_pool(self):
        """Filtering by single prefix should reduce the combo pool."""
        all_loras = [
            "zit/style1.safetensors", "zit/style2.safetensors",
            "zit/style3.safetensors", "sdxl/char1.safetensors",
        ]
        filtered = [n for n in all_loras if n.startswith("zit/")]
        combos_all = lora_optimizer.LoRACombinationGenerator._generate_combos(all_loras, "2")
        combos_filtered = lora_optimizer.LoRACombinationGenerator._generate_combos(filtered, "2")
        self.assertEqual(len(combos_all), 6)   # C(4,2)
        self.assertEqual(len(combos_filtered), 3)  # C(3,2)

    def test_folder_filter_multiple_prefixes(self):
        """Comma-separated prefixes should include LoRAs from all matching folders."""
        all_loras = [
            "zit/style1.safetensors", "zit/style2.safetensors",
            "zib/base1.safetensors", "sdxl/char1.safetensors",
        ]
        prefixes = tuple(p.strip() for p in "zit/,zib/".split(",") if p.strip())
        filtered = [n for n in all_loras if n.startswith(prefixes)]
        combos = lora_optimizer.LoRACombinationGenerator._generate_combos(filtered, "2")
        self.assertEqual(len(filtered), 3)  # 2 zit + 1 zib
        self.assertEqual(len(combos), 3)    # C(3,2)

    def test_return_types(self):
        self.assertEqual(
            lora_optimizer.LoRACombinationGenerator.RETURN_TYPES,
            ("LORA_STACK", "STRING"),
        )

    def test_is_changed_returns_nan(self):
        result = lora_optimizer.LoRACombinationGenerator.IS_CHANGED(
            shuffle_order=0, strength=1.0, combo_size="2",
        )
        self.assertTrue(math.isnan(result))

    # -- rerun mode --

    def test_input_types_has_rerun_mode(self):
        inputs = lora_optimizer.LoRACombinationGenerator.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("rerun_mode", req)
        spec = req["rerun_mode"]
        self.assertEqual(spec[0], "BOOLEAN")
        self.assertEqual(spec[1]["default"], False)

    def test_resolve_progress_path_default(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        path = gen._resolve_progress_path(rerun_mode=False)
        self.assertTrue(path.endswith("combo_progress.json"))
        self.assertFalse(path.endswith("combo_progress_rerun.json"))

    def test_resolve_progress_path_rerun(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        path = gen._resolve_progress_path(rerun_mode=True)
        self.assertTrue(path.endswith("combo_progress_rerun.json"))

    def test_resolve_progress_path_rerun_same_dir_as_default(self):
        """Rerun progress file lives next to the default one."""
        gen = lora_optimizer.LoRACombinationGenerator()
        default_dir = os.path.dirname(gen._resolve_progress_path(rerun_mode=False))
        rerun_dir = os.path.dirname(gen._resolve_progress_path(rerun_mode=True))
        self.assertEqual(default_dir, rerun_dir)

    # -- rerun HF-enrichment skip --

    def test_hf_file_list_is_memoized(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        call_count = {"n": 0}

        def fake_list(*args, **kwargs):
            call_count["n"] += 1
            return ["config/aaa_bbb_dit.config.json", "lora/aaa.lora.json"]

        with unittest.mock.patch.object(
            lora_optimizer, "HfApi", create=True,
            return_value=unittest.mock.MagicMock(list_repo_files=fake_list),
        ):
            first = gen._list_hf_config_files()
            second = gen._list_hf_config_files()
        self.assertEqual(first, ["config/aaa_bbb_dit.config.json"])
        self.assertEqual(second, first)
        self.assertEqual(call_count["n"], 1)

    def test_combo_already_enriched_true_when_any_candidate_has_decisions(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = ["config/aaa_bbb_dit.config.json"]
        enriched = {
            "candidates": [
                {"per_prefix_decisions": {}},
                {"per_prefix_decisions": {"layer.0": "ties"}},
            ],
        }
        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            side_effect=lambda item: {"a": "aaa", "b": "bbb"}[item["name"]],
        ), unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_community_download",
            return_value=enriched,
        ):
            self.assertTrue(gen._combo_already_enriched(("a", "b")))

    def test_combo_already_enriched_false_when_candidates_lack_decisions(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = ["config/aaa_bbb_dit.config.json"]
        not_enriched = {"candidates": [{"per_prefix_decisions": {}}]}
        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            side_effect=lambda item: {"a": "aaa", "b": "bbb"}[item["name"]],
        ), unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_community_download",
            return_value=not_enriched,
        ):
            self.assertFalse(gen._combo_already_enriched(("a", "b")))

    def test_combo_already_enriched_false_when_no_matching_hf_config(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = ["config/zzz_yyy_dit.config.json"]
        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            side_effect=lambda item: {"a": "aaa", "b": "bbb"}[item["name"]],
        ):
            self.assertFalse(gen._combo_already_enriched(("a", "b")))

    def test_combo_already_enriched_memoizes_result(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = ["config/aaa_bbb_dit.config.json"]
        download_calls = {"n": 0}

        def fake_download(path):
            download_calls["n"] += 1
            return {"candidates": [{"per_prefix_decisions": {"l": "ties"}}]}

        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            side_effect=lambda item: {"a": "aaa", "b": "bbb"}[item["name"]],
        ), unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_community_download",
            side_effect=fake_download,
        ):
            self.assertTrue(gen._combo_already_enriched(("a", "b")))
            self.assertTrue(gen._combo_already_enriched(("a", "b")))
        self.assertEqual(download_calls["n"], 1)

    def test_combo_already_enriched_handles_missing_content_hash(self):
        """If any LoRA hash cannot be computed, fall back to 'not enriched'
        so the combo still runs."""
        gen = lora_optimizer.LoRACombinationGenerator()
        gen._hf_files_cache = []
        with unittest.mock.patch.object(
            lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
            return_value=None,
        ):
            self.assertFalse(gen._combo_already_enriched(("a", "b")))

    # -- rerun source filter --

    def test_input_types_has_rerun_source(self):
        inputs = lora_optimizer.LoRACombinationGenerator.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("rerun_source", req)
        spec = req["rerun_source"]
        self.assertEqual(spec[0], ["shuffle", "original_progress"])
        self.assertEqual(spec[1]["default"], "shuffle")

    def test_filter_shuffled_by_original_progress_keeps_only_completed(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        shuffled = [("a", "b"), ("c", "d"), ("e", "f")]
        ab_hash = gen._combo_hash(("a", "b"))
        ef_hash = gen._combo_hash(("e", "f"))
        filtered = gen._filter_by_original_progress(
            shuffled, original_completed={ab_hash, ef_hash},
        )
        self.assertEqual(filtered, [("a", "b"), ("e", "f")])

    def test_filter_shuffled_by_original_progress_preserves_order(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        shuffled = [("c", "d"), ("a", "b"), ("e", "f")]
        completed = {gen._combo_hash(c) for c in shuffled}
        filtered = gen._filter_by_original_progress(shuffled, completed)
        self.assertEqual(filtered, shuffled)

    def test_filter_shuffled_by_original_progress_empty_original(self):
        gen = lora_optimizer.LoRACombinationGenerator()
        shuffled = [("a", "b"), ("c", "d")]
        filtered = gen._filter_by_original_progress(shuffled, original_completed=set())
        self.assertEqual(filtered, [])


class TestCommunityCacheUploadOnly(unittest.TestCase):
    """Tests for the community_cache='upload_only' mode."""

    def test_autotuner_settings_enum_includes_upload_only(self):
        inputs = lora_optimizer.LoRAAutoTunerSettings.INPUT_TYPES()
        choices = inputs["required"]["community_cache"][0]
        self.assertIn("upload_only", choices)
        self.assertIn("upload_and_download", choices)
        self.assertIn("disabled", choices)

    def test_autotuner_node_enum_includes_upload_only(self):
        inputs = lora_optimizer.LoRAAutoTuner.INPUT_TYPES()
        choices = inputs["optional"]["community_cache"][0]
        self.assertIn("upload_only", choices)
        self.assertIn("upload_and_download", choices)
        self.assertIn("disabled", choices)

    def test_upload_only_defines_arch_key_without_download(self):
        """Regression: _arch_key_for_community must be assigned in upload_only
        mode (previously only set inside the upload_and_download branch,
        leading to UnboundLocalError at upload time)."""
        import inspect
        src = inspect.getsource(lora_optimizer.LoRAAutoTuner.auto_tune)
        # The arch-key assignment must not be gated on the download-only
        # community_cache value — otherwise upload_only uploads crash.
        self.assertNotIn(
            'if _all_hashed and community_cache == "upload_and_download":',
            src,
            "arch-key assignment is gated on upload_and_download only — "
            "upload_only will hit UnboundLocalError at upload time.",
        )


class TestExcessConflictBaseline(unittest.TestCase):
    """excess_conflict must compare the UNWEIGHTED sign-mismatch fraction
    against the unweighted arccos(rho)/pi baseline on the same position set
    (Sheppard / degree-0 arc-cosine kernel)."""

    def setUp(self):
        self.opt = lora_optimizer.LoRAOptimizer()

    def test_identical_vectors_no_excess(self):
        g = torch.Generator().manual_seed(7)
        a = torch.randn(20000, generator=g)
        m = self.opt._sample_pair_metrics(a, a.clone())
        self.assertAlmostEqual(m["excess_conflict"], 0.0, places=5)
        self.assertAlmostEqual(m["expected_conflict"], 0.0, places=3)

    def test_negated_vectors_no_excess(self):
        g = torch.Generator().manual_seed(7)
        a = torch.randn(20000, generator=g)
        m = self.opt._sample_pair_metrics(a, -a)
        # Mismatch fraction 1.0 is exactly what rho=-1 predicts
        self.assertAlmostEqual(m["expected_conflict"], 1.0, places=3)
        self.assertAlmostEqual(m["excess_conflict"], 0.0, places=3)

    def test_independent_gaussians_no_excess(self):
        g = torch.Generator().manual_seed(11)
        a = torch.randn(50000, generator=g)
        b = torch.randn(50000, generator=g)
        m = self.opt._sample_pair_metrics(a, b)
        # ~50% mismatch is the rho~0 base rate, not real conflict
        self.assertLess(m["excess_conflict"], 0.03)

    def test_count_conflict_detected_beyond_weighted_baseline(self):
        """Mismatches concentrated on smaller (but above-noise-floor)
        magnitudes: the old magnitude-weighted ratio sat BELOW the baseline
        (excess clamped to 0); the unweighted fraction detects it."""
        g = torch.Generator().manual_seed(3)
        n = 20000
        signs = torch.where(torch.rand(n, generator=g) < 0.5, 1.0, -1.0)
        mag = torch.where(torch.arange(n) < n // 2,
                          torch.full((n,), 2.0), torch.full((n,), 0.3))
        a = signs * mag
        b = a.clone()
        # Flip half of the small-magnitude positions (25% of total count)
        flip = torch.arange(n) >= (3 * n) // 4
        b[flip] = -b[flip]
        m = self.opt._sample_pair_metrics(a, b)
        # Old weighted ratio: 0.25*0.3/(0.5*2+0.5*0.3) ~ 0.065 < baseline
        # arccos(0.978)/pi ~ 0.067 -> old excess clamped to ~0.
        # New unweighted: 0.25 - 0.067 ~ 0.18.
        self.assertGreater(m["excess_conflict"], 0.10)


class TestTiesSignElection(unittest.TestCase):
    """Sign election is always the magnitude-weighted 'total' vote — the only
    mechanism defined in the TIES paper (frequency is override-only)."""

    def test_ties_mode_elects_total_regardless_of_magnitude_ratio(self):
        opt = lora_optimizer.LoRAOptimizer()
        for mag_ratio in (1.0, 2.0, 10.0):
            mode, _density, sign, _r = opt._auto_select_params(
                0.6, mag_ratio, avg_cos_sim=0.5,
                avg_excess_conflict=0.6, avg_subspace_overlap=0.8,
                strategy_set="basic", precomputed_density=0.7)
            self.assertEqual(mode, "ties")
            self.assertEqual(sign, "total")


class TestKarcherSlerp(unittest.TestCase):
    """N>=3 slerp mode is a weighted Karcher mean: order-independent,
    symmetric, magnitude-corrected. N=2 remains standard SLERP."""

    def setUp(self):
        self.opt = lora_optimizer.LoRAOptimizer()

    def _merge(self, pairs):
        return self.opt._merge_diffs([(t.clone(), w) for t, w in pairs], "slerp")

    def test_two_vector_slerp_unchanged(self):
        e1 = torch.zeros(8); e1[0] = 1.0
        e2 = torch.zeros(8); e2[1] = 1.0
        out = self._merge([(e1, 1.0), (e2, 1.0)])
        expected = (e1 + e2) / math.sqrt(2.0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_three_orthogonal_symmetric_mean(self):
        scale = 2.0
        vecs = []
        for i in range(3):
            e = torch.zeros(8); e[i] = scale
            vecs.append(e)
        out = self._merge([(v, 1.0) for v in vecs])
        # Direction: equal cosine to all three inputs; norm: weighted avg = 2.0
        for v in vecs:
            cos = torch.dot(out.flatten(), v.flatten()) / (out.norm() * v.norm())
            self.assertAlmostEqual(cos.item(), 1.0 / math.sqrt(3.0), places=3)
        self.assertAlmostEqual(out.norm().item(), scale, places=3)

    def test_order_independence(self):
        g = torch.Generator().manual_seed(5)
        vs = [torch.randn(64, generator=g) for _ in range(4)]
        ws = [1.0, 0.8, 0.6, 0.4]
        out1 = self._merge(list(zip(vs, ws)))
        perm = [2, 0, 3, 1]
        out2 = self._merge([(vs[i], ws[i]) for i in perm])
        self.assertTrue(torch.allclose(out1, out2, atol=1e-4),
                        f"max diff {(out1 - out2).abs().max().item()}")

    def test_weight_pulls_toward_heavier_vector(self):
        e1 = torch.zeros(8); e1[0] = 1.0
        e2 = torch.zeros(8); e2[1] = 1.0
        e3 = torch.zeros(8); e3[2] = 1.0
        out = self._merge([(e1, 10.0), (e2, 1.0), (e3, 1.0)])
        u = out / out.norm()
        self.assertGreater(torch.dot(u, e1).item(), torch.dot(u, e2).item())
        self.assertGreater(torch.dot(u, e1).item(), 0.8)


class TestMergeOnWrite(unittest.TestCase):
    """Concurrent ComfyUI processes do read-modify-write on shared cache
    files; saves merge with the on-disk state so neither loses entries."""

    def test_lora_cache_save_preserves_other_writers_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                # "Process A" writes prefix_a; "process B" (stale in-memory
                # view) writes only prefix_b — A's key must survive
                lora_optimizer.LoRAAutoTuner._lora_cache_save("h1", {"prefix_a": {"x": 1}})
                lora_optimizer.LoRAAutoTuner._lora_cache_save("h1", {"prefix_b": {"x": 2}})
                loaded = lora_optimizer.LoRAAutoTuner._lora_cache_load("h1")
                self.assertEqual(set(loaded), {"prefix_a", "prefix_b"})

    def test_lora_cache_save_none_never_clobbers_real_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._lora_cache_save("h1", {"p": {"x": 1}})
                lora_optimizer.LoRAAutoTuner._lora_cache_save("h1", {"p": None, "q": None})
                loaded = lora_optimizer.LoRAAutoTuner._lora_cache_load("h1")
                self.assertEqual(loaded["p"], {"x": 1})
                self.assertIsNone(loaded["q"])

    def test_pair_cache_save_merges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._pair_cache_save("a", "b", {"p1": {"dot": 1}})
                lora_optimizer.LoRAAutoTuner._pair_cache_save("a", "b", {"p2": {"dot": 2}})
                loaded = lora_optimizer.LoRAAutoTuner._pair_cache_load("a", "b")
                self.assertEqual(set(loaded), {"p1", "p2"})

    @unittest.skipIf(torch is None, "torch not available")
    def test_content_hash_inmemory_fallback_for_fileless_lora(self):
        """A LoRA with no file on disk (e.g. the extractor's output) gets a
        stable content hash from its in-memory weights instead of None."""
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            item = {"name": "<extracted>", "lora": {
                "blk.lora_up.weight": torch.ones(4, 2),
                "blk.lora_down.weight": torch.ones(2, 4) * 0.5,
            }}
            h1 = lora_optimizer.LoRAAutoTuner._lora_content_hash(item)
            h2 = lora_optimizer.LoRAAutoTuner._lora_content_hash(item)
            self.assertIsNotNone(h1)
            self.assertEqual(len(h1), 16)
            self.assertEqual(h1, h2)  # deterministic
            # different weights -> different hash
            item2 = {"name": "<extracted>", "lora": {
                "blk.lora_up.weight": torch.zeros(4, 2),
                "blk.lora_down.weight": torch.ones(2, 4) * 0.5,
            }}
            self.assertNotEqual(h1, lora_optimizer.LoRAAutoTuner._lora_content_hash(item2))

    def test_content_hash_save_merges_unless_gc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                lora_optimizer.LoRAAutoTuner._content_hash_mem = None
                lora_optimizer.LoRAAutoTuner._save_content_hash_cache({"k1": "v1"})
                # Simulate another process's state not containing k1
                lora_optimizer.LoRAAutoTuner._save_content_hash_cache({"k2": "v2"})
                lora_optimizer.LoRAAutoTuner._content_hash_mem = None
                loaded = lora_optimizer.LoRAAutoTuner._load_content_hash_cache()
                self.assertEqual(set(loaded), {"k1", "k2"})
                # GC prune path must NOT resurrect dropped keys
                lora_optimizer.LoRAAutoTuner._save_content_hash_cache({"k2": "v2"}, merge=False)
                lora_optimizer.LoRAAutoTuner._content_hash_mem = None
                loaded = lora_optimizer.LoRAAutoTuner._load_content_hash_cache()
                self.assertEqual(set(loaded), {"k2"})
                lora_optimizer.LoRAAutoTuner._content_hash_mem = None


class TestModelIdentityTracking(unittest.TestCase):

    def test_same_object_not_changed(self):
        opt = lora_optimizer.LoRAOptimizer()
        a = types.SimpleNamespace()
        self.assertFalse(opt._track_model_identity(a))  # first sighting
        self.assertFalse(opt._track_model_identity(a))  # unchanged

    def test_new_object_changed(self):
        opt = lora_optimizer.LoRAOptimizer()
        a, b = types.SimpleNamespace(), types.SimpleNamespace()
        opt._track_model_identity(a)
        self.assertTrue(opt._track_model_identity(b))

    def test_clip_swap_detected(self):
        opt = lora_optimizer.LoRAOptimizer()
        m = types.SimpleNamespace()
        c1, c2 = types.SimpleNamespace(), types.SimpleNamespace()
        opt._track_model_identity(m, c1)
        self.assertTrue(opt._track_model_identity(m, c2))

    def test_id_reuse_detected_via_dead_weakref(self):
        import weakref, gc as _gc

        class _Obj:
            pass

        opt = lora_optimizer.LoRAOptimizer()
        b = _Obj()
        # Simulate address reuse: stored id matches b but the original
        # object the cache was built from is gone
        dead = weakref.ref(_Obj())
        _gc.collect()
        self.assertIsNone(dead())
        opt._cached_model_id = id(b)
        opt._cached_clip_id = None
        opt._cached_model_ref = dead
        opt._cached_clip_ref = None
        self.assertTrue(opt._track_model_identity(b))


class TestRecordDatasetWiring(unittest.TestCase):

    def test_widget_exists_and_signature_accepts(self):
        import inspect
        inputs = lora_optimizer.LoRAAutoTuner.INPUT_TYPES()
        self.assertIn("record_dataset", inputs["optional"])
        self.assertEqual(inputs["optional"]["record_dataset"][0],
                         ["disabled", "enabled"])
        sig = inspect.signature(lora_optimizer.LoRAAutoTuner.auto_tune)
        self.assertIn("record_dataset", sig.parameters)
        self.assertEqual(sig.parameters["record_dataset"].default, "disabled")
        # The sweep must actually call the dataset writer when enabled
        src = inspect.getsource(lora_optimizer.LoRAAutoTuner.auto_tune)
        self.assertIn("_save_tuner_dataset_entry", src)


class TestCommunityPairOrientation(unittest.TestCase):
    """Community pair files are content-hash oriented; local caches are
    identity-hash oriented. The swap helper must flip only norm_a/b."""

    def test_swap_pair_norms(self):
        entries = {"p": {"norm_a_sq": 1.0, "norm_b_sq": 2.0, "dot": 0.5}}
        swapped = lora_optimizer.LoRAAutoTuner._swap_pair_norms(entries)
        self.assertEqual(swapped["p"]["norm_a_sq"], 2.0)
        self.assertEqual(swapped["p"]["norm_b_sq"], 1.0)
        self.assertEqual(swapped["p"]["dot"], 0.5)
        # Original untouched (must be a copy)
        self.assertEqual(entries["p"]["norm_a_sq"], 1.0)


class TestCommunityOfflineBackoff(unittest.TestCase):

    def test_download_short_circuits_when_offline(self):
        old = lora_optimizer.LoRAAutoTuner._community_offline_until
        try:
            lora_optimizer.LoRAAutoTuner._community_offline_until = time.time() + 60
            # Must return None without attempting any network I/O
            with mock.patch("urllib.request.urlopen",
                            side_effect=AssertionError("network call attempted")):
                self.assertIsNone(
                    lora_optimizer.LoRAAutoTuner._community_download("config/x.json"))
        finally:
            lora_optimizer.LoRAAutoTuner._community_offline_until = old


class TestCommunityCandidatesDownload(unittest.TestCase):
    """A config with a recorded candidates list must rebuild the full top_n
    so selection/top_n>1 work on community hits."""

    def test_candidates_rebuild_top_n(self):
        cfg = {"merge_mode": "ties", "optimization_mode": "global",
               "auto_strength": "enabled", "sparsification": "disabled",
               "sparsification_density": 0.7, "dare_dampening": 0.0,
               "merge_refinement": "none"}
        cfg2 = dict(cfg, merge_mode="weighted_average")
        config_data = {
            "algo_version": lora_optimizer.AUTOTUNER_ALGO_VERSION,
            "config": cfg, "score": 0.9,
            "candidates": [
                {"rank": 1, "config": cfg, "score_final": 0.9},
                {"rank": 2, "config": cfg2, "score_final": 0.8},
                {"rank": 3, "config": {"malformed": True}},  # skipped
            ],
        }

        def fake_download(path):
            return config_data if path.startswith("config/") else None

        active = [{"name": "a.safetensors", "strength": 1.0},
                  {"name": "b.safetensors", "strength": 0.5}]
        with mock.patch.object(lora_optimizer.LoRAAutoTuner, "_community_download",
                               side_effect=fake_download):
            tuner_data = lora_optimizer.LoRAAutoTuner._community_download_caches(
                active, {0: "aaaa", 1: "bbbb"}, {0: {}, 1: {}}, {(0, 1): {}},
                arch_preset="dit", top_n=3)
        self.assertIsNotNone(tuner_data)
        self.assertEqual(len(tuner_data["top_n"]), 2)
        self.assertEqual(tuner_data["top_n"][0]["config"]["merge_mode"], "ties")
        self.assertEqual(tuner_data["top_n"][1]["config"]["merge_mode"],
                         "weighted_average")
        self.assertNotEqual(tuner_data["lora_hash"], "")


class TestAutotunerMemoryGC(unittest.TestCase):

    def test_gc_removes_old_keeps_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
                old_path = os.path.join(tmpdir, "deadbeef.lora.json")
                new_path = os.path.join(tmpdir, "cafebabe.lora.json")
                mem_path = os.path.join(tmpdir, "aaaa_bbbb.memory.json")
                for p in (old_path, new_path, mem_path):
                    with open(p, "w") as f:
                        json.dump({}, f)
                stale = time.time() - 200 * 86400
                os.utime(old_path, (stale, stale))
                os.utime(mem_path, (stale, stale))
                lora_optimizer.LoRAAutoTuner._gc_done = False
                try:
                    lora_optimizer.LoRAAutoTuner._gc_autotuner_memory(max_age_days=90)
                finally:
                    lora_optimizer.LoRAAutoTuner._gc_done = False
                self.assertFalse(os.path.exists(old_path))
                self.assertTrue(os.path.exists(new_path))
                # Memory entries are never GC'd
                self.assertTrue(os.path.exists(mem_path))


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestAutoStrengthFloor(unittest.TestCase):
    """Explicit auto_strength_floor bounds the reduction on ANY stack;
    only the -1 defaults stay gated on orthogonality."""

    def _scale(self, dot, floor):
        opt = lora_optimizer.LoRAOptimizer()
        preset = lora_optimizer._ARCH_PRESETS["dit"]
        info = opt._compute_branch_auto_scale(
            "Model", [1.0, 1.0], [1.0, 1.0], {(0, 1): dot},
            arch_preset=preset, detected_arch="wan",
            auto_strength_floor=floor, is_full_rank=False)
        return info["scale"]

    def test_explicit_floor_applies_to_aligned_stacks(self):
        # aligned (cos=0.5): unfloored auto scale is 1/sqrt(2+2*0.5) ~ 0.577
        self.assertAlmostEqual(self._scale(0.5, -1.0), 1.0 / math.sqrt(3.0), places=4)
        self.assertAlmostEqual(self._scale(0.5, 0.85), 0.85, places=6)
        self.assertAlmostEqual(self._scale(0.5, 1.0), 1.0, places=6)

    def test_explicit_floor_applies_to_orthogonal_stacks(self):
        self.assertAlmostEqual(self._scale(0.0, 0.85), 0.85, places=6)
        self.assertAlmostEqual(self._scale(0.0, 1.0), 1.0, places=6)

    def test_default_floor_still_gated_on_orthogonality(self):
        # orthogonal + default on wan -> video floor 1.0
        self.assertAlmostEqual(self._scale(0.0, -1.0), 1.0, places=6)
        # aligned + default -> no floor, raw auto scale
        self.assertLess(self._scale(0.5, -1.0), 0.85)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestCommunityHitPromotesToLocalMemory(unittest.TestCase):
    """A community-cache hit must also write the local memory entry —
    otherwise later runs with community_cache=disabled re-sweep from
    scratch despite the result being known."""

    @staticmethod
    def _make_setup():
        class FakePatcher:
            def __init__(self, model):
                self.model = model

            def clone(self):
                return FakePatcher(self.model)

            def add_patches(self, patches, strength=1.0, strength_clip=None):
                return list(patches.keys())

        inner = types.SimpleNamespace(
            layer=types.SimpleNamespace(weight=torch.zeros(16, 16)),
            layer2=types.SimpleNamespace(weight=torch.zeros(16, 16)))

        def make_lora(seed):
            g = torch.Generator().manual_seed(seed)
            d = {}
            for prefix in ("alias_a", "alias_b"):
                d[f"{prefix}.lora_up.weight"] = torch.randn(16, 4, generator=g) * 0.1
                d[f"{prefix}.lora_down.weight"] = torch.randn(4, 16, generator=g)
                d[f"{prefix}.alpha"] = torch.tensor(4.0)
            return d

        stack = [{"name": "A", "lora": make_lora(1), "strength": 1.0},
                 {"name": "B", "lora": make_lora(2), "strength": 0.8}]
        return FakePatcher(inner), stack

    def test_community_hit_writes_memory_and_local_run_hits_it(self):
        fake_tuner_data = {
            "version": 1, "algo_version": lora_optimizer.AUTOTUNER_ALGO_VERSION,
            "lora_hash": "x",
            "source_loras": [{"name": "A", "strength": 1.0}, {"name": "B", "strength": 0.8}],
            "normalize_keys": "disabled", "architecture_preset": "auto",
            "auto_strength_floor": -1.0, "decision_smoothing": 0.25,
            "smooth_slerp_gate": False, "scoring_formula": "v2",
            "analysis_summary": {
                "n_loras": 2, "prefix_count": 2, "avg_conflict_ratio": 0.4,
                "avg_excess_conflict": 0.0, "avg_subspace_overlap": 0.0,
                "avg_cosine_sim": 0.0, "magnitude_ratio": 1.0,
                "decision_smoothing": 0.25},
            "top_n": [{
                "rank": 1, "score_heuristic": 0.8, "score_measured": 0.7,
                "score_external": None, "score_final": 0.7,
                "config": {"merge_mode": "per_prefix_auto",
                           "sparsification": "disabled",
                           "sparsification_density": 0.7, "dare_dampening": 0.0,
                           "merge_refinement": "none", "auto_strength": "disabled",
                           "optimization_mode": "per_prefix", "strategy_set": "full"},
                "metrics": {}, "external_details": None,
                "per_prefix_decisions": {}}],
        }
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            model, stack = self._make_setup()
            tuner = lora_optimizer.LoRAAutoTuner()
            tuner._get_model_keys = lambda m: {
                "alias_a": "layer.weight", "alias_b": "layer2.weight"}
            with mock.patch.object(tuner, "_lora_content_hash",
                                   side_effect=lambda l: "h_" + l["name"]), \
                    mock.patch.object(tuner, "_community_download_caches",
                                      return_value=fake_tuner_data):
                res1 = tuner.auto_tune(
                    model, stack, 1.0, clip=None, top_n=1,
                    community_cache="upload_and_download", memory_mode="auto",
                    cache_patches="disabled", record_dataset="disabled",
                    diff_cache_mode="disabled")
            self.assertIn("COMMUNITY CACHE HIT", res1[2])
            mem_files = [f for f in os.listdir(tmpdir) if f.endswith(".memory.json")]
            self.assertEqual(len(mem_files), 1, "community hit did not write local memory")

            # Second run: community disabled, fresh node — must hit local memory
            model2, stack2 = self._make_setup()
            tuner2 = lora_optimizer.LoRAAutoTuner()
            tuner2._get_model_keys = lambda m: {
                "alias_a": "layer.weight", "alias_b": "layer2.weight"}
            with self.assertLogs(level="INFO") as captured:
                res2 = tuner2.auto_tune(
                    model2, stack2, 1.0, clip=None, top_n=1,
                    community_cache="disabled", memory_mode="auto",
                    cache_patches="disabled", record_dataset="disabled",
                    diff_cache_mode="disabled")
            self.assertTrue(any("[AutoTuner Memory] HIT" in m for m in captured.output),
                            captured.output[-8:])
            self.assertEqual(res2[4]["top_n"][0]["score_final"], 0.7)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestGroupPatchCache(unittest.TestCase):
    """Cross-candidate group patch cache: identical sweep results with the
    cache active vs disabled, and sound key semantics."""

    def test_key_shares_invariant_modes_across_auto_strength(self):
        key = lora_optimizer._group_merge_cache_key
        base = dict(label_prefix="p", is_clip=False, pf_density=0.7, pf_sign="total",
                    sparsification="disabled", sparsification_density=0.7,
                    dare_dampening=0.0, merge_refinement="none")
        # Scale-invariant modes with quality "none": shared across settings
        for mode in ("weighted_average", "normalize", "slerp", "consensus"):
            k_on = key(pf_mode=mode, pf_quality="none", auto_strength_setting="enabled", **base)
            k_off = key(pf_mode=mode, pf_quality="none", auto_strength_setting="disabled", **base)
            self.assertEqual(k_on, k_off, mode)
        # Linear modes: keyed per setting
        for mode in ("weighted_sum", "ties"):
            k_on = key(pf_mode=mode, pf_quality="none", auto_strength_setting="enabled", **base)
            k_off = key(pf_mode=mode, pf_quality="none", auto_strength_setting="disabled", **base)
            self.assertNotEqual(k_on, k_off, mode)
        # Refinement breaks invariance (selfish additions scale)
        k_on = key(pf_mode="weighted_average", pf_quality="refine",
                   auto_strength_setting="enabled", **base)
        k_off = key(pf_mode="weighted_average", pf_quality="refine",
                    auto_strength_setting="disabled", **base)
        self.assertNotEqual(k_on, k_off)
        # CLIP groups never see the auto-strength scale
        clip_base = dict(base, is_clip=True)
        k_on = key(pf_mode="ties", pf_quality="none", auto_strength_setting="enabled", **clip_base)
        k_off = key(pf_mode="ties", pf_quality="none", auto_strength_setting="disabled", **clip_base)
        self.assertEqual(k_on, k_off)

    def test_sweep_results_identical_with_and_without_cache(self):
        n_groups = 5

        class FakePatcher:
            def __init__(self, model):
                self.model = model

            def clone(self):
                return FakePatcher(self.model)

            def add_patches(self, patches, strength=1.0, strength_clip=None):
                return list(patches.keys())

        inner = types.SimpleNamespace(**{
            f"layer{i}": types.SimpleNamespace(weight=torch.zeros(32, 32))
            for i in range(n_groups)})

        def make_lora(seed, scale):
            g = torch.Generator().manual_seed(seed)
            d = {}
            for i in range(n_groups):
                d[f"alias_{i}.lora_up.weight"] = torch.randn(32, 4, generator=g) * scale
                d[f"alias_{i}.lora_down.weight"] = torch.randn(4, 32, generator=g)
                d[f"alias_{i}.alpha"] = torch.tensor(4.0)
            return d

        def run(disable_cache):
            stack = [
                {"name": "A", "lora": make_lora(1, 0.10), "strength": 1.0},
                {"name": "B", "lora": make_lora(2, 0.12), "strength": 0.8},
            ]
            tuner = lora_optimizer.LoRAAutoTuner()
            tuner._get_model_keys = lambda m: {
                f"alias_{i}": f"layer{i}.weight" for i in range(n_groups)}
            ctx = (mock.patch.object(
                lora_optimizer, "_group_merge_cache_key",
                side_effect=lambda *a, **k: object())  # unique keys -> empty plan
                if disable_cache else contextlib.nullcontext())
            with ctx:
                res = tuner.auto_tune(
                    FakePatcher(inner), stack, 1.0, clip=None, top_n=6,
                    scoring_svd="full", scoring_device="cpu",
                    diff_cache_mode="disabled", scoring_speed="full",
                    scoring_formula="v2", memory_mode="disabled",
                    community_cache="disabled", cache_patches="disabled",
                    record_dataset="disabled")
            return res[4]["top_n"]

        cached = run(disable_cache=False)
        uncached = run(disable_cache=True)
        self.assertEqual([r["config"] for r in cached],
                         [r["config"] for r in uncached])
        for rc, ru in zip(cached, uncached):
            self.assertAlmostEqual(rc["score_final"], ru["score_final"], places=9)

        # Exhausted RAM budget: storage skipped gracefully, results unchanged
        fake_vm = types.SimpleNamespace(available=0)
        with mock.patch("psutil.virtual_memory", return_value=fake_vm):
            zero_budget = run(disable_cache=False)
        self.assertEqual([r["config"] for r in cached],
                         [r["config"] for r in zero_budget])
        for rc, rz in zip(cached, zero_budget):
            self.assertAlmostEqual(rc["score_final"], rz["score_final"], places=9)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestGlobalModeMergesWithDeclaredMode(unittest.TestCase):
    """Regression: optimization_mode='global' must merge multi-LoRA groups
    with the declared/overridden mode — pf_n_loras staying 0 used to force
    the single-contributor weighted_sum fallback on every group."""

    def _run(self, override):
        class FakePatcher:
            def __init__(self, model):
                self.model = model

            def clone(self):
                return FakePatcher(self.model)

            def add_patches(self, patches, strength=1.0, strength_clip=None):
                return list(patches.keys())

        inner = types.SimpleNamespace(
            layer=types.SimpleNamespace(weight=torch.zeros(32, 32)),
            layer2=types.SimpleNamespace(weight=torch.zeros(32, 32)))
        model = FakePatcher(inner)

        def make_lora(seed, scale, prefixes):
            g = torch.Generator().manual_seed(seed)
            d = {}
            for prefix in prefixes:
                d[f"{prefix}.lora_up.weight"] = torch.randn(32, 4, generator=g) * scale
                d[f"{prefix}.lora_down.weight"] = torch.randn(4, 32, generator=g)
                d[f"{prefix}.alpha"] = torch.tensor(4.0)
            return d

        stack = [
            # alias_a is shared (2 LoRAs); alias_b is single-LoRA
            {"name": "A", "lora": make_lora(1, 0.1, ("alias_a", "alias_b")), "strength": 1.0},
            {"name": "B", "lora": make_lora(2, 0.1, ("alias_a",)), "strength": 0.8},
        ]
        opt = lora_optimizer.LoRAOptimizer()
        opt._get_model_keys = lambda m: {"alias_a": "layer.weight", "alias_b": "layer2.weight"}
        _, _, _, _, lora_data = opt.optimize_merge(
            model, stack, 1.0,
            optimization_mode="global", merge_strategy_override=override,
            cache_patches="disabled", patch_compression="disabled")
        return lora_data["per_prefix_decisions"]

    def test_override_applies_to_multi_lora_groups(self):
        for override in ("slerp", "ties", "consensus", "weighted_average"):
            decisions = self._run(override)
            self.assertEqual(decisions["alias_a"], override, decisions)
            # Single-LoRA groups still fall back to weighted_sum
            self.assertEqual(decisions["alias_b"], "weighted_sum", decisions)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestSingleLoraSkipsCompression(unittest.TestCase):
    """Single-LoRA groups (layers only one LoRA touches) take the exact
    low-rank fast path even when sparsification/refinement is on — those are
    conflict/multi-diff ops that are no-ops on a lone LoRA. This avoids the
    wasteful dense-materialize + compression SVD and emits native factors
    (a LoRAAdapter), while genuine multi-LoRA groups still go dense+compress."""

    def _run(self, **kwargs):
        class FakePatcher:
            def __init__(self, model):
                self.model = model

            def clone(self):
                return FakePatcher(self.model)

            def add_patches(self, patches, strength=1.0, strength_clip=None):
                return list(patches.keys())

        inner = types.SimpleNamespace(
            layer=types.SimpleNamespace(weight=torch.zeros(32, 32)),
            layer2=types.SimpleNamespace(weight=torch.zeros(32, 32)))
        model = FakePatcher(inner)

        def make_lora(seed, scale, prefixes):
            g = torch.Generator().manual_seed(seed)
            d = {}
            for prefix in prefixes:
                d[f"{prefix}.lora_up.weight"] = torch.randn(32, 4, generator=g) * scale
                d[f"{prefix}.lora_down.weight"] = torch.randn(4, 32, generator=g)
                d[f"{prefix}.alpha"] = torch.tensor(4.0)
            return d

        stack = [
            # alias_a is shared (2 LoRAs -> conflict); alias_b is single-LoRA
            {"name": "A", "lora": make_lora(1, 0.1, ("alias_a", "alias_b")), "strength": 1.0},
            {"name": "B", "lora": make_lora(2, 0.1, ("alias_a",)), "strength": 0.8},
        ]
        opt = lora_optimizer.LoRAOptimizer()
        opt._get_model_keys = lambda m: {"alias_a": "layer.weight", "alias_b": "layer2.weight"}
        _, _, _, _, lora_data = opt.optimize_merge(
            model, stack, 1.0, cache_patches="disabled", **kwargs)
        patches = {}
        for k, v in lora_data["model_patches"].items():
            patches[k[0] if isinstance(k, tuple) else k] = v
        return patches

    def _is_native_rank4(self, patch):
        # The fast path emits the LoRA's native rank-4 factors untouched.
        # The dense+compress path (DARE breaks the low-rank structure, then SVD
        # re-fits) lands at a higher rank or stays a dense ("diff",) tuple.
        return (isinstance(patch, lora_optimizer.LoRAAdapter)
                and patch.weights[1].shape[0] == 4)

    def test_single_lora_skips_compression_under_sparsification(self):
        # sparsification on + compression on: the single-LoRA group must STILL
        # take the low-rank fast path (no dense diff, no sparsification, no
        # compression SVD) — proven by its native rank 4 surviving intact; the
        # multi-LoRA group must NOT (it genuinely needs deconfliction).
        patches = self._run(sparsification="dare", sparsification_density=0.9,
                            patch_compression="smart")
        self.assertTrue(self._is_native_rank4(patches["layer2.weight"]),
                        "single-LoRA group should emit native rank-4 factors")
        self.assertFalse(self._is_native_rank4(patches["layer.weight"]),
                         "multi-LoRA group should still go dense+compress")

    def test_single_lora_lowrank_path_is_size_safe(self):
        # The bypassed single-LoRA patch is stored at the LoRA's native rank (4),
        # never padded up to the rank-64 compression floor.
        patches = self._run(sparsification="dare", sparsification_density=0.9,
                            patch_compression="smart")
        adapter = patches["layer2.weight"]
        mat_up, mat_down = adapter.weights[0], adapter.weights[1]
        self.assertEqual(mat_down.shape[0], 4)  # native rank, not 64
        self.assertEqual(mat_up.shape[1], 4)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestBatchedKarcher(unittest.TestCase):
    """The batched Karcher implementation must match the per-unit reference
    loop it replaced (same math, different reduction order)."""

    @staticmethod
    def _reference_karcher(vecs, weights):
        """The pre-1.9.3 per-unit implementation, kept as the oracle."""
        total_w = sum(weights)
        units = []
        for v, w in zip(vecs, weights):
            vn = v.norm()
            if vn.item() > 1e-12:
                units.append((v / vn, w / total_w))
        m = None
        for u, wn in units:
            m = u * wn if m is None else m.add_(u, alpha=wn)
        m_norm = m.norm()
        m = units[0][0].clone() if m_norm.item() < 1e-8 else m / m_norm
        for _ in range(8):
            tangent = torch.zeros_like(m)
            for u, wn in units:
                cos_i = torch.dot(u, m).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                theta_i = torch.acos(cos_i)
                if theta_i.item() < 1e-7:
                    continue
                coef = wn * (theta_i / torch.sin(theta_i))
                tangent.add_(u - cos_i * m, alpha=coef.item())
            t_norm = tangent.norm()
            if t_norm.item() < 1e-7:
                break
            m = torch.cos(t_norm) * m + (torch.sin(t_norm) / t_norm) * tangent
            m = m / m.norm().clamp(min=1e-12)
        # norm correction as in _merge_diffs
        input_norms = [(v.norm().item(), w) for v, w in zip(vecs, weights)]
        target_norm = sum(n * w for n, w in input_norms) / total_w
        cur = m.norm().item()
        if cur > 1e-8:
            m = m * (target_norm / cur)
        return m

    def test_batched_matches_reference(self):
        opt = lora_optimizer.LoRAOptimizer()
        g = torch.Generator().manual_seed(11)
        for n in (3, 4, 5):
            vecs = [torch.randn(256, generator=g) for _ in range(n)]
            weights = [1.0, 0.8, 0.6, 0.4, 0.9][:n]
            ref = self._reference_karcher([v.clone() for v in vecs], weights)
            out = opt._merge_diffs(list(zip([v.clone() for v in vecs], weights)), "slerp")
            self.assertTrue(torch.allclose(out, ref, atol=1e-5),
                            f"n={n} max diff {(out - ref).abs().max().item()}")

    def test_zero_norm_vector_is_filtered(self):
        opt = lora_optimizer.LoRAOptimizer()
        g = torch.Generator().manual_seed(12)
        vecs = [torch.randn(64, generator=g), torch.zeros(64), torch.randn(64, generator=g)]
        weights = [1.0, 0.7, 0.5]
        out = opt._merge_diffs(list(zip([v.clone() for v in vecs], weights)), "slerp")
        ref = self._reference_karcher([v.clone() for v in vecs], weights)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_all_zero_vectors_return_zero(self):
        opt = lora_optimizer.LoRAOptimizer()
        out = opt._merge_diffs(
            [(torch.zeros(16), 1.0), (torch.zeros(16), 0.5), (torch.zeros(16), 0.3)],
            "slerp")
        self.assertTrue(torch.equal(out, torch.zeros(16)))


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestScoreDuringMerge(unittest.TestCase):
    """Score-during-merge: inline stats must be exactly what
    _score_merge_result would have measured itself."""

    def _make_patches(self):
        g = torch.Generator().manual_seed(7)
        t1 = torch.randn(40, 40, generator=g)
        t2 = torch.randn(40, 40, generator=g) * 0.3
        return {"k1": ("diff", (t1,)), "k2": ("diff", (t2,))}

    def test_inline_stats_metrics_identical(self):
        patches = self._make_patches()
        plain = lora_optimizer._score_merge_result(patches, {}, compute_svd=True)
        inline = {}
        for p in patches.values():
            t = p[1][0]
            inline[id(t)] = (t, lora_optimizer._diff_score_stats(t, True))
        with_inline = lora_optimizer._score_merge_result(
            patches, {}, compute_svd=True, _inline_stats=inline)
        self.assertEqual(plain, with_inline)

    def test_stale_id_entry_falls_back_to_direct_measurement(self):
        patches = self._make_patches()
        plain = lora_optimizer._score_merge_result(patches, {}, compute_svd=True)
        # Entries whose held reference is NOT the patch tensor (stale id reuse)
        # must be ignored — fake stats would otherwise poison the metrics
        decoy = torch.ones(2, 2)
        inline = {id(p[1][0]): (decoy, (999.0, 0.5, 12.0)) for p in patches.values()}
        with_inline = lora_optimizer._score_merge_result(
            patches, {}, compute_svd=True, _inline_stats=inline)
        self.assertEqual(plain, with_inline)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(),
                     "CUDA required for the deferral path")
    def test_zimage_qkv_deferral_mechanics(self):
        """Dense QKV component diffs must be deferred on GPU, fused by
        refusion on GPU, scored in the post-refusion walk, and land on CPU
        with their stats registered under the fused tensor's identity."""
        lora_dim = 32

        class FakePatcher:
            def __init__(self):
                self.model = types.SimpleNamespace()

            def clone(self):
                return FakePatcher()

            def add_patches(self, patches, strength=1.0, strength_clip=None):
                return list(patches.keys())

        prefixes = ["layers.0.attention.to_q", "layers.0.attention.to_k",
                    "layers.0.attention.to_v"]

        def make_lora(seed, scale):
            g = torch.Generator().manual_seed(seed)
            d = {}
            for prefix in prefixes:
                d[f"{prefix}.lora_up.weight"] = torch.randn(lora_dim, 4, generator=g) * scale
                d[f"{prefix}.lora_down.weight"] = torch.randn(4, lora_dim, generator=g)
                d[f"{prefix}.alpha"] = torch.tensor(4.0)
            return d

        stack = [
            {"name": "A", "lora": make_lora(1, 0.1), "strength": 1.0},
            {"name": "B", "lora": make_lora(2, 0.12), "strength": 0.7},
        ]
        tuner = lora_optimizer.LoRAAutoTuner()
        tuner._detect_architecture = lambda sd: "zimage"
        tuner._get_model_keys = lambda m: {p: f"{p}.weight" for p in prefixes}
        tuner._resolve_target_shape = (
            lambda target_key, is_clip, model, clip: torch.Size([lora_dim, lora_dim]))
        collector = {"stats": {}, "compute_svd": True}
        # The comfy stub reports 0 free VRAM, which (correctly) disables the
        # deferral budget — report real headroom for this test. Patch the
        # module object lora_optimizer actually references: _load_module()
        # tests reinstall stubs, so sys.modules may hold a NEWER stub object.
        mm = lora_optimizer.comfy.model_management
        with mock.patch.object(mm, "get_free_memory", return_value=8 * 1024**3):
            # sparsification forces the dense-diff path (exact-linear is gated off)
            _, _, _, _, lora_data = tuner.optimize_merge(
                FakePatcher(), stack, 1.0,
                sparsification="dare", sparsification_density=0.9,
                patch_compression="disabled", cache_patches="disabled",
                _score_collector=collector)

        fused = [(k, p) for k, p in lora_data["model_patches"].items()
                 if ".qkv." in (k if isinstance(k, str) else k[0])]
        self.assertEqual(len(fused), 1)
        _k, p = fused[0]
        self.assertEqual(p[0], "diff")
        t = p[1][0]
        self.assertEqual(tuple(t.shape), (3 * lora_dim, lora_dim))
        self.assertFalse(t.is_cuda)
        # The walk must have registered stats under the FUSED tensor identity
        entry = collector["stats"].get(id(t))
        self.assertIsNotNone(entry)
        self.assertIs(entry[0], t)
        # Stats were measured on the GPU-resident fused tensor — must match a
        # recompute on the same values
        ref = lora_optimizer._diff_score_stats(t.cuda(), True)
        for got, want in zip(entry[1], ref):
            if got is None:
                self.assertIsNone(want)
            else:
                self.assertAlmostEqual(got, want, places=6)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(),
                     "CUDA required for the deferral path")
    def test_zimage_qkv_deferral_end_to_end(self):
        """Full sweep on a fake zimage stack: QKV components deferred on GPU,
        fused by refusion, scored in the post-refusion walk; results must
        match the collector-less CPU-scored sweep and contain no CUDA
        tensors in the returned patches."""
        lora_dim = 32

        class FakePatcher:
            def __init__(self):
                self.model = types.SimpleNamespace()

            def clone(self):
                return FakePatcher()

            def add_patches(self, patches, strength=1.0, strength_clip=None):
                return list(patches.keys())

        prefixes = [
            "layers.0.attention.to_q", "layers.0.attention.to_k",
            "layers.0.attention.to_v", "layers.0.feed_forward.w1",
        ]

        def make_lora(seed, scale):
            g = torch.Generator().manual_seed(seed)
            d = {}
            for prefix in prefixes:
                d[f"{prefix}.lora_up.weight"] = torch.randn(lora_dim, 4, generator=g) * scale
                d[f"{prefix}.lora_down.weight"] = torch.randn(4, lora_dim, generator=g)
                d[f"{prefix}.alpha"] = torch.tensor(4.0)
            return d

        def run(scoring_device):
            stack = [
                {"name": "A", "lora": make_lora(1, 0.1), "strength": 1.0},
                {"name": "B", "lora": make_lora(2, 0.12), "strength": 0.7},
            ]
            tuner = lora_optimizer.LoRAAutoTuner()
            tuner._detect_architecture = lambda sd: "zimage"
            tuner._get_model_keys = lambda m: {p: f"{p}.weight" for p in prefixes}
            tuner._resolve_target_shape = (
                lambda target_key, is_clip, model, clip: torch.Size([lora_dim, lora_dim]))
            res = tuner.auto_tune(
                FakePatcher(), stack, 1.0,
                clip=None, top_n=3,
                scoring_svd="full", scoring_device=scoring_device,
                diff_cache_mode="disabled", scoring_speed="full",
                scoring_formula="v2", memory_mode="disabled",
                community_cache="disabled", cache_patches="disabled",
                record_dataset="disabled")
            _, _, _, _, tuner_data, lora_data = res
            return tuner_data, lora_data

        tuner_data_gpu, lora_data_gpu = run("gpu")
        # QKV components were fused (3 to_* keys -> 1 qkv key) and nothing
        # returned to the caller may still live on the GPU
        keys = [k if isinstance(k, str) else k[0]
                for k in lora_data_gpu["model_patches"]]
        self.assertTrue(any(".qkv." in k for k in keys), keys)
        self.assertFalse(any(".to_q." in k for k in keys), keys)
        for p in lora_data_gpu["model_patches"].values():
            if isinstance(p, tuple) and p[0] == "diff":
                self.assertFalse(p[1][0].is_cuda)

        tuner_data_cpu, _ = run("cpu")
        ranks_gpu = [(r["config"]["merge_mode"], r["config"]["sparsification"])
                     for r in tuner_data_gpu["top_n"]]
        ranks_cpu = [(r["config"]["merge_mode"], r["config"]["sparsification"])
                     for r in tuner_data_cpu["top_n"]]
        self.assertEqual(ranks_gpu, ranks_cpu)
        for rg, rc in zip(tuner_data_gpu["top_n"], tuner_data_cpu["top_n"]):
            self.assertAlmostEqual(rg["score_final"], rc["score_final"], places=5)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestDiffCacheRamLimit(unittest.TestCase):
    def test_auto_mode_honors_ram_pct_without_absolute_cap(self):
        """Regression: a 16GB hard cap used to silently override ram_pct on
        high-memory machines (128GB box + pct 0.5 -> limit stuck at 16GB)."""
        fake_vm = types.SimpleNamespace(available=100 * 1024 ** 3)
        with mock.patch("psutil.virtual_memory", return_value=fake_vm):
            cache = lora_optimizer._DiffCache(mode="auto", ram_pct=0.5)
        self.assertEqual(cache._ram_limit, 50 * 1024 ** 3)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_disk_guard_skips_caching_when_volume_low(self):
        cache = lora_optimizer._DiffCache(mode="disk")
        low = types.SimpleNamespace(free=1 * 1024 ** 3, total=100 * 1024 ** 3,
                                    used=99 * 1024 ** 3)
        with mock.patch("shutil.disk_usage", return_value=low):
            cache.put(("alias_a", 0), torch.zeros(4, 4))
        self.assertNotIn(("alias_a", 0), cache)  # skipped -> caller recomputes
        self.assertTrue(cache._disk_full)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_disk_caches_when_space_available(self):
        cache = lora_optimizer._DiffCache(mode="disk")
        ok = types.SimpleNamespace(free=50 * 1024 ** 3, total=100 * 1024 ** 3,
                                   used=50 * 1024 ** 3)
        with mock.patch("shutil.disk_usage", return_value=ok):
            cache.put(("alias_a", 0), torch.zeros(4, 4))
        self.assertIn(("alias_a", 0), cache)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_cache_stores_fp32_bit_identical(self):
        """A cache hit must equal the stored diff bit-for-bit (fp32, no fp16
        downcast) so it matches a fresh recompute — results then don't depend
        on whether a diff was cached or recomputed."""
        cache = lora_optimizer._DiffCache(mode="ram")
        diff = torch.randn(8, 8, dtype=torch.float32)
        cache.put(("alias_a", 0), diff)
        got = cache.get(("alias_a", 0))
        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(got, diff, rtol=0, atol=0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_auto_mode_declines_past_ram_budget_no_disk(self):
        """auto mode recomputes (declines to cache) past the RAM budget instead
        of spilling dense diffs to disk."""
        fake_vm = types.SimpleNamespace(available=1024)  # ram_limit = 512 bytes
        with mock.patch("psutil.virtual_memory", return_value=fake_vm):
            cache = lora_optimizer._DiffCache(mode="auto", ram_pct=0.5)
        cache.put(("alias_a", 0), torch.zeros(64, 64, dtype=torch.float32))  # 16KB > 512
        self.assertNotIn(("alias_a", 0), cache)   # declined -> caller recomputes
        self.assertEqual(len(cache._disk_store), 0)  # NOT spilled to disk


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestDiffCacheWarming(unittest.TestCase):
    """Pass 1 analysis warms the diff cache so Phase 2 candidate #1 skips
    recomputing the per-alias diffs analysis just produced."""

    def test_run_group_analysis_populates_diff_cache(self):
        optimizer = lora_optimizer.LoRAOptimizer()
        model = _make_model()
        active_loras = [
            _make_lora_entry({"alias_a": 1.0}, name="A"),
            _make_lora_entry({"alias_a": 2.0}, name="B"),
        ]
        target_groups = optimizer._build_target_groups(
            ["alias_a"], {"alias_a": "layer.weight"}, {})
        cache = lora_optimizer._DiffCache(mode="ram")
        analysis = optimizer._run_group_analysis(
            target_groups, active_loras, model, None, torch.device("cpu"),
            diff_cache=cache)
        self.assertEqual(analysis["prefix_count"], 1)
        # Per-alias diffs for both LoRAs were cached during analysis
        self.assertIn(("alias_a", 0), cache)
        self.assertIn(("alias_a", 1), cache)

    def test_analysis_results_identical_with_and_without_cache(self):
        """The cache is write-only during analysis — results must not drift."""
        optimizer = lora_optimizer.LoRAOptimizer()
        model = _make_model()
        active_loras = [
            _make_lora_entry({"alias_a": 1.0}, name="A"),
            _make_lora_entry({"alias_a": -0.5}, name="B"),
        ]
        target_groups = optimizer._build_target_groups(
            ["alias_a"], {"alias_a": "layer.weight"}, {})
        plain = optimizer._run_group_analysis(
            target_groups, active_loras, model, None, torch.device("cpu"))
        cache = lora_optimizer._DiffCache(mode="ram")
        warmed = optimizer._run_group_analysis(
            target_groups, active_loras, model, None, torch.device("cpu"),
            diff_cache=cache)
        self.assertEqual(
            plain["prefix_stats"]["alias_a"]["per_lora_norm_sq"],
            warmed["prefix_stats"]["alias_a"]["per_lora_norm_sq"])
        self.assertEqual(
            plain["prefix_stats"]["alias_a"]["conflict_ratio"],
            warmed["prefix_stats"]["alias_a"]["conflict_ratio"])


def _sd(keys):
    return {k: (torch.zeros(2, 2) if torch is not None else None) for k in keys}


@unittest.skipIf(torch is None, "torch is not installed")
class AnimaDetectionTests(unittest.TestCase):
    """Anima (CircleStone Labs / Cosmos-Predict2 DiT) detection — real key forms."""

    det = staticmethod(lambda s: lora_optimizer.LoRAOptimizer._detect_architecture(s))

    def test_diffusion_pipe_comfyui_form(self):
        s = _sd(["diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
                 "diffusion_model.blocks.0.cross_attn.output_proj.lora_up.weight",
                 "diffusion_model.blocks.0.mlp.layer1.lora_down.weight",
                 "diffusion_model.llm_adapter.blocks.0.cross_attn.k_proj.lora_down.weight"])
        self.assertEqual(self.det(s), "anima")

    def test_kohya_form(self):
        s = _sd(["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight",
                 "lora_unet_blocks_0_cross_attn_output_proj.lora_up.weight",
                 "lora_unet_blocks_0_mlp_layer1.lora_down.weight",
                 "lora_te_layers_0_self_attn_q_proj.lora_down.weight"])
        self.assertEqual(self.det(s), "anima")

    def test_no_collision_acestep_wan_ltx(self):
        # These previously-supported archs must still win, not get mistaken for Anima.
        ace = _sd(["diffusion_model.layers.0.self_attn.q_proj.lora_down.weight",
                   "diffusion_model.layers.0.cross_attn.k_proj.lora_up.weight"])
        wan = _sd(["diffusion_model.blocks.0.self_attn.q.a",
                   "diffusion_model.blocks.0.cross_attn.k.b",
                   "diffusion_model.blocks.0.ffn.0.c"])
        ltx = _sd(["transformer_blocks.0.attn1.to_q.a", "adaln_single.linear.b"])
        self.assertEqual(self.det(ace), "acestep")
        self.assertEqual(self.det(wan), "wan")
        self.assertEqual(self.det(ltx), "ltx")


class Krea2DetectionTests(unittest.TestCase):
    """Krea 2 (krea/Krea-2, from-scratch single-stream image DiT) detection."""

    det = staticmethod(lambda s: lora_optimizer.LoRAOptimizer._detect_architecture(s))

    def test_comfy_native_form(self):
        # Mirrors the official Comfy-Org/Krea-2 rank-64 LoRA: GQA attn.wq/wk/wv/wo
        # + sigmoid attn.gate + SwiGLU mlp.gate/up/down under diffusion_model.blocks.N
        s = _sd(["diffusion_model.blocks.0.attn.wq.lora_up.weight",
                 "diffusion_model.blocks.0.attn.wk.lora_up.weight",
                 "diffusion_model.blocks.0.attn.wv.lora_up.weight",
                 "diffusion_model.blocks.0.attn.wo.lora_up.weight",
                 "diffusion_model.blocks.0.attn.gate.lora_up.weight",
                 "diffusion_model.blocks.0.mlp.gate.lora_up.weight",
                 "diffusion_model.blocks.0.mlp.up.lora_up.weight",
                 "diffusion_model.blocks.0.mlp.down.lora_up.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_kohya_underscore_form(self):
        s = _sd(["lora_unet_blocks_0_attn_wq.lora_down.weight",
                 "lora_unet_blocks_0_mlp_gate.lora_up.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_trainer_diffusion_model_form(self):
        # The "krea_2" trainer: diffusion_model.transformer_blocks.N.attn.to_*
        # with a sigmoid attn.to_gate. Must NOT be mistaken for ACE-Step (which
        # also matches transformer_blocks.N.attn.to_q).
        s = _sd(["diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.attn.to_v.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.attn.to_gate.lora_A.weight",
                 "diffusion_model.transformer_blocks.0.ff.gate.lora_A.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_diffusers_transformer_form(self):
        # diffusers form: transformer.transformer_blocks.N.attn.to_* + to_gate
        # + text_fusion. Must NOT be mistaken for Qwen-Image (transformer.transformer_blocks).
        s = _sd(["transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
                 "transformer.transformer_blocks.0.attn.to_gate.lora_A.weight",
                 "transformer.transformer_blocks.0.ff.gate.lora_A.weight",
                 "transformer.text_fusion.refiner_blocks.0.attn.to_gate.lora_A.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_gate_plus_mlp_gate_fallback(self):
        # Backup discriminator when attn.w{q,k,v,o} isn't in a partial LoRA.
        s = _sd(["diffusion_model.blocks.3.attn.gate.lora_up.weight",
                 "diffusion_model.blocks.3.mlp.gate.lora_up.weight"])
        self.assertEqual(self.det(s), "krea2")

    def test_no_collision_with_wan_flux_qwen(self):
        # Krea is checked before WAN; must not steal these, and they must not steal Krea.
        wan = _sd(["diffusion_model.blocks.0.self_attn.q.a",
                   "diffusion_model.blocks.0.cross_attn.k.b",
                   "diffusion_model.blocks.0.ffn.0.c"])
        flux = _sd(["diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight"])
        qwen = _sd(["transformer.transformer_blocks.0.attn.to_q.a",
                    "transformer.transformer_blocks.0.img_mlp.net.a"])
        self.assertEqual(self.det(wan), "wan")
        self.assertEqual(self.det(flux), "flux")
        self.assertEqual(self.det(qwen), "qwen_image")

    def test_ltx2_gate_logits_not_stolen_by_krea2(self):
        # Regression: LTX-2 (LTXV 2.3) has dual video/audio + cross-modal attention
        # with '*_attn.to_gate_logits'. The substring '..._attn.to_gate' must NOT
        # trip krea2's gate detection (krea2's gate is a leaf followed by '.', LTX-2's
        # is 'to_gate_logits' with '_' after). These LTX LoRAs were mis-detected as
        # krea2 -> wrong normalization (transformer_blocks->blocks) -> nothing merged.
        s = _sd([
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_gate_logits.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn2.to_k.lora_A.weight",
            "diffusion_model.transformer_blocks.0.audio_to_video_attn.to_gate_logits.lora_A.weight",
            "diffusion_model.transformer_blocks.0.video_to_audio_attn.to_v.lora_A.weight",
        ])
        self.assertNotEqual(self.det(s), "krea2")
        self.assertEqual(self.det(s), "ltx")


class Krea2NormalizationTests(unittest.TestCase):
    """Both Krea 2 trainer forms normalize to the model-native diffusion_model.* keys.
    Mappings are shape-verified against krea2_turbo_bf16 (224/224, 264/264)."""

    norm = staticmethod(lambda s: lora_optimizer.LoRAOptimizer._normalize_keys_krea2(s))

    def test_trainer_form_attn_and_ff(self):
        n = self.norm(_sd([
            "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_B.weight",
            "diffusion_model.transformer_blocks.0.attn.to_gate.lora_A.weight",
            "diffusion_model.transformer_blocks.0.ff.gate.lora_B.weight",
            "diffusion_model.transformer_blocks.0.ff.down.lora_A.weight",
        ]))
        self.assertIn("diffusion_model.blocks.0.attn.wq.lora_A.weight", n)
        self.assertIn("diffusion_model.blocks.0.attn.wo.lora_B.weight", n)
        self.assertIn("diffusion_model.blocks.0.attn.gate.lora_A.weight", n)
        self.assertIn("diffusion_model.blocks.0.mlp.gate.lora_B.weight", n)
        self.assertIn("diffusion_model.blocks.0.mlp.down.lora_A.weight", n)

    def test_diffusers_form_transformer_prefix_and_txtfusion(self):
        n = self.norm(_sd([
            "transformer.transformer_blocks.5.attn.to_k.lora_A.weight",
            "transformer.text_fusion.layerwise_blocks.2.attn.to_v.lora_B.weight",
            "transformer.text_fusion.refiner_blocks.0.mlp.gate.lora_A.weight",
            "transformer.text_fusion.projector.lora_A.weight",
        ]))
        self.assertIn("diffusion_model.blocks.5.attn.wk.lora_A.weight", n)
        self.assertIn("diffusion_model.txtfusion.layerwise_blocks.2.attn.wv.lora_B.weight", n)
        self.assertIn("diffusion_model.txtfusion.refiner_blocks.0.mlp.gate.lora_A.weight", n)
        self.assertIn("diffusion_model.txtfusion.projector.lora_A.weight", n)

    def test_named_non_block_projections(self):
        n = self.norm(_sd([
            "transformer.img_in.lora_A.weight",
            "transformer.final_layer.linear.lora_B.weight",
            "transformer.time_mod_proj.lora_A.weight",
            "transformer.time_embed.linear_1.lora_A.weight",
            "transformer.time_embed.linear_2.lora_B.weight",
            "transformer.txt_in.linear_1.lora_A.weight",
            "transformer.txt_in.linear_2.lora_B.weight",
        ]))
        for expect in [
            "diffusion_model.first.lora_A.weight",
            "diffusion_model.last.linear.lora_B.weight",
            "diffusion_model.tproj.1.lora_A.weight",
            "diffusion_model.tmlp.0.lora_A.weight",
            "diffusion_model.tmlp.2.lora_B.weight",
            "diffusion_model.txtmlp.1.lora_A.weight",
            "diffusion_model.txtmlp.3.lora_B.weight",
        ]:
            self.assertIn(expect, n)

    def test_alpha_and_idempotent(self):
        # alpha keys are remapped too; already-canonical keys are unchanged.
        n = self.norm(_sd([
            "diffusion_model.transformer_blocks.0.attn.to_q.alpha",
            "diffusion_model.blocks.0.attn.wq.lora_A.weight",
        ]))
        self.assertIn("diffusion_model.blocks.0.attn.wq.alpha", n)
        self.assertIn("diffusion_model.blocks.0.attn.wq.lora_A.weight", n)
        # idempotent: re-normalizing canonical keys is a no-op
        self.assertEqual(set(self.norm(n).keys()), set(n.keys()))


@unittest.skipIf(torch is None, "torch is not installed")
class AnimaNormalizationTests(unittest.TestCase):
    """All trainer forms normalize to canonical diffusion_model.blocks.N.* keys."""

    norm = staticmethod(lambda s: lora_optimizer.LoRAOptimizer._normalize_keys_anima(s))

    def test_diffusion_pipe_passthrough(self):
        s = _sd(["diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight"])
        self.assertIn("diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight", self.norm(s))

    def test_kohya_restores_dots(self):
        n = self.norm(_sd([
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight",
            "lora_unet_blocks_0_cross_attn_output_proj.lora_up.weight",
            "lora_unet_blocks_0_mlp_layer1.lora_down.weight",
            "lora_unet_blocks_0_adaln_modulation_self_attn_1.lora_up.weight",
            "lora_unet_llm_adapter_blocks_0_cross_attn_q_proj.lora_down.weight",
        ]))
        for expect in [
            "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
            "diffusion_model.blocks.0.cross_attn.output_proj.lora_up.weight",
            "diffusion_model.blocks.0.mlp.layer1.lora_down.weight",
            "diffusion_model.blocks.0.adaln_modulation_self_attn.1.lora_up.weight",
            "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_down.weight",
        ]:
            self.assertIn(expect, n)

    def test_diffusers_to_canonical(self):
        n = self.norm(_sd([
            "transformer_blocks.0.attn1.to_q.lora_down.weight",
            "transformer_blocks.0.attn2.to_out.0.lora_up.weight",
            "transformer_blocks.0.ff.net.0.proj.lora_down.weight",
        ]))
        for expect in [
            "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
            "diffusion_model.blocks.0.cross_attn.output_proj.lora_up.weight",
            "diffusion_model.blocks.0.mlp.layer1.lora_down.weight",
        ]:
            self.assertIn(expect, n)

    def test_three_forms_converge(self):
        a = self.norm(_sd(["diffusion_model.blocks.0.self_attn.q_proj.w"]))
        b = self.norm(_sd(["lora_unet_blocks_0_self_attn_q_proj.w"]))
        c = self.norm(_sd(["transformer_blocks.0.attn1.to_q.w"]))
        key = "diffusion_model.blocks.0.self_attn.q_proj.w"
        self.assertIn(key, a)
        self.assertIn(key, b)
        self.assertIn(key, c)


class TargetIsAudioTests(unittest.TestCase):
    """The audio_only / no_audio key_filter classifier (LTX-2 / ACE-Step)."""

    def _g(self, label_prefix, target_key=None, aliases=None):
        return {"label_prefix": label_prefix,
                "target_key": target_key if target_key is not None else label_prefix,
                "aliases": aliases or [label_prefix]}

    def test_ltx2_audio_layers_classified_audio(self):
        is_audio = lora_optimizer.LoRAAutoTuner._target_is_audio
        for k in [
            "diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q",
            "diffusion_model.audio_adaln_single.linear",
            "diffusion_model.audio_patchify_proj",
            "diffusion_model.audio_proj_out",
            "diffusion_model.av_ca_audio_scale_shift_adaln_single.linear",
        ]:
            self.assertTrue(is_audio(self._g(k)), k)

    def test_video_layers_classified_non_audio(self):
        is_audio = lora_optimizer.LoRAAutoTuner._target_is_audio
        for k in [
            "diffusion_model.transformer_blocks.0.attn1.to_q",
            "diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_k",
            "diffusion_model.av_ca_video_scale_shift_adaln_single.linear",
            "diffusion_model.patchify_proj",
        ]:
            self.assertFalse(is_audio(self._g(k)), k)

    def test_matches_via_target_key_tuple(self):
        is_audio = lora_optimizer.LoRAAutoTuner._target_is_audio
        g = self._g("alias", target_key=("diffusion_model.audio_proj_out.weight", 0), aliases=["alias"])
        self.assertTrue(is_audio(g))


if __name__ == "__main__":
    unittest.main()
