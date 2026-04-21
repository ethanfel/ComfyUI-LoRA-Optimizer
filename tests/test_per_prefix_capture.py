"""Tests for per-prefix merge decision capture in lora_data.

Covers Task A1 (capture in lora_data) and A2 (propagate into HF upload payload).
"""
import importlib.util
import json
import os
import sys
import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

# Reuse the stub installer from the main test module so we don't duplicate it.
from tests.test_lora_optimizer import _install_stubs, lora_optimizer


@unittest.skipIf(torch is None, "torch not installed")
class TestPerPrefixCaptureInLoraData(unittest.TestCase):
    """A1: lora_data must expose per_prefix_decisions after a per-prefix merge."""

    def test_lora_data_exposes_per_prefix_decisions(self):
        """Given a populated prefix_decisions list, lora_data should carry the derived dict.

        We drive this through optimize_merge by patching ComfyUI key-lookup stubs
        to produce a trivial target group, then assert the key exists even if empty.
        """
        opt = lora_optimizer.LoRAOptimizer()

        # LoRA that contributes a simple scalar patch via weighted_sum path.
        up = torch.tensor([[1.0]], dtype=torch.float32)
        down = torch.tensor([[1.0]], dtype=torch.float32)
        stack = [
            {
                "name": "a",
                "lora": {
                    "lora_unet_layer.lora_up.weight": up,
                    "lora_unet_layer.lora_down.weight": down,
                },
                "strength": 1.0,
            },
            {
                "name": "b",
                "lora": {
                    "lora_unet_layer.lora_up.weight": up.clone(),
                    "lora_unet_layer.lora_down.weight": down.clone(),
                },
                "strength": 1.0,
            },
        ]

        # Build a fake ModelPatcher-like that satisfies clone/add_patches.
        class FakeModel:
            def __init__(self):
                layer = mock.MagicMock()
                layer.weight = torch.zeros(1, 1)
                self.model = mock.MagicMock()
                self.model.layer = layer
                self.size = 0

            def clone(self):
                return self

            def add_patches(self, patches, strength):
                return None

        fake = FakeModel()

        with mock.patch.object(lora_optimizer.comfy.lora, "model_lora_keys_unet",
                               return_value={"lora_unet_layer": "layer.weight"}):
            with mock.patch.object(opt, "_update_model_size"):
                result = opt.optimize_merge(
                    fake, stack, 1.0,
                    optimization_mode="per_prefix",
                )

        lora_data = result[4]
        self.assertIsNotNone(lora_data)
        self.assertIn("per_prefix_decisions", lora_data)
        decisions = lora_data["per_prefix_decisions"]
        self.assertIsInstance(decisions, dict)
        # Each prefix resolves to a concrete merge strategy name.
        valid_modes = {"ties", "weighted_average", "weighted_sum",
                       "consensus", "slerp", "normalize"}
        for prefix, strat in decisions.items():
            self.assertIsInstance(prefix, str)
            self.assertIn(strat, valid_modes)


if __name__ == "__main__":
    unittest.main()
