"""Tests for the inline chain-filter optimizer node."""
import os
import struct
import types
import unittest
import uuid
from unittest import mock

import torch

# Reuse the stub installer / module instance from the main test module.
from tests.test_lora_optimizer import lora_optimizer


_ALPHA_DEFAULT = object()


def _adapter(rank=4, out_dim=8, in_dim=8, up=None, down=None,
             alpha=_ALPHA_DEFAULT, dora_scale=None, reshape=None):
    """Minimal LoRAAdapter-like payload the engine can expand. Explicit
    up/down matrices make the expanded diff deterministic (alpha == rank, so
    the diff is exactly up @ down). Pass alpha=None to mimic PEFT/diffusers/
    AI-Toolkit LoRAs whose files carry no .alpha key — comfy stores such
    adapters with alpha=None (scale 1.0)."""
    if up is None:
        up = torch.randn(out_dim, rank)
    if down is None:
        down = torch.randn(rank, in_dim)
    if alpha is _ALPHA_DEFAULT:
        alpha = float(rank)  # alpha == rank -> scale 1.0, diff == up @ down
    return lora_optimizer.LoRAAdapter(
        loaded_keys=set(),
        weights=(up, down, alpha, None, dora_scale, reshape),
    )


def _entry(strength, payload, strength_model=1.0, offset=None, function=None):
    """A ModelPatcher patch-list entry, shaped like model_patcher.py:807."""
    return (strength, payload, strength_model, offset, function)


def _lora_adapter(up, down, alpha):
    """A comfy LoRAAdapter payload with explicit up/down/alpha (mid=None)."""
    return lora_optimizer.LoRAAdapter(
        loaded_keys=set(), weights=(up, down, alpha, None, None, None))


def _virtual_item(payloads, strength=1.0, clip_strength=None, name="v",
                  conflict_mode="all", key_filter="all", preserve=False):
    """A captured/virtual active_loras item: its lora dict maps MODEL TARGET
    KEYS -> adapter payloads (not trainer-format lora_up/down keys)."""
    return {
        "name": name, "lora": dict(payloads), "_precomputed_diffs": True,
        "strength": strength, "clip_strength": clip_strength,
        "conflict_mode": conflict_mode, "key_filter": key_filter,
        "preserve": preserve, "metadata": {},
    }


def _layer_model(out=8, in_=8, key_attr="layer"):
    """Model whose model.<key_attr>.weight has the given shape, so
    _resolve_target_shape('<key_attr>.weight') resolves."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(**{
            key_attr: types.SimpleNamespace(weight=torch.zeros(out, in_))}))


class TestPatchClassification(unittest.TestCase):
    def test_adapter_is_capturable(self):
        e = _entry(0.8, _adapter())
        self.assertTrue(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_diff_tuple_is_capturable(self):
        e = _entry(1.0, ("diff", (torch.randn(4, 4),)))
        self.assertTrue(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_set_tuple_passes_through(self):
        e = _entry(1.0, ("set", (torch.randn(4, 4),)))
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_nonunit_strength_model_passes_through(self):
        e = _entry(1.0, _adapter(), strength_model=0.5)
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_function_entry_passes_through(self):
        e = _entry(1.0, _adapter(), function=lambda w: w)
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_unknown_object_passes_through(self):
        e = _entry(1.0, object())
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_non_5_tuple_entry_passes_through(self):
        # older/nonstandard third-party nodes may store short entries — the
        # classifier must not crash on unpack, just pass them through
        e = (1.0, _adapter(), 1.0)
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_padded_diff_passes_through(self):
        # comfy pads the BASE weight at apply time for this shape — we cannot
        # faithfully expand it to a bare diff tensor
        e = _entry(1.0, ("diff", (torch.randn(4, 4), {"pad_weight": True})))
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_malformed_diff_passes_through(self):
        e = _entry(1.0, ("diff", None))
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))

    def test_dora_adapter_passes_through(self):
        # DoRA depends on the current base weight, so expanding only up @ down
        # and re-appending it later is not an additive-equivalent operation.
        payloads = [
            _adapter(dora_scale=torch.ones(8, 1)),
            lora_optimizer.LoKrAdapter(
                set(), (torch.randn(4, 4), torch.randn(2, 2), None,
                        None, None, None, None, None, torch.ones(8, 1))),
            lora_optimizer.LoHaAdapter(
                set(), (torch.randn(8, 2), torch.randn(2, 8), 2.0,
                        torch.randn(8, 2), torch.randn(2, 8), None, None,
                        torch.ones(8, 1))),
        ]
        for payload in payloads:
            with self.subTest(payload=type(payload).__name__):
                self.assertFalse(
                    lora_optimizer._LoRAMergeBase._is_capturable_entry(
                        _entry(1.0, payload)))

    def test_reshape_adapter_passes_through(self):
        # Comfy pads/replaces the working weight shape before applying a
        # reshape-bearing LoRA.  A bare merged delta loses that operation.
        up = torch.randn(9, 4)
        down = torch.randn(4, 8)
        e = _entry(1.0, _adapter(up=up, down=down, reshape=[9, 8]))
        self.assertFalse(lora_optimizer._LoRAMergeBase._is_capturable_entry(e))


class TestNoneAlphaExpansion(unittest.TestCase):
    """PEFT/diffusers/AI-Toolkit LoRAs have no .alpha key, so a captured comfy
    LoRAAdapter carries alpha=None. _expand_patch_to_diff must treat that as
    scale 1.0 (comfy semantics: scale = alpha/rank if alpha is not None else
    1.0) instead of crashing on None / rank."""

    def test_none_alpha_adapter_expands_at_scale_one(self):
        up = torch.randn(8, 4)
        down = torch.randn(4, 8)
        adapter = _adapter(up=up, down=down, alpha=None)
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(adapter)
        self.assertTrue(torch.allclose(got, up @ down, atol=1e-6))

    def test_numeric_alpha_still_rescales(self):
        # regression net: a real alpha still divides by rank
        up = torch.randn(8, 4)
        down = torch.randn(4, 8)
        adapter = _adapter(up=up, down=down, alpha=2.0)  # rank 4 -> scale 0.5
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(adapter)
        self.assertTrue(torch.allclose(got, (up @ down) * 0.5, atol=1e-6))


def _chain_patches(*loras):
    """Simulate a loader chain: each lora is {key: payload} applied at a
    strength. Distinct float objects per call, entries appended in order —
    exactly what ModelPatcher.add_patches does."""
    patches = {}
    seen_ids = set()
    for strength_value, lora in loras:
        # struct round-trip mints a FRESH float object even for values whose
        # literals CPython folds/interns (e.g. 1.0 used twice in one test).
        # float(x) would return the SAME object for an exact float input.
        s = struct.unpack("d", struct.pack("d", strength_value))[0]
        assert id(s) not in seen_ids, "float object reused across simulated calls"
        seen_ids.add(id(s))
        for key, payload in lora.items():
            patches.setdefault(key, []).append(_entry(s, payload))
    return patches


class TestChainGroupReconstruction(unittest.TestCase):
    def _reconstruct(self, patches):
        return lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(patches)

    def test_single_lora(self):
        patches = _chain_patches((0.8, {"a": _adapter(), "b": _adapter()}))
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(groups[0]["strength"], 0.8)
        self.assertEqual(set(groups[0]["entries"]), {"a", "b"})

    def test_two_loras_chain_order(self):
        patches = _chain_patches(
            (0.8, {"a": _adapter(), "b": _adapter()}),
            (0.5, {"a": _adapter(), "c": _adapter()}),
        )
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(groups[0]["strength"], 0.8)
        self.assertAlmostEqual(groups[1]["strength"], 0.5)
        self.assertEqual(set(groups[1]["entries"]), {"a", "c"})

    def test_subset_alignment(self):
        # A patches attn only; B patches attn+mlp. Naive index-as-identity
        # would misattribute B's mlp entry (position 0 there) to A.
        patches = _chain_patches(
            (0.8, {"attn": _adapter()}),
            (0.5, {"attn": _adapter(), "mlp": _adapter()}),
        )
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 2)
        self.assertEqual(set(groups[0]["entries"]), {"attn"})
        self.assertEqual(set(groups[1]["entries"]), {"attn", "mlp"})

    def test_same_strength_value_distinct_objects(self):
        patches = _chain_patches(
            (1.0, {"a": _adapter()}),
            (1.0, {"a": _adapter(), "b": _adapter()}),
        )
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 2)

    def test_shared_float_object_fallback(self):
        # Pathological: two calls share ONE float object (interning). The
        # id-group then holds two entries on key "a" — must split by per-key
        # order instead of returning a corrupt group.
        s = 1.0
        patches = {}
        for lora in ({"a": _adapter()}, {"a": _adapter(), "b": _adapter()}):
            for key, payload in lora.items():
                patches.setdefault(key, []).append(_entry(s, payload))
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 2)
        for g in groups:
            keys = [k for k in g["entries"]]
            self.assertEqual(len(keys), len(set(keys)))

    def test_offset_entries_get_tuple_keys(self):
        off = (0, 0, 4)
        patches = {"qkv": [_entry(0.7, _adapter(), offset=off)]}
        groups = self._reconstruct(patches)
        self.assertEqual(list(groups[0]["entries"]), [("qkv", off)])

    def test_noncapturable_entry_on_other_target_ignored(self):
        patches = _chain_patches((0.8, {"a": _adapter()}))
        patches["z"] = [_entry(1.0, ("set", (torch.zeros(2, 2),)))]
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]["entries"]), {"a"})

    def test_empty_patches(self):
        self.assertEqual(self._reconstruct({}), [])

    def test_non_5_tuple_entry_ignored_without_raising(self):
        patches = _chain_patches((0.8, {"a": _adapter()}))
        patches["z"] = [(1.0, _adapter(), 1.0)]  # nonstandard 3-tuple
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]["entries"]), {"a"})

    def test_unsafe_entry_blocks_every_target_from_the_same_loader_call(self):
        # One add_patches call shares the exact strength object across all of
        # its target entries.  If one entry is DoRA/reshape/order-dependent,
        # capturing only the other entry would partially rewrite one LoRA.
        patches = _chain_patches(
            (0.8, {"unsafe": _adapter(dora_scale=torch.ones(8, 1)),
                   "otherwise_safe": _adapter()}),
            (0.5, {"independent": _adapter()}),
        )
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(groups[0]["strength"], 0.5)
        self.assertEqual(set(groups[0]["entries"]), {"independent"})

    def test_unsafe_barrier_blocks_prefix_but_allows_additive_suffix(self):
        # Call A is unsafe on x and otherwise ordinary on y.  Call B also
        # touches y.  Moving B is safe only when B is AFTER A already: Inline
        # removes it and appends the optimized diff back at the same tail
        # position.  When B precedes A, moving it across A would change order.
        unsafe_call = (
            0.8,
            {"x": _adapter(dora_scale=torch.ones(8, 1)),
             "y": _adapter()},
        )
        shared_call = (0.5, {"y": _adapter()})

        groups = self._reconstruct(_chain_patches(unsafe_call, shared_call))
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(groups[0]["strength"], 0.5)
        self.assertEqual(set(groups[0]["entries"]), {"y"})

        groups = self._reconstruct(_chain_patches(shared_call, unsafe_call))
        self.assertEqual(groups, [])

    def test_include_blocked_retains_loader_slot_placeholder(self):
        patches = _chain_patches(
            (0.8, {"unsafe": _adapter(dora_scale=torch.ones(8, 1))}),
            (0.5, {"safe": _adapter()}),
        )
        groups = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            patches, include_blocked=True)
        self.assertEqual(len(groups), 2)
        self.assertFalse(groups[0]["mergeable"])
        self.assertEqual(groups[0]["captured"], [])
        self.assertEqual(set(groups[0]["entries"]), {"unsafe"})
        self.assertTrue(groups[1]["mergeable"])
        self.assertEqual(set(groups[1]["entries"]), {"safe"})

    def test_shape_changing_barrier_blocks_additive_suffix(self):
        # The base target may be 8x8, while reshape pads it to 9x8 before the
        # later 9x8 LoRA. Inline resolves expansion shapes from the unpatched
        # base, so capturing that suffix would drop it as a mismatch.
        reshape = _adapter(
            up=torch.randn(9, 4), down=torch.randn(4, 8), reshape=[9, 8])
        post_reshape = _adapter(up=torch.randn(9, 4), down=torch.randn(4, 8))
        patches = _chain_patches(
            (0.8, {"layer.weight": reshape}),
            (0.5, {"layer.weight": post_reshape}),
        )
        self.assertEqual(self._reconstruct(patches), [])
        groups = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            patches, include_blocked=True)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(not g["mergeable"] for g in groups))
        self.assertTrue(all(g["captured"] == [] for g in groups))

    def test_interleaved_collision_no_fragmentation(self):
        # Distinct-strength loader X on {a} first, then two interned-strength
        # loaders B and C both patching {a, b}. The collision sub-gid must be
        # the per-target-key collision ORDINAL — splitting on the absolute
        # per-key position fragments this into 4 groups [a],[a,b],[a],[b]
        # because X shifts the positions on "a" but not on "b".
        d = struct.unpack("d", struct.pack("d", 0.6))[0]
        s = 1.0
        pX = _adapter()
        pB_a, pB_b = _adapter(), _adapter()
        pC_a, pC_b = _adapter(), _adapter()
        patches = {
            "a": [_entry(d, pX), _entry(s, pB_a), _entry(s, pC_a)],
            "b": [_entry(s, pB_b), _entry(s, pC_b)],
        }
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 3)
        self.assertEqual(set(groups[0]["entries"]), {"a"})
        self.assertEqual(set(groups[1]["entries"]), {"a", "b"})
        self.assertEqual(set(groups[2]["entries"]), {"a", "b"})
        # ordinal-k entries belong to the k-th colliding call on EVERY key
        self.assertIs(groups[1]["entries"]["a"], pB_a)
        self.assertIs(groups[1]["entries"]["b"], pB_b)
        self.assertIs(groups[2]["entries"]["a"], pC_a)
        self.assertIs(groups[2]["entries"]["b"], pC_b)

    def test_collision_subgroup_reordered_by_precedence(self):
        # Chain: two interned-strength calls A, B on "a", then distinct call D
        # on {z, a}. Dict iteration starts at "z", so D's group is CREATED
        # first; both collision groups (base + ordinal sub-group) must be
        # sorted forward past D via the shared-key "a" positions — this
        # exercises the insertion sort on a collision-created sub-group.
        s = 1.0
        d = struct.unpack("d", struct.pack("d", 0.5))[0]
        pA, pB, pD_a, pD_z = _adapter(), _adapter(), _adapter(), _adapter()
        patches = {
            "z": [_entry(d, pD_z)],
            "a": [_entry(s, pA), _entry(s, pB), _entry(d, pD_a)],
        }
        groups = self._reconstruct(patches)
        self.assertEqual(len(groups), 3)
        self.assertIs(groups[0]["entries"]["a"], pA)
        self.assertIs(groups[1]["entries"]["a"], pB)
        self.assertEqual(set(groups[2]["entries"]), {"z", "a"})


class _FakePatcher:
    """Minimal ModelPatcher stand-in: ordered patch lists + clone()."""
    def __init__(self, patches=None):
        self.patches = patches if patches is not None else {}
        self.patches_uuid = object()

    def clone(self):
        return _FakePatcher({k: v[:] for k, v in self.patches.items()})

    def add_patches(self, patches, strength=1.0):
        for key, patch in patches.items():
            self.patches.setdefault(key, []).append((strength, patch, 1.0, None, None))
        return list(patches)


class TestStripCaptured(unittest.TestCase):
    def test_interleaved_noncapturable_entry_blocks_target_and_preserves_order(self):
        keep = _entry(1.0, ("set", (torch.zeros(2, 2),)))
        patches = _chain_patches((0.8, {"a": _adapter(), "b": _adapter()}))
        patches["a"].append(keep)
        tail_strength = struct.unpack("d", struct.pack("d", 0.6))[0]
        tail = _entry(tail_strength, _adapter())
        patches["a"].append(tail)
        original_a = list(patches["a"])
        original_b = list(patches["b"])
        patcher = _FakePatcher(patches)
        groups = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(patcher.patches)
        clone = patcher.clone()
        lora_optimizer._LoRAMergeBase._strip_captured_entries(clone, groups)
        # A generic weight-set barrier may replace the working tensor shape.
        # Inline resolves against the unpatched base, so both the shared prefix
        # and the additive suffix remain untouched conservatively.
        self.assertEqual(groups, [])
        self.assertEqual(clone.patches,
                         {"a": original_a, "b": original_b})
        self.assertIs(clone.patches["a"][0], original_a[0])
        self.assertIs(clone.patches["a"][1], keep)
        self.assertIs(clone.patches["a"][2], tail)
        self.assertEqual(len(patcher.patches["a"]), 3)  # original untouched

    def test_uuid_regenerated(self):
        patcher = _FakePatcher(_chain_patches((0.8, {"a": _adapter()})))
        groups = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(patcher.patches)
        clone = patcher.clone()
        before = clone.patches_uuid
        lora_optimizer._LoRAMergeBase._strip_captured_entries(clone, groups)
        self.assertNotEqual(clone.patches_uuid, before)


def _slot(enabled=True, strength=1.0, model_strength=1.0, clip_strength=1.0,
          conflict_mode="all", key_filter="all", preserve=False):
    return dict(enabled=enabled, strength=strength, model_strength=model_strength,
                clip_strength=clip_strength, conflict_mode=conflict_mode,
                key_filter=key_filter, preserve=preserve)


class TestChainStackBuild(unittest.TestCase):
    def _build(self, model_groups, clip_groups, slots, visibility="simple"):
        return lora_optimizer.LoRAOptimizerInline._chain_groups_to_stack(
            model_groups, clip_groups, slots, visibility)

    def test_basic_item_schema(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        stack = self._build(mg, [], [_slot()])
        item = stack[0]
        self.assertTrue(item["_precomputed_diffs"])
        self.assertAlmostEqual(item["strength"], 0.8)     # loader strength kept
        self.assertEqual(item["clip_strength"], 0.0)
        self.assertEqual(item["conflict_mode"], "all")
        self.assertIn("a", item["lora"])

    def test_simple_mode_multiplier(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        stack = self._build(mg, [], [_slot(strength=0.5)])
        self.assertAlmostEqual(stack[0]["strength"], 0.4)  # 0.8 loader × 0.5 slot

    def test_advanced_mode_split_multipliers(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot(model_strength=0.5, clip_strength=2.0)],
                            visibility="advanced")
        self.assertAlmostEqual(stack[0]["strength"], 0.4)        # 0.8 × 0.5
        self.assertAlmostEqual(stack[0]["clip_strength"], 1.2)   # 0.6 × 2.0
        self.assertIn("te.a", stack[0]["lora"])                  # clip keys merged in

    def test_disabled_slot_excluded(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        stack = self._build(mg, [], [_slot(enabled=False), _slot()])
        self.assertEqual(len(stack), 1)
        self.assertAlmostEqual(stack[0]["strength"], 0.5)

    def test_missing_slots_get_defaults(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        stack = self._build(mg, [], [_slot(preserve=True)])   # only 1 slot for 2 loras
        self.assertEqual(len(stack), 2)
        self.assertTrue(stack[0]["preserve"])
        self.assertFalse(stack[1]["preserve"])

    def test_leftover_clip_groups_become_clip_only_items(self):
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build([], cg, [])
        self.assertEqual(len(stack), 1)
        self.assertIn("te.a", stack[0]["lora"])

    def test_leftover_strength_is_clip_product_nonzero(self):
        # the engine drops strength == 0 items (active_loras filter in
        # optimize_merge) — clip-only items must ride on the clip product
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build([], cg, [])
        self.assertNotEqual(stack[0]["strength"], 0.0)
        self.assertAlmostEqual(stack[0]["strength"], 0.6)
        self.assertAlmostEqual(stack[0]["clip_strength"], 0.6)

    def test_zeroed_model_branch_keeps_live_clip_branch(self):
        # stock-UI case: LoraLoader with strength_model=0, strength_clip=1.0
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot(model_strength=0.0, clip_strength=1.0)],
                            visibility="advanced")
        self.assertEqual(len(stack), 1)
        self.assertNotIn("a", stack[0]["lora"])       # unet keys dropped
        self.assertIn("te.a", stack[0]["lora"])
        self.assertAlmostEqual(stack[0]["strength"], 0.6)      # clip product
        self.assertAlmostEqual(stack[0]["clip_strength"], 0.6)

    def test_both_branches_zeroed_item_excluded(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot(model_strength=0.0, clip_strength=0.0)],
                            visibility="advanced")
        self.assertEqual(stack, [])

    def test_leftover_after_model_groups_uses_own_slot_index(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()}), (0.4, {"te.b": _adapter()})))
        stack = self._build(mg, cg, [_slot(), _slot()])
        self.assertEqual(len(stack), 2)
        self.assertEqual(stack[1]["name"], "chain lora #2 (clip-only)")
        self.assertIn("te.b", stack[1]["lora"])
        self.assertAlmostEqual(stack[1]["strength"], 0.4)
        # slot index j (not 0) gates the leftover — disabling slot 2 drops it
        stack = self._build(mg, cg, [_slot(), _slot(enabled=False)])
        self.assertEqual(len(stack), 1)
        self.assertNotIn("(clip-only)", stack[0]["name"])

    def test_simple_visibility_scales_paired_clip(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot(strength=0.5)])   # simple visibility
        self.assertAlmostEqual(stack[0]["strength"], 0.4)          # 0.8 × 0.5
        self.assertAlmostEqual(stack[0]["clip_strength"], 0.3)     # 0.6 × 0.5

    def test_global_clip_multiplier_stays_out_of_relative_item_weights(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        cg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.6, {"te.a": _adapter()})))
        stack = self._build(mg, cg, [_slot()])
        self.assertAlmostEqual(stack[0]["strength"], 0.8)
        self.assertAlmostEqual(stack[0]["clip_strength"], 0.6)
        self.assertTrue(
            lora_optimizer.LoRAOptimizerInline._force_explicit_clip_multiplier)

    def test_extra_slots_ignored(self):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"a": _adapter()})))
        stack = self._build(mg, [], [_slot(), _slot(strength=9.0), _slot(enabled=False)])
        self.assertEqual(len(stack), 1)
        self.assertAlmostEqual(stack[0]["strength"], 0.8)


class _FakeCLIP:
    def __init__(self, patcher):
        self.patcher = patcher
        self.cond_stage_model = types.SimpleNamespace()

    def clone(self):
        return _FakeCLIP(self.patcher.clone())


class TestInlineExecute(unittest.TestCase):
    def _node(self):
        return lora_optimizer.LoRAOptimizerInline()

    def _model(self, patches):
        m = _FakePatcher(patches)
        m.model = types.SimpleNamespace()
        return m

    @staticmethod
    def _out(result):
        return result["result"] if isinstance(result, dict) else result

    def _capture_merge(self, node, seen):
        """Mock optimize_merge, recording every kwarg + the stack/model."""
        def fake_merge(m, stack, output_strength, **kw):
            seen["model"] = m
            seen["stack"] = stack
            seen["output_strength"] = output_strength
            seen.update(kw)
            return (m, kw.get("clip"), "engine report", None, None)
        node.optimize_merge = fake_merge

    def test_no_lora_patches_passthrough(self):
        model = self._model({})
        node = self._node()
        result = node.execute_inline(model, output_strength=1.0)
        out = self._out(result)
        self.assertIs(out[0], model)          # unchanged object back
        self.assertIn("No LoRA patches", out[2])
        self.assertIn("AFTER your Load LoRA nodes", out[2])
        self.assertIsNone(out[3])
        self.assertIsNone(out[4])

    def test_non_mergeable_only_patches_reported_honestly(self):
        # Patches exist but none are capturable (OFT/"set"/function entries):
        # "place this node AFTER your loaders" would be WRONG advice.
        patches = {"a": [_entry(1.0, ("set", (torch.zeros(2, 2),))),
                         _entry(1.0, _adapter(), function=lambda w: w)]}
        model = self._model(patches)
        node = self._node()
        result = node.execute_inline(model, output_strength=1.0)
        out = self._out(result)
        self.assertIs(out[0], model)
        self.assertIn("2 non-mergeable patch entries", out[2])
        self.assertIn("passed through", out[2])
        self.assertNotIn("AFTER your Load LoRA nodes", out[2])

    def test_merge_called_with_virtual_stack_and_stripped_model(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        node.execute_inline(model, output_strength=1.0)
        self.assertEqual(len(seen["stack"]), 2)
        self.assertTrue(all(i["_precomputed_diffs"] for i in seen["stack"]))
        self.assertEqual(seen["model"].patches, {})   # stripped clone
        self.assertIsNot(seen["model"], model)
        self.assertEqual(seen["normalize_keys"], "disabled")

    def test_no_settings_inherits_simple_defaults_with_inline_pins(self):
        # Standalone Inline should make the same quality decisions as the
        # simplified dedicated optimizer.  Only the two inline-inapplicable
        # cache/key-normalization settings are deliberately pinned off.
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        node.execute_inline(model, output_strength=1.0)

        for key, value in lora_optimizer.LoRAOptimizerSimple._SIMPLE_DEFAULTS.items():
            expected = "disabled" if key in ("normalize_keys", "cache_patches") else value
            self.assertIn(key, seen)
            self.assertEqual(seen[key], expected, key)

    def test_report_prepends_fingerprints(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter(), "b": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "engine report", None, None)
        result = node.execute_inline(model, output_strength=1.0)
        report = self._out(result)[2]
        self.assertIn("#1", report)
        self.assertIn("2 keys", report)
        self.assertIn("0.80", report)
        self.assertIn("rank 4", report)
        self.assertIn("engine report", report)

    def test_disabled_slot_stripped_but_not_merged(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        node.execute_inline(model, output_strength=1.0,
                            chain_options={"visibility": "simple",
                                           "slots": [_slot(enabled=False), _slot()]})
        self.assertEqual(len(seen["stack"]), 1)
        self.assertAlmostEqual(seen["stack"][0]["strength"], 0.5)
        self.assertEqual(seen["model"].patches, {})   # disabled LoRA stripped too

    def test_blocked_loader_keeps_its_slot_and_does_not_shift_safe_suffix(self):
        # Loader #1 is DoRA and must pass through. Loader #2 is ordinary and
        # remains attributable to slot #2; compacting blocked groups would
        # incorrectly apply the disabled slot #1 to it.
        unsafe = _adapter(dora_scale=torch.ones(8, 1))
        safe = _adapter()
        model = self._model(_chain_patches(
            (0.8, {"unsafe": unsafe}),
            (0.5, {"safe": safe}),
        ))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        result = node.execute_inline(
            model, output_strength=1.0,
            chain_options={"visibility": "simple",
                           "slots": [_slot(enabled=False), _slot()]})

        self.assertEqual(len(seen["stack"]), 1)
        self.assertTrue(seen["stack"][0]["name"].startswith("chain lora #2"))
        self.assertEqual(set(seen["stack"][0]["lora"]), {"safe"})
        self.assertAlmostEqual(seen["stack"][0]["strength"], 0.5)
        self.assertEqual(set(seen["model"].patches), {"unsafe"})
        self.assertIs(seen["model"].patches["unsafe"][0][1], unsafe)
        report = self._out(result)[2]
        self.assertIn("#1: model passthrough/order-blocked", report)
        self.assertIn("#2: 1 keys", report)

    def test_asymmetric_model_clip_blocking_never_cross_pairs_files(self):
        # Slot #1: MODEL is DoRA-blocked, CLIP is additive.
        # Slot #2: MODEL is additive, CLIP is DoRA-blocked.
        # Both branches need placeholders or independent compaction would
        # create one hybrid virtual LoRA containing slot #2 MODEL + slot #1 CLIP.
        model_unsafe = _adapter(dora_scale=torch.ones(8, 1))
        model_safe = _adapter()
        clip_safe = _adapter()
        clip_unsafe = _adapter(dora_scale=torch.ones(8, 1))
        model = self._model(_chain_patches(
            (0.8, {"model.unsafe": model_unsafe}),
            (0.5, {"model.safe": model_safe}),
        ))
        clip = _FakeCLIP(_FakePatcher(_chain_patches(
            (0.8, {"te.safe": clip_safe}),
            (0.5, {"te.unsafe": clip_unsafe}),
        )))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        result = node.execute_inline(model, output_strength=1.0, clip=clip)

        self.assertEqual(len(seen["stack"]), 2)
        first, second = seen["stack"]
        self.assertTrue(first["name"].startswith("chain lora #1"))
        self.assertEqual(set(first["lora"]), {"te.safe"})
        self.assertAlmostEqual(first["strength"], 0.8)
        self.assertAlmostEqual(first["clip_strength"], 0.8)
        self.assertTrue(second["name"].startswith("chain lora #2"))
        self.assertEqual(set(second["lora"]), {"model.safe"})
        self.assertAlmostEqual(second["strength"], 0.5)
        self.assertEqual(second["clip_strength"], 0.0)

        # Only the two blocked branches remain on the clones passed onward.
        self.assertEqual(set(seen["model"].patches), {"model.unsafe"})
        self.assertEqual(set(seen["clip"].patcher.patches), {"te.unsafe"})
        report = self._out(result)[2]
        self.assertIn("#1: model passthrough/order-blocked", report)
        self.assertIn("#2: 1 keys, rank 4", report)
        self.assertIn("clip passthrough/order-blocked", report)

    def test_clip_multiplier_is_stable_with_or_without_model_only_item(self):
        # Explicit captured CLIP strengths used to bypass the node-wide
        # multiplier. Worse, adding any MODEL-only item changed the engine's
        # all-explicit decision and suddenly scaled every CLIP patch.
        for add_model_only in (False, True):
            with self.subTest(add_model_only=add_model_only):
                calls = [(0.8, {"model.a": _adapter()})]
                if add_model_only:
                    calls.append((0.4, {"model.only": _adapter()}))
                model = self._model(_chain_patches(*calls))
                clip = _FakeCLIP(_FakePatcher(_chain_patches(
                    (0.6, {"te.a": _adapter()}))))
                node = self._node()
                seen = {}
                self._capture_merge(node, seen)
                node.execute_inline(
                    model, output_strength=1.0, clip=clip,
                    clip_strength_multiplier=0.5)

                clip_items = [item for item in seen["stack"]
                              if "te.a" in item["lora"]]
                self.assertEqual(len(clip_items), 1)
                # Per-item weights keep loader-relative ratios; the node-wide
                # 0.5 is forced once at the engine's CLIP add_patches stage.
                self.assertAlmostEqual(clip_items[0]["clip_strength"], 0.6)
                self.assertTrue(node._force_explicit_clip_multiplier)
                self.assertTrue(all(item["clip_strength"] is not None
                                    for item in seen["stack"]))
                if add_model_only:
                    model_only = [item for item in seen["stack"]
                                  if "model.only" in item["lora"]][0]
                    self.assertEqual(model_only["clip_strength"], 0.0)

    def test_unconnected_chain_options_merges_all_with_defaults(self):
        # chain_options unconnected (None): every captured LoRA merges with
        # default options, and the report shows NO slots-vs-LoRAs note
        # (merge-all-defaults is not a mismatch).
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertEqual(len(seen["stack"]), 2)               # both merged
        self.assertAlmostEqual(seen["stack"][0]["strength"], 0.8)  # default mult
        self.assertAlmostEqual(seen["stack"][1]["strength"], 0.5)
        self.assertFalse(any(i.get("preserve") for i in seen["stack"]))
        self.assertNotIn("option slot", report)               # no slots-vs-LoRAs
        self.assertEqual(seen["normalize_keys"], "disabled")

    def test_fingerprint_warns_on_group_count_mismatch(self):
        # 2 model groups vs 1 clip group (both non-empty) -> order-attribution
        # warning naming the usual suspects
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        clip = _FakeCLIP(_FakePatcher(_chain_patches((0.6, {"te.a": _adapter()}))))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        result = node.execute_inline(model, output_strength=1.0, clip=clip)
        report = self._out(result)[2]
        self.assertIn("2 model-side vs 1 clip-side", report)
        self.assertIn("LoraLoaderModelOnly", report)
        # the clip clone passed to the engine is stripped, original untouched
        self.assertIsNot(seen["clip"], clip)
        self.assertEqual(seen["clip"].patcher.patches, {})
        self.assertEqual(len(clip.patcher.patches["te.a"]), 1)

    def test_fingerprint_counts_passthrough_entries(self):
        patches = _chain_patches((0.8, {"a": _adapter()}))
        # Keep the noncapturable entry on a distinct target so key a remains
        # mergeable; a barrier on key a intentionally leaves both entries and
        # is covered by TestStripCaptured.
        patches["z"] = [_entry(1.0, ("set", (torch.zeros(2, 2),)))]
        model = self._model(patches)
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        result = node.execute_inline(model, output_strength=1.0)
        report = self._out(result)[2]
        self.assertIn("1 non-mergeable", report)
        self.assertIn("passed through untouched", report)

    def test_no_passthrough_note_when_all_captured(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertNotIn("non-mergeable", report)

    def test_fingerprint_notes_fewer_slots_than_loras(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(
            model, output_strength=1.0,
            chain_options={"visibility": "simple", "slots": [_slot()]}))[2]
        self.assertIn("2 LoRAs detected but only 1 option slot", report)
        self.assertIn("default options", report)

    def test_fingerprint_notes_more_slots_than_loras(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(
            model, output_strength=1.0,
            chain_options={"visibility": "simple",
                           "slots": [_slot(), _slot(), _slot()]}))[2]
        self.assertIn("3 option slots but only 1 LoRA", report)
        self.assertIn("extra slots ignored", report)

    def test_advanced_settings_delegated_normalize_keys_pinned(self):
        settings = {
            "mode": "advanced",
            "auto_strength": "disabled", "auto_strength_floor": 0.5,
            "optimization_mode": "global",
            "sparsification": "dare", "sparsification_density": 0.5,
            "dare_dampening": 0.1,
            "merge_refinement": "refine", "strategy_set": "basic",
            "normalize_keys": "enabled",   # must NOT reach the engine
            "architecture_preset": "dit", "decision_smoothing": 0.1,
            "smooth_slerp_gate": True,
            "star_eta": 0.9, "tame_layers": 0.1, "tame_threshold": 0.2,
            "cache_patches": "disabled", "patch_compression": "disabled",
            "svd_device": "cpu", "free_vram_between_passes": "enabled",
            "vram_budget": 0.25,
        }
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        result = node.execute_inline(model, output_strength=1.0, settings=settings)
        report = self._out(result)[2]
        self.assertNotIn("not supported inline", report)
        self.assertEqual(seen["normalize_keys"], "disabled")  # pinned
        self.assertEqual(seen["optimization_mode"], "global")
        self.assertEqual(seen["sparsification"], "dare")
        self.assertAlmostEqual(seen["sparsification_density"], 0.5)
        self.assertEqual(seen["cache_patches"], "disabled")
        self.assertEqual(seen["svd_device"], "cpu")
        self.assertAlmostEqual(seen["star_eta"], 0.9)
        # explicit preset from the settings node -> no arch-unknown warning
        self.assertNotIn("architecture: unknown", report)

    def test_fingerprint_warns_arch_unknown_without_preset(self):
        # Virtual items skip _normalize_stack's arch detection, so auto
        # detection can never resolve — the report must say so.
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertIn("architecture: unknown (inline capture)", report)
        self.assertIn("architecture_preset", report)

    def test_passthrough_count_includes_clip_side(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        clip_patches = _chain_patches((0.6, {"te.a": _adapter()}))
        clip_patches["te.z"] = [
            _entry(1.0, ("set", (torch.zeros(2, 2),)))
        ]
        clip = _FakeCLIP(_FakePatcher(clip_patches))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, kw.get("clip"), "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0,
                                               clip=clip))[2]
        self.assertIn("1 non-mergeable", report)     # the clip-side "set" entry
        self.assertIn("passed through untouched", report)

    def test_lokr_payload_reports_dense_rank(self):
        # LoKr weights[1] is the w2 Kronecker factor, NOT a rank — the
        # fingerprint must not misreport its shape as "rank N".
        lokr = lora_optimizer.LoKrAdapter(
            loaded_keys=set(),
            weights=(torch.randn(4, 4), torch.randn(2, 2), None, None,
                     None, None, None, None))
        model = self._model(_chain_patches((0.8, {"a": lokr})))
        node = self._node()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertIn("dense", report)
        self.assertNotIn("rank 2", report)

    def test_advanced_visibility_widgets_reach_stack(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
        clip = _FakeCLIP(_FakePatcher(_chain_patches(
            (0.6, {"te.a": _adapter()}), (0.4, {"te.b": _adapter()}))))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        node.execute_inline(model, output_strength=1.0, clip=clip,
                            chain_options={"visibility": "advanced",
                                           "slots": [_slot(),
                                                     _slot(model_strength=0.5,
                                                           clip_strength=2.0)]})
        self.assertEqual(len(seen["stack"]), 2)
        self.assertAlmostEqual(seen["stack"][0]["strength"], 0.8)       # defaults
        self.assertAlmostEqual(seen["stack"][0]["clip_strength"], 0.6)
        self.assertAlmostEqual(seen["stack"][1]["strength"], 0.25)      # 0.5 x 0.5
        self.assertAlmostEqual(seen["stack"][1]["clip_strength"], 0.8)  # 0.4 x 2.0

    def test_salt_names_differ_across_source_patchers(self):
        # Two DIFFERENT loader chains at identical strengths must yield
        # different virtual item names, or optimize_merge's in-node cache
        # would false-hit when the user swaps a LoRA file.
        names_per_run = []
        for _ in range(2):
            model = self._model(_chain_patches(
                (0.8, {"a": _adapter()}), (0.5, {"b": _adapter()})))
            model.patches_uuid = uuid.uuid4()
            node = self._node()
            seen = {}
            self._capture_merge(node, seen)
            node.execute_inline(model, output_strength=1.0)
            names_per_run.append([i["name"] for i in seen["stack"]])
        self.assertNotEqual(names_per_run[0], names_per_run[1])
        for names in names_per_run:
            self.assertTrue(names[0].startswith("chain lora #1"))
            self.assertTrue(names[1].startswith("chain lora #2"))

    def test_salt_names_stable_for_same_patcher(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        model.patches_uuid = uuid.uuid4()
        node = self._node()
        names_per_run = []
        def fake_merge(m, stack, o, **kw):
            names_per_run.append([i["name"] for i in stack])
            return (m, None, "r", None, None)
        node.optimize_merge = fake_merge
        node.execute_inline(model, output_strength=1.0)
        node.execute_inline(model, output_strength=1.0)
        self.assertEqual(names_per_run[0], names_per_run[1])

    def test_input_types_surface(self):
        cls = lora_optimizer.LoRAOptimizerInline
        it = cls.INPUT_TYPES()
        req, opt = it["required"], it["optional"]
        # widgets the user types live in "required"
        for w in ("model", "output_strength", "clip_strength_multiplier"):
            self.assertIn(w, req)
        # per-LoRA option widgets moved to LoRAInlineChainOptions — the inline
        # node no longer carries settings_visibility/lora_count or any slot
        self.assertNotIn("settings_visibility", req)
        self.assertNotIn("lora_count", req)
        for base in ("enabled", "strength", "model_strength", "clip_strength",
                     "conflict_mode", "key_filter", "preserve"):
            self.assertNotIn(f"{base}_1", req)
        # genuine node-to-node wires stay optional
        self.assertIn("clip", opt)
        self.assertIn("settings", opt)
        self.assertIn("chain_options", opt)           # new side-node input
        self.assertEqual(cls.FUNCTION, "execute_inline")
        # inherited from LoRAOptimizer — lora_data keeps SaveMergedLoRA working
        self.assertEqual(cls.RETURN_TYPES,
                         ("MODEL", "CLIP", "STRING", "TUNER_DATA", "LORA_DATA"))
        self.assertNotIn("RETURN_TYPES", vars(cls))   # not redeclared


class TestInlineIsChanged(unittest.TestCase):
    """LoRAOptimizerInline.IS_CHANGED must accept the node's ACTUAL (now much
    smaller) inputs — the inherited LoRAOptimizer signature expects lora_stack
    and would raise -> node re-merges on every queue press — and key on the
    upstream patcher state + chain_options content + settings."""

    def _model(self):
        m = _FakePatcher(_chain_patches((0.8, {"a": _adapter()})))
        m.patches_uuid = uuid.uuid4()
        return m

    def test_accepts_actual_inputs_and_is_stable(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        first = cls.IS_CHANGED(model, 1.0)      # must not raise
        self.assertEqual(first, cls.IS_CHANGED(model, 1.0))

    def test_changes_when_chain_options_content_changes(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        co = {"visibility": "simple", "slots": [_slot()]}
        first = cls.IS_CHANGED(model, 1.0, chain_options=co)
        self.assertEqual(first, cls.IS_CHANGED(model, 1.0, chain_options=co))
        changed = {"visibility": "simple", "slots": [_slot(enabled=False)]}
        self.assertNotEqual(first,
                            cls.IS_CHANGED(model, 1.0, chain_options=changed))

    def test_changes_when_upstream_chain_changes(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        first = cls.IS_CHANGED(model, 1.0)
        model.patches_uuid = uuid.uuid4()       # loader chain re-executed
        self.assertNotEqual(first, cls.IS_CHANGED(model, 1.0))

    def test_changes_with_output_strength_and_clip_multiplier(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        first = cls.IS_CHANGED(model, 1.0)
        self.assertNotEqual(first, cls.IS_CHANGED(model, 0.5))
        self.assertNotEqual(first, cls.IS_CHANGED(
            model, 1.0, clip_strength_multiplier=0.5))

    def test_changes_with_clip_and_settings(self):
        cls = lora_optimizer.LoRAOptimizerInline
        model = self._model()
        first = cls.IS_CHANGED(model, 1.0)
        clip = _FakeCLIP(_FakePatcher(_chain_patches((0.6, {"te.a": _adapter()}))))
        clip.patcher.patches_uuid = uuid.uuid4()
        with_clip = cls.IS_CHANGED(model, 1.0, clip=clip)
        self.assertNotEqual(first, with_clip)
        clip.patcher.patches_uuid = uuid.uuid4()
        self.assertNotEqual(with_clip, cls.IS_CHANGED(model, 1.0, clip=clip))
        self.assertNotEqual(first, cls.IS_CHANGED(
            model, 1.0, settings={"mode": "advanced"}))


def _realistic_load_lora_for_models(calls):
    """A comfy.sd.load_lora_for_models stand-in that mimics the REAL failure
    mode instead of blindly passing the model through: it resolves only
    TRAINER-format keys ({x}.lora_up.weight / .lora_down.weight). Virtual
    chain items carry model-target keys, so nothing matches and the models
    come back unchanged — exactly what live comfy does, silently."""
    def stub(model, clip, lora_dict, strength_model, strength_clip):
        matched = [k for k in lora_dict
                   if isinstance(k, str) and (k.endswith(".lora_up.weight")
                                              or k.endswith(".lora_down.weight"))]
        calls.append({"matched": matched, "n_keys": len(lora_dict)})
        return (model, clip)   # nothing baked for unmatched keys
    return stub


def _pipeline_model(patches, applied, dim=16):
    """Fake ModelPatcher threading the REAL optimize_merge pipeline: a .model
    with a dim x dim base weight so _resolve_target_shape/_resolve_base_norm
    work, faithful clone() (copies the CURRENT patch lists, like the real
    ModelPatcher.clone — so a clone of the stripped clone stays stripped),
    and an add_patches recorder capturing the merged patches the engine
    re-applies (plus which patcher received them)."""
    base = types.SimpleNamespace(
        layer=types.SimpleNamespace(weight=torch.zeros(dim, dim)))

    def _attach(p):
        p.model = base
        p.add_patches = lambda patches_, strength=1.0, strength_clip=None: (
            applied.update(patches=dict(patches_), strength=strength,
                           patcher=p),
            list(patches_.keys()))[1]
        p.clone = lambda: _attach(
            _FakePatcher({k: v[:] for k, v in p.patches.items()}))
        return p

    return _attach(_FakePatcher(patches))


def _pipeline_clip(patches, applied, dim=16):
    """CLIP counterpart to _pipeline_model with a real clone/add_patches
    recording path and a readable base target for _resolve_target_shape."""
    base = types.SimpleNamespace(
        clip_layer=types.SimpleNamespace(weight=torch.zeros(dim, dim)))

    def _attach_patcher(p):
        p.clone = lambda: _attach_patcher(
            _FakePatcher({k: v[:] for k, v in p.patches.items()}))
        return p

    class _PipelineCLIP:
        def __init__(self, patcher):
            self.patcher = patcher
            self.cond_stage_model = base

        def clone(self):
            return _PipelineCLIP(self.patcher.clone())

        def add_patches(self, patches_, strength=1.0, strength_clip=None):
            applied.update(patches=dict(patches_), strength=strength,
                           clip=self)
            return list(patches_.keys())

    return _PipelineCLIP(_attach_patcher(_FakePatcher(patches)))


class TestSingleLoraVirtualPath(unittest.TestCase):
    """A 1-LoRA chain (or N LoRAs with all but one disabled) must NOT take
    optimize_merge's single-LoRA fast path: load_lora_for_models looks up
    trainer-format keys, virtual items carry model-target keys, so it would
    load NOTHING — and with the originals stripped, the output model would
    silently lose the LoRA entirely."""

    def _fake_model(self, patches, applied):
        return _pipeline_model(patches, applied)

    def test_single_virtual_lora_goes_through_pipeline(self):
        key = "layer.weight"
        applied = {}
        model = self._fake_model(
            _chain_patches((0.9, {key: _adapter(rank=4, out_dim=16, in_dim=16)})),
            applied)
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": key}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            result = node.execute_inline(model, output_strength=1.0)
        out = result["result"] if isinstance(result, dict) else result
        self.assertEqual(calls, [])               # fast path NOT taken
        self.assertTrue(applied.get("patches"))   # pipeline re-applied patches
        self.assertIn("chain lora #1", out[2])

    def test_disabled_down_to_single_also_uses_pipeline(self):
        key = "layer.weight"
        applied = {}
        model = self._fake_model(
            _chain_patches((0.9, {key: _adapter(rank=4, out_dim=16, in_dim=16)}),
                           (0.5, {key: _adapter(rank=4, out_dim=16, in_dim=16)})),
            applied)
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": key}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            node.execute_inline(model, output_strength=1.0,
                                chain_options={"visibility": "simple",
                                               "slots": [_slot(),
                                                         _slot(enabled=False)]})
        self.assertEqual(calls, [])
        self.assertTrue(applied.get("patches"))

    def test_plain_single_lora_stack_emits_patch_data(self):
        # Single files share the filtered/exportable path with multi-LoRA stacks.
        model = _FakePatcher({})
        model.model = types.SimpleNamespace(
            layer=types.SimpleNamespace(weight=torch.zeros(16, 16)))
        stack = [{"name": "A", "strength": 1.0,
                  "lora": {"alias_layer.lora_up.weight": torch.randn(16, 4),
                           "alias_layer.lora_down.weight": torch.randn(4, 16),
                           "alias_layer.alpha": torch.tensor(4.0)}}]
        node = lora_optimizer.LoRAOptimizer()
        node._get_model_keys = lambda m: {"alias_layer": "layer.weight"}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            result = node.optimize_merge(model, stack, 1.0)
        self.assertEqual(calls, [])
        patch = result[4]["model_patches"]["layer.weight"]
        torch.testing.assert_close(node._expand_patch_to_diff(patch),
                                   stack[0]["lora"]["alias_layer.lora_up.weight"] @
                                   stack[0]["lora"]["alias_layer.lora_down.weight"])


class TestVirtualDiffDeviceExpansion(unittest.TestCase):
    """Commit B: captured (virtual) diffs are expanded on the COMPUTE device.
    The small low-rank factors move to the device BEFORE the up@down matmul, so
    the matmul runs on-device and only the factors cross the bus — not the big
    dense [out x in] result from a CPU matmul. Pure device move: the merged
    result is unchanged (allclose vs the plain CPU expand)."""

    KEY = "layer.weight"

    def _group(self):
        return {"target_key": self.KEY, "is_clip": False,
                "aliases": [self.KEY], "label_prefix": self.KEY}

    def test_cpu_result_matches_direct_expand(self):
        # Output invariance on CPU: prepared diff == plain up@down expand.
        up = torch.randn(8, 2)
        down = torch.randn(2, 8)
        item = _virtual_item({self.KEY: _lora_adapter(up, down, 2.0)})  # scale 1
        opt = lora_optimizer.LoRAOptimizer()
        prepared = opt._prepare_group_diffs(
            self._group(), [item], _layer_model(), None,
            torch.device("cpu"), auto_scale=1.0)
        self.assertTrue(torch.allclose(prepared["diffs"][0], up @ down, atol=1e-6))

    def test_expand_runs_on_compute_device_not_cpu(self):
        # Teeth without a GPU: 'meta' is a real non-cpu device carrying no data,
        # so the up@down matmul's INPUT device is observable. Factor-first move
        # (commit B) => the mm runs on 'meta'. The old code expanded on the CPU
        # factors then shipped the dense result, so its mm ran on 'cpu'.
        up = torch.randn(8, 2)
        down = torch.randn(2, 8)
        item = _virtual_item({self.KEY: _lora_adapter(up, down, 2.0)})
        opt = lora_optimizer.LoRAOptimizer()
        mm_devices = []
        orig_mm = torch.mm

        def spy_mm(a, b, *args, **kwargs):
            mm_devices.append(a.device.type)
            return orig_mm(a, b, *args, **kwargs)

        with mock.patch("torch.mm", spy_mm):
            opt._prepare_group_diffs(
                self._group(), [item], _layer_model(), None,
                torch.device("meta"), auto_scale=1.0)
        self.assertIn("meta", mm_devices)     # matmul ran on the compute device
        self.assertNotIn("cpu", mm_devices)   # not a CPU matmul + dense transfer

    def test_cpu_device_never_moves_factors(self):
        # Guard: for a CPU compute device there is no bus to cross, so the
        # factor-move helper must NOT be invoked (use_gpu is False). Still works.
        up = torch.randn(8, 2)
        down = torch.randn(2, 8)
        item = _virtual_item({self.KEY: _lora_adapter(up, down, 2.0)})
        opt = lora_optimizer.LoRAOptimizer()
        moves = []
        orig_move = lora_optimizer._LoRAMergeBase._move_patch_to_device

        def spy_move(patch, device):
            moves.append(device)
            return orig_move(patch, device)

        with mock.patch.object(lora_optimizer._LoRAMergeBase,
                               "_move_patch_to_device",
                               staticmethod(spy_move)):
            prepared = opt._prepare_group_diffs(
                self._group(), [item], _layer_model(), None,
                torch.device("cpu"), auto_scale=1.0)
        self.assertEqual(moves, [])
        self.assertTrue(torch.allclose(prepared["diffs"][0], up @ down, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_gpu_expand_on_device_matches_cpu(self):
        up = torch.randn(8, 2)
        down = torch.randn(2, 8)
        item = _virtual_item({self.KEY: _lora_adapter(up, down, 2.0)})
        opt = lora_optimizer.LoRAOptimizer()
        prepared = opt._prepare_group_diffs(
            self._group(), [item], _layer_model(), None,
            torch.device("cuda"), auto_scale=1.0)
        got = prepared["diffs"][0]
        self.assertEqual(got.device.type, "cuda")
        self.assertTrue(torch.allclose(got.cpu(), up @ down, atol=1e-5))


class TestVirtualLinearFastPath(unittest.TestCase):
    """Commit A: captured LoRAAdapter chains take the exact low-rank concat
    fast path (like file items) instead of materializing a dense diff per
    contributor. The emitted low-rank patch, reconstructed, must equal the
    dense _prepare_group_diffs + _merge_diffs result bit-for-bit (float tol)."""

    KEY = "layer.weight"

    def setUp(self):
        self.opt = lora_optimizer.LoRAOptimizer()

    def _group(self):
        return {"target_key": self.KEY, "is_clip": False,
                "aliases": [self.KEY], "label_prefix": self.KEY}

    def _dense(self, active, mode, model, scale=1.0):
        prepared = self.opt._prepare_group_diffs(
            self._group(), active, model, None, torch.device("cpu"),
            auto_scale=scale)
        idx = sorted(prepared["diffs"])
        diffs_list = [(prepared["diffs"][i], prepared["eff_strengths"][i])
                      for i in idx]
        return self.opt._merge_diffs(diffs_list, mode)

    def _fast(self, active, mode, model, scale=1.0):
        info = self.opt._build_exact_linear_patch(
            self._group(), active, len(active), mode, model_scale=scale)
        self.assertIsNotNone(info, "fast path unexpectedly fell back to dense")
        return info, self.opt._expand_patch_to_diff(info["patch"])

    def _assert_equiv(self, active, mode, model, scale=1.0, atol=5e-6):
        dense = self._dense(active, mode, model, scale)
        _info, recon = self._fast(active, mode, model, scale)
        self.assertEqual(tuple(recon.shape), tuple(dense.shape))
        self.assertTrue(
            torch.allclose(recon, dense, atol=atol, rtol=0.0),
            f"mode={mode} scale={scale} max err "
            f"{(recon - dense).abs().max().item()}")

    # ---- numerical equivalence across adapter configs --------------------
    def test_single_lora_weighted_sum(self):
        up = torch.randn(8, 2)
        down = torch.randn(2, 8)
        active = [_virtual_item({self.KEY: _lora_adapter(up, down, 2.0)},
                                strength=0.8)]
        self._assert_equiv(active, "weighted_sum", _layer_model())

    def test_two_lora_weighted_sum_alpha_eq_rank(self):
        active = [
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 2),
                                                   torch.randn(2, 8), 2.0)},
                          strength=1.0),
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 3),
                                                   torch.randn(3, 8), 3.0)},
                          strength=0.7),
        ]
        self._assert_equiv(active, "weighted_sum", _layer_model())

    def test_two_lora_weighted_average(self):
        active = [
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 2),
                                                   torch.randn(2, 8), 2.0)},
                          strength=1.3),
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 2),
                                                   torch.randn(2, 8), 2.0)},
                          strength=0.4),
        ]
        self._assert_equiv(active, "weighted_average", _layer_model())

    def test_two_lora_normalize(self):
        active = [
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 2),
                                                   torch.randn(2, 8), 2.0)},
                          strength=1.1),
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 2),
                                                   torch.randn(2, 8), 2.0)},
                          strength=0.9),
        ]
        self._assert_equiv(active, "normalize", _layer_model())

    def test_alpha_none_scale_one(self):
        # alpha=None -> scale 1.0 in both paths
        active = [
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 2),
                                                   torch.randn(2, 8), None)},
                          strength=1.0),
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 2),
                                                   torch.randn(2, 8), None)},
                          strength=0.6),
        ]
        self._assert_equiv(active, "weighted_sum", _layer_model())

    def test_alpha_ne_rank_rescales(self):
        # rank 4 but alpha 2.0 -> scale 0.5, must fold identically
        active = [
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 4),
                                                   torch.randn(4, 8), 2.0)},
                          strength=1.0),
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 4),
                                                   torch.randn(4, 8), 8.0)},
                          strength=0.5),
        ]
        self._assert_equiv(active, "weighted_average", _layer_model())

    def test_model_scale_applied_identically(self):
        active = [
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 2),
                                                   torch.randn(2, 8), 2.0)},
                          strength=1.0),
            _virtual_item({self.KEY: _lora_adapter(torch.randn(8, 3),
                                                   torch.randn(3, 8), 3.0)},
                          strength=0.7),
        ]
        self._assert_equiv(active, "weighted_sum", _layer_model(), scale=0.85)

    def test_alpha_as_tensor(self):
        # comfy sometimes stores alpha as a 0-d tensor
        active = [
            _virtual_item({self.KEY: _lora_adapter(
                torch.randn(8, 2), torch.randn(2, 8), torch.tensor(2.0))},
                strength=1.0),
        ]
        self._assert_equiv(active, "weighted_sum", _layer_model())

    def test_reduced_precision_factors_are_scaled_in_float32(self):
        # Comfy casts adapter factors to its float32 intermediate dtype before
        # applying alpha/strength.  Scaling the source BF16/FP16 up matrix first
        # irreversibly rounds it and makes Inline additive differ from Load LoRA.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                up_a = torch.linspace(-1.0, 1.0, 8 * 4).reshape(8, 4).to(dtype)
                down_a = torch.linspace(0.7, -0.4, 4 * 8).reshape(4, 8).to(dtype)
                up_b = torch.linspace(-0.3, 1.2, 8 * 4).reshape(8, 4).to(dtype)
                down_b = torch.linspace(-0.8, 0.5, 4 * 8).reshape(4, 8).to(dtype)
                active = [
                    _virtual_item({self.KEY: _lora_adapter(
                        up_a, down_a, 2.0)}, strength=0.37),
                    _virtual_item({self.KEY: _lora_adapter(
                        up_b, down_b, 6.0)}, strength=-0.61),
                ]
                self._assert_equiv(
                    active, "weighted_sum", _layer_model(), atol=5e-6)

    # ---- fallback: non-qualifying payloads keep the dense path -----------
    def _lokr(self):
        return lora_optimizer.LoKrAdapter(set(), (
            torch.randn(8, 8), torch.randn(1, 1), 1.0,
            None, None, None, None, None, None))

    def test_virtual_payload_is_linear_ok_positive(self):
        p = _lora_adapter(torch.randn(8, 2), torch.randn(2, 8), 2.0)
        self.assertTrue(self.opt._virtual_payload_is_linear_ok(p))

    def test_virtual_payload_is_linear_ok_rejects_non_adapters(self):
        f = self.opt._virtual_payload_is_linear_ok
        self.assertFalse(f(self._lokr()))
        self.assertFalse(f(("diff", (torch.randn(8, 8),))))
        self.assertFalse(f(torch.randn(8, 8)))
        # mid != None (LoCon)
        self.assertFalse(f(lora_optimizer.LoRAAdapter(
            set(), (torch.randn(8, 2, 1, 1), torch.randn(2, 8),
                    2.0, torch.randn(2, 2, 3, 3), None, None))))
        # non-2D up
        self.assertFalse(f(lora_optimizer.LoRAAdapter(
            set(), (torch.randn(8, 2, 1, 1), torch.randn(2, 8, 1, 1),
                    2.0, None, None, None))))
        # Base-dependent/shape-changing LoRA metadata must never enter the
        # supposedly exact additive concat path.
        self.assertFalse(f(lora_optimizer.LoRAAdapter(
            set(), (torch.randn(8, 2), torch.randn(2, 8), 2.0, None,
                    torch.ones(8, 1), None))))
        self.assertFalse(f(lora_optimizer.LoRAAdapter(
            set(), (torch.randn(9, 2), torch.randn(2, 8), 2.0, None,
                    None, [9, 8]))))

    def test_group_ok_true_for_pure_file_group(self):
        # No virtual contributor -> True (file path stays exactly as before)
        file_item = {"name": "F", "strength": 1.0, "clip_strength": None,
                     "lora": {}}
        self.assertTrue(self.opt._virtual_group_is_linear_ok(
            self._group(), [file_item], _layer_model(), None))

    def test_group_ok_false_for_lokr_contributor(self):
        active = [_virtual_item({self.KEY: self._lokr()})]
        self.assertFalse(self.opt._virtual_group_is_linear_ok(
            self._group(), active, _layer_model(), None))
        # and _build_exact_linear_patch bails -> dense path
        self.assertIsNone(self.opt._build_exact_linear_patch(
            self._group(), active, 1, "weighted_sum"))

    def test_group_ok_false_for_dense_tensor_contributor(self):
        active = [_virtual_item({self.KEY: torch.randn(8, 8)})]
        self.assertFalse(self.opt._virtual_group_is_linear_ok(
            self._group(), active, _layer_model(), None))
        self.assertIsNone(self.opt._build_exact_linear_patch(
            self._group(), active, 1, "weighted_sum"))

    def test_group_ok_false_for_tuple_target_key(self):
        # offset-sliced (QKV refusion) target keys stay dense
        tg = {"target_key": (self.KEY, (0, 0, 4)), "is_clip": False,
              "aliases": [self.KEY], "label_prefix": self.KEY}
        active = [_virtual_item(
            {self.KEY: _lora_adapter(torch.randn(8, 2), torch.randn(2, 8), 2.0)})]
        self.assertFalse(self.opt._virtual_group_is_linear_ok(
            tg, active, _layer_model(), None))

    def test_group_ok_false_when_target_not_2d(self):
        # 4D conv target: dense path reshapes, fast path can't -> keep dense
        model = types.SimpleNamespace(model=types.SimpleNamespace(
            layer=types.SimpleNamespace(weight=torch.zeros(8, 8, 1, 1))))
        active = [_virtual_item(
            {self.KEY: _lora_adapter(torch.randn(8, 2), torch.randn(2, 8), 2.0)})]
        self.assertFalse(self.opt._virtual_group_is_linear_ok(
            self._group(), active, model, None))


class TestChainOptionsNode(unittest.TestCase):
    """The side node carries the per-LoRA widgets and emits a
    LORA_CHAIN_OPTIONS payload the inline node consumes by chain order."""

    def _node(self):
        return lora_optimizer.LoRAInlineChainOptions()

    @staticmethod
    def _full_slot_kwargs():
        """One distinct value per widget across all 10 slots so both the
        per-slot mapping and lora_count truncation are observable."""
        cls = lora_optimizer.LoRAInlineChainOptions
        kw = {}
        for i in range(1, cls.MAX_LORAS + 1):
            kw[f"enabled_{i}"] = (i % 2 == 0)
            kw[f"strength_{i}"] = float(i)
            kw[f"model_strength_{i}"] = float(i) + 0.1
            kw[f"clip_strength_{i}"] = float(i) + 0.2
            kw[f"conflict_mode_{i}"] = "all"
            kw[f"key_filter_{i}"] = "all"
            kw[f"preserve_{i}"] = (i % 2 == 1)
        return kw

    def test_returns_visibility_passthrough_and_slot_count(self):
        node = self._node()
        (out,) = node.build_options("advanced", 3, **self._full_slot_kwargs())
        self.assertEqual(out["visibility"], "advanced")
        self.assertEqual(len(out["slots"]), 3)

    def test_slot_dicts_have_all_seven_keys_with_passed_values(self):
        node = self._node()
        (out,) = node.build_options("simple", 2, **self._full_slot_kwargs())
        expected_keys = set(lora_optimizer.LoRAOptimizerInline._SLOT_DEFAULTS)
        self.assertEqual(len(expected_keys), 7)
        for i, slot in enumerate(out["slots"], start=1):
            self.assertEqual(set(slot), expected_keys)
            self.assertEqual(slot["strength"], float(i))
            self.assertEqual(slot["model_strength"], float(i) + 0.1)
            self.assertEqual(slot["clip_strength"], float(i) + 0.2)
            self.assertEqual(slot["enabled"], (i % 2 == 0))
            self.assertEqual(slot["preserve"], (i % 2 == 1))

    def test_lora_count_truncates_slots(self):
        node = self._node()
        (out,) = node.build_options("simple", 2, **self._full_slot_kwargs())
        self.assertEqual(len(out["slots"]), 2)  # only 2 though 10 widgets set

    def test_missing_widget_uses_slot_default(self):
        # widgets not passed fall back to _SLOT_DEFAULTS (single source)
        node = self._node()
        (out,) = node.build_options("simple", 1)
        self.assertEqual(out["slots"][0],
                         dict(lora_optimizer.LoRAOptimizerInline._SLOT_DEFAULTS))

    def test_input_types_surface(self):
        cls = lora_optimizer.LoRAInlineChainOptions
        it = cls.INPUT_TYPES()
        req = it["required"]
        self.assertIn("settings_visibility", req)
        self.assertIn("lora_count", req)
        for i in (1, cls.MAX_LORAS):
            for base in ("enabled", "strength", "model_strength",
                         "clip_strength", "conflict_mode", "key_filter",
                         "preserve"):
                self.assertIn(f"{base}_{i}", req)
        self.assertNotIn("optional", it)  # pure widget node, no sockets
        self.assertEqual(cls.FUNCTION, "build_options")
        self.assertEqual(cls.RETURN_TYPES, ("LORA_CHAIN_OPTIONS",))
        self.assertEqual(cls.RETURN_NAMES, ("chain_options",))


class TestRegistration(unittest.TestCase):
    def test_inline_node_registered(self):
        self.assertIn("LoRAOptimizerInline", lora_optimizer.NODE_CLASS_MAPPINGS)
        self.assertIs(lora_optimizer.NODE_CLASS_MAPPINGS["LoRAOptimizerInline"],
                      lora_optimizer.LoRAOptimizerInline)
        self.assertIn("LoRAOptimizerInline",
                      lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS["LoRAOptimizerInline"],
            "LoRA Optimizer (Inline Chain)")

    def test_options_node_registered(self):
        self.assertIn("LoRAInlineChainOptions",
                      lora_optimizer.NODE_CLASS_MAPPINGS)
        self.assertIs(
            lora_optimizer.NODE_CLASS_MAPPINGS["LoRAInlineChainOptions"],
            lora_optimizer.LoRAInlineChainOptions)
        self.assertEqual(
            lora_optimizer.NODE_DISPLAY_NAME_MAPPINGS["LoRAInlineChainOptions"],
            "LoRA Inline Chain Options")


def _advanced_settings(**overrides):
    """Full OPTIMIZER_SETTINGS advanced-mode dict with deterministic,
    test-friendly engine knobs (no merge cache, no compression SVD, CPU SVD).
    The settings input is the node's supported surface for engine kwargs —
    execute_inline deliberately does not forward loose **kwargs to
    optimize_merge."""
    settings = {
        "mode": "advanced",
        "auto_strength": "disabled", "auto_strength_floor": -1.0,
        "optimization_mode": "per_prefix",
        "sparsification": "disabled", "sparsification_density": 0.7,
        "dare_dampening": 0.0,
        "merge_strategy_override": "",
        "merge_refinement": "none", "strategy_set": "full",
        "normalize_keys": "disabled",
        "architecture_preset": "dit", "decision_smoothing": 0.25,
        "smooth_slerp_gate": False,
        "star_eta": 100.0, "tame_layers": 0.0, "tame_threshold": 0.3,
        "cache_patches": "disabled", "patch_compression": "disabled",
        "svd_device": "cpu", "free_vram_between_passes": "disabled",
        "vram_budget": 0.0,
    }
    settings.update(overrides)
    return settings


class TestInlineClipMultiplierEndToEnd(unittest.TestCase):
    KEY = "clip_layer.weight"

    def _run(self, multiplier):
        applied = {}
        model = _pipeline_model({}, {})
        up_a = torch.linspace(-1.0, 1.0, 16 * 4).reshape(16, 4)
        down_a = torch.linspace(0.8, -0.4, 4 * 16).reshape(4, 16)
        up_b = torch.linspace(-0.2, 1.1, 16 * 4).reshape(16, 4)
        down_b = torch.linspace(-0.7, 0.6, 4 * 16).reshape(4, 16)
        clip = _pipeline_clip(_chain_patches(
            (0.8, {self.KEY: _adapter(up=up_a, down=down_a)}),
            (0.6, {self.KEY: _adapter(up=up_b, down=down_b)}),
        ), applied)
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"model_alias": "layer.weight"}
        settings = _advanced_settings(
            optimization_mode="global",
            merge_strategy_override="weighted_average")
        with mock.patch.object(
                lora_optimizer.comfy.lora, "model_lora_keys_clip",
                return_value={"clip_alias": self.KEY}):
            node.execute_inline(
                model, output_strength=1.0, clip=clip,
                clip_strength_multiplier=multiplier, settings=settings)
        patch = applied["patches"][self.KEY]
        return lora_optimizer._LoRAMergeBase._expand_patch_to_diff(patch), \
            applied["strength"]

    def test_multiplier_scales_normalized_merge_once_at_apply(self):
        full_patch, full_apply = self._run(1.0)
        half_patch, half_apply = self._run(0.5)
        # weighted_average normalizes relative per-LoRA weights, so its patch
        # is intentionally unchanged; the node-wide control must scale the
        # final CLIP application once instead of disappearing in normalization.
        self.assertTrue(torch.allclose(full_patch, half_patch, atol=1e-6,
                                       rtol=0.0))
        self.assertAlmostEqual(full_apply, 1.0)
        self.assertAlmostEqual(half_apply, 0.5)


class TestMultiLoraEndToEnd(unittest.TestCase):
    """Captured MULTI-LoRA chains through the REAL optimize_merge: two rank-4
    adapters on the same 16x16 layer key, different loader strengths. The
    single-LoRA virtual path is covered above; this closes the >= 2-LoRA gap
    (full Pass 1 conflict analysis + Pass 2 group merge on virtual items)."""

    KEY = "layer.weight"

    @staticmethod
    def _det_mats(seed_row):
        """Deterministic rank-2 up/down matrices (each linspace reshape has
        rows/columns affine in the index, so the span is {1, arange} — rank 2;
        deterministic on purpose, no randn luck)."""
        up = torch.linspace(-1.0 + seed_row, 1.0 + seed_row, 16 * 4).reshape(16, 4)
        down = torch.linspace(0.5 - seed_row, -0.5 + seed_row, 4 * 16).reshape(4, 16)
        return up, down

    def _run(self, chain, settings=None, output_strength=1.0, **exec_kwargs):
        applied = {}
        model = _pipeline_model(_chain_patches(*chain), applied)
        # Snapshot the loader entries BEFORE execute_inline runs — comparing
        # a post-run snapshot to the post-run state would be tautological and
        # could never catch the node mutating its input model's patch lists.
        orig_entries = {k: list(v) for k, v in model.patches.items()}
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": self.KEY}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            result = node.execute_inline(model, output_strength=output_strength,
                                         settings=settings, **exec_kwargs)
        out = result["result"] if isinstance(result, dict) else result
        return model, applied, calls, out, orig_entries

    def test_two_lora_chain_weighted_sum_is_numerically_exact(self):
        up_a, down_a = self._det_mats(0.0)
        up_b, down_b = self._det_mats(0.25)
        chain = (
            (1.0, {self.KEY: _adapter(up=up_a, down=down_a)}),
            (0.7, {self.KEY: _adapter(up=up_b, down=down_b)}),
        )
        # weighted_sum override makes the expected result exact:
        # merged = 1.0 * (up_a @ down_a) + 0.7 * (up_b @ down_b)
        settings = _advanced_settings(optimization_mode="global",
                                      merge_strategy_override="weighted_sum")
        model, applied, calls, out, orig_entries = self._run(
            chain, settings=settings)

        # (a) report opens with the chain fingerprints, engine report follows
        report = out[2]
        self.assertIn("#1: 1 keys, rank 4, loader strength 1.00", report)
        self.assertIn("#2: 1 keys, rank 4, loader strength 0.70", report)
        self.assertIn("chain lora #1", report)

        # (b) merged patches re-applied via add_patches on a clone
        self.assertEqual(calls, [])                      # no fast-path bypass
        self.assertIn("patches", applied)
        self.assertIsNot(applied["patcher"], model)
        self.assertIs(out[0], applied["patcher"])        # patched clone returned
        self.assertAlmostEqual(applied["strength"], 1.0)

        # (c) the applied patch is the exact weighted combination
        self.assertEqual(set(applied["patches"]), {self.KEY})
        # ...and it is a low-rank LoRAAdapter, proving the virtual-aware fast
        # path actually engaged for this qualifying group (weighted_sum, 2D
        # linear, plain adapters) rather than silently falling back to the
        # dense ("diff", tensor) path.
        self.assertIsInstance(applied["patches"][self.KEY],
                              lora_optimizer.LoRAAdapter)
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        expected = up_a @ down_a + 0.7 * (up_b @ down_b)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5),
                        f"max err {(got - expected).abs().max().item()}")

        # (d) the original input model still carries its loader entries
        # (orig_entries snapshotted pre-run in _run, so this has teeth)
        self.assertEqual(model.patches[self.KEY], orig_entries[self.KEY])
        self.assertEqual(len(model.patches[self.KEY]), 2)

    def test_output_strength_scales_at_apply_not_in_patches(self):
        # Same exact-weighted-sum setup, but output_strength=0.5: the master
        # volume must ride on add_patches ONCE and never ALSO be baked into
        # the patch tensors (which would double-scale the merged result).
        up_a, down_a = self._det_mats(0.0)
        up_b, down_b = self._det_mats(0.25)
        chain = (
            (1.0, {self.KEY: _adapter(up=up_a, down=down_a)}),
            (0.7, {self.KEY: _adapter(up=up_b, down=down_b)}),
        )
        settings = _advanced_settings(optimization_mode="global",
                                      merge_strategy_override="weighted_sum")
        model, applied, calls, out, orig_entries = self._run(
            chain, settings=settings, output_strength=0.5)

        self.assertEqual(calls, [])
        self.assertAlmostEqual(applied["strength"], 0.5)
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        # UNscaled weighted sum — identical to the output_strength=1.0 case
        expected = up_a @ down_a + 0.7 * (up_b @ down_b)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5),
                        f"max err {(got - expected).abs().max().item()}")

    def test_none_alpha_captured_adapters_merge_without_crash(self):
        # End-to-end guard: a chain of PEFT/diffusers LoRAs (alpha=None on the
        # captured comfy adapters) must merge and re-apply without a
        # TypeError, treating None alpha as scale 1.0.
        up_a, down_a = self._det_mats(0.0)
        up_b, down_b = self._det_mats(0.25)
        chain = (
            (1.0, {self.KEY: _adapter(up=up_a, down=down_a, alpha=None)}),
            (0.7, {self.KEY: _adapter(up=up_b, down=down_b, alpha=None)}),
        )
        settings = _advanced_settings(optimization_mode="global",
                                      merge_strategy_override="weighted_sum")
        model, applied, calls, out, orig_entries = self._run(
            chain, settings=settings)
        self.assertEqual(calls, [])
        self.assertIn("patches", applied)
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        self.assertTrue(torch.isfinite(got).all())
        # alpha=None -> scale 1.0, so the exact weighted sum is unchanged
        expected = up_a @ down_a + 0.7 * (up_b @ down_b)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5),
                        f"max err {(got - expected).abs().max().item()}")

    def test_conflicting_loras_complete_through_conflict_analysis(self):
        # Opposite-sign directions built deliberately: B is A negated with a
        # small deterministic perturbation (near-antiparallel, cos ~ -1) so
        # the per_prefix conflict path really engages without relying on a
        # degenerate exactly-antiparallel corner.
        up_a, down_a = self._det_mats(0.0)
        down_b = down_a.clone()
        down_b[0] += 0.05
        chain = (
            (1.0, {self.KEY: _adapter(up=up_a, down=down_a)}),
            (0.7, {self.KEY: _adapter(up=-up_a, down=down_b)}),
        )
        model, applied, calls, out, orig_entries = self._run(
            chain, settings=_advanced_settings())

        self.assertEqual(calls, [])
        report = out[2]
        self.assertIn("#1: 1 keys, rank 4, loader strength 1.00", report)
        self.assertIn("#2: 1 keys, rank 4, loader strength 0.70", report)
        # merged patches exist and are finite/nonzero for the shared key
        self.assertEqual(set(applied["patches"]), {self.KEY})
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        self.assertEqual(tuple(got.shape), (16, 16))
        self.assertTrue(torch.isfinite(got).all())
        self.assertGreater(got.abs().max().item(), 0.0)
        # direction teeth: A (strength 1.0) beats the weaker near-antiparallel
        # B (0.7), so any sane conflict resolution must land on A's side of
        # the axis — positive cosine alignment with A's diff.
        diff_a = (up_a @ down_a).flatten()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), diff_a, dim=0).item()
        self.assertGreater(cos, 0.0,
                           f"merged patch flipped to the weaker LoRA's "
                           f"direction (cos={cos:.4f})")
        # original chain untouched on the input model (pre-run snapshot)
        self.assertEqual(model.patches[self.KEY], orig_entries[self.KEY])
        self.assertEqual(len(model.patches[self.KEY]), 2)


class TestDenseAdditiveSmartCompression(unittest.TestCase):
    """Smart compression must not make the advertised exact additive mode
    lossy when a captured dense delta has no trustworthy low-rank bound."""

    KEY = "layer.weight"

    def test_full_rank_dense_delta_above_rank64_remains_exact(self):
        dim = 80
        # Exactly full-rank with 16 singular directions beyond the historical
        # rank-64 floor.  A truncated smart-compression SVD loses those values
        # by a wide margin, making this a deterministic regression rather than
        # relying on the numerical rank of a random matrix.
        dense = torch.diag(torch.linspace(1.0, 2.0, dim))
        strength = 0.75
        patches = _chain_patches(
            (strength, {self.KEY: ("diff", (dense,))}))
        applied = {}
        model = _pipeline_model(patches, applied, dim=dim)
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": self.KEY}

        settings = _advanced_settings(
            optimization_mode="additive", patch_compression="smart")
        node.execute_inline(model, output_strength=1.0, settings=settings)

        self.assertEqual(set(applied["patches"]), {self.KEY})
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        expected = dense * strength
        self.assertTrue(
            torch.allclose(got, expected, atol=1e-6, rtol=0.0),
            f"smart compression truncated an unknown-rank additive delta; "
            f"max err {(got - expected).abs().max().item()}",
        )


# Model-space target keys the loader chain leaves on the ModelPatcher. These
# still carry structural architecture markers even though they are NOT
# trainer-format LoRA keys, so _detect_architecture's structural heuristics
# resolve them.
_LTX_KEY = "diffusion_model.transformer_blocks.0.attn1.to_q.weight"
_ZIMAGE_KEY = "diffusion_model.layers.0.attention.to_q.weight"
# SDXL UNet cross-attention: has transformer_blocks + attn1 (which naively
# looks like LTX) but lives under input_blocks -> the LTX branch's UNet-block
# exclusion must keep it out of 'ltx' and land it on sd15 -> sd_unet.
_SDXL_UNET_KEY = ("diffusion_model.input_blocks.4.1.transformer_blocks.0"
                  ".attn1.to_q.weight")


class TestVirtualArchDetection(unittest.TestCase):
    """_normalize_stack must fall back to detecting architecture from the
    MODEL-SPACE keys of virtual (_precomputed_diffs) items when no file-based
    item resolves. Inline-captured chains are 100% virtual, so without this
    fallback every inline merge resolved to sd_unet (wrong for DiT models)."""

    def _virtual_stack(self, model_key):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {model_key: _adapter()})))
        return lora_optimizer.LoRAOptimizerInline._chain_groups_to_stack(
            mg, [], [_slot()], "simple")

    def test_ltx_model_keys_detect_ltx(self):
        node = lora_optimizer.LoRAOptimizerInline()
        stack = self._virtual_stack(_LTX_KEY)
        self.assertTrue(all(i["_precomputed_diffs"] for i in stack))
        node._normalize_stack(stack)
        self.assertEqual(node._detected_arch, "ltx")

    def test_zimage_model_keys_detect_zimage(self):
        node = lora_optimizer.LoRAOptimizerInline()
        stack = self._virtual_stack(_ZIMAGE_KEY)
        node._normalize_stack(stack)
        self.assertEqual(node._detected_arch, "zimage")

    def test_tuple_keys_do_not_crash_and_still_detect(self):
        # Fused-QKV captures key the virtual dict by (str_key, offset) TUPLES.
        # _detect_architecture indexes k.lower()/'in k', which would crash on a
        # tuple -> the fallback must stringify the key view first.
        off = (0, 0, 4)
        patches = {_LTX_KEY: [_entry(0.7, _adapter(), offset=off)]}
        groups = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(patches)
        self.assertEqual(list(groups[0]["entries"]), [(_LTX_KEY, off)])
        stack = lora_optimizer.LoRAOptimizerInline._chain_groups_to_stack(
            groups, [], [_slot()], "simple")
        node = lora_optimizer.LoRAOptimizerInline()
        node._normalize_stack(stack)     # must not raise on the tuple key
        self.assertEqual(node._detected_arch, "ltx")
        # the real lora dict is untouched — its key is still the TUPLE
        self.assertIn((_LTX_KEY, off), stack[0]["lora"])

    def test_file_item_detection_not_overridden_by_virtual_fallback(self):
        # Regression guard: a file-based item with recognizable trainer keys
        # resolves first (break), so the virtual LTX item never reaches the
        # fallback. Proves the fallback fires ONLY when file detection fails.
        file_item = {
            "name": "sdxl_style.safetensors",
            "lora": {"lora_te1_text_model_encoder_layers_0_self_attn_q_proj"
                     ".lora_up.weight": torch.zeros(4, 4)},
            "strength": 1.0,
        }
        virtual = self._virtual_stack(_LTX_KEY)[0]
        node = lora_optimizer.LoRAOptimizerInline()
        node._normalize_stack([file_item, virtual])
        self.assertEqual(node._detected_arch, "sdxl")   # file wins, not 'ltx'

    def test_sdxl_unet_virtual_keys_resolve_to_sd_unet_not_ltx(self):
        # No false LTX match: SDXL UNet cross-attn keys carry transformer_blocks
        # + attn1 but under input_blocks -> sd15 -> sd_unet preset.
        node = lora_optimizer.LoRAOptimizerInline()
        stack = self._virtual_stack(_SDXL_UNET_KEY)
        node._normalize_stack(stack)
        self.assertEqual(node._detected_arch, "sd15")
        self.assertNotEqual(node._detected_arch, "ltx")
        key, _ = lora_optimizer._resolve_arch_preset("auto", node._detected_arch)
        self.assertEqual(key, "sd_unet")

    def test_resolve_arch_preset_ltx_and_zimage_map_to_dit(self):
        # Prove the preset now resolves correctly once detection works.
        self.assertEqual(lora_optimizer._resolve_arch_preset("auto", "ltx")[0],
                         "dit")
        self.assertEqual(
            lora_optimizer._resolve_arch_preset("auto", "zimage")[0], "dit")


# Model-space captured key shared by attention-only Qwen-Image AND ACE-Step
# v1.0 chains: both surface as transformer_blocks.N.attn.to_q, so key-pattern
# detection cannot tell them apart (it falls through to the ACE-Step regex).
_QWEN_ACE_AMBIG_KEY = "diffusion_model.transformer_blocks.0.attn.to_q.weight"


class TestModelClassArchDetection(unittest.TestCase):
    """FIX #12: for CAPTURED inline chains, architecture is resolved from the
    comfy MODEL class (type(model.model).__name__), which is authoritative and
    disambiguates architectures that are indistinguishable from model-space
    keys alone. Fully guarded — never raises."""

    @staticmethod
    def _model_with(inner):
        m = _FakePatcher()
        m.model = inner
        return m

    def test_qwen_image_class_maps_to_qwen_image(self):
        class QwenImage:
            pass
        self.assertEqual(
            lora_optimizer._LoRAMergeBase._model_class_arch(
                self._model_with(QwenImage())), "qwen_image")

    def test_acestep_class_maps_to_acestep(self):
        class ACEStep:
            pass
        self.assertEqual(
            lora_optimizer._LoRAMergeBase._model_class_arch(
                self._model_with(ACEStep())), "acestep")

    def test_acestep15_class_maps_to_acestep(self):
        class ACEStep15:
            pass
        self.assertEqual(
            lora_optimizer._LoRAMergeBase._model_class_arch(
                self._model_with(ACEStep15())), "acestep")

    def test_ltxv_class_maps_to_ltx(self):
        class LTXV:
            pass
        self.assertEqual(
            lora_optimizer._LoRAMergeBase._model_class_arch(
                self._model_with(LTXV())), "ltx")

    def test_flux_class_maps_to_flux(self):
        class Flux:
            pass
        self.assertEqual(
            lora_optimizer._LoRAMergeBase._model_class_arch(
                self._model_with(Flux())), "flux")

    def test_subclass_resolves_via_mro(self):
        # comfy has many WAN subclasses (WAN22, WAN21_Vace, …); they must all
        # resolve to 'wan' via the base class in the MRO.
        class WAN21:
            pass
        class WAN22(WAN21):
            pass
        self.assertEqual(
            lora_optimizer._LoRAMergeBase._model_class_arch(
                self._model_with(WAN22())), "wan")

    def test_unknown_class_returns_none(self):
        class SomeUnsupportedModel:
            pass
        self.assertIsNone(
            lora_optimizer._LoRAMergeBase._model_class_arch(
                self._model_with(SomeUnsupportedModel())))

    def test_none_model_returns_none(self):
        self.assertIsNone(lora_optimizer._LoRAMergeBase._model_class_arch(None))

    def test_model_without_inner_model_returns_none(self):
        # _FakePatcher has no .model attribute -> None, no raise.
        self.assertIsNone(
            lora_optimizer._LoRAMergeBase._model_class_arch(_FakePatcher()))


class TestArchHintDisambiguation(unittest.TestCase):
    """FIX #12: attention-only Qwen-Image and ACE-Step v1.0 captured chains are
    INDISTINGUISHABLE from their model-space keys. The _arch_hint (derived from
    the model class) disambiguates them for the VIRTUAL path only; file-based
    stacks keep pure key-based detection."""

    def _virtual_stack(self, model_key):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {model_key: _adapter()})))
        return lora_optimizer.LoRAOptimizerInline._chain_groups_to_stack(
            mg, [], [_slot()], "simple")

    def test_ambiguous_key_misdetects_as_acestep_without_hint(self):
        # Documents the bug: keys alone resolve to ACE-Step (-> acestep_dit),
        # even when this is really an attention-only Qwen-Image chain.
        node = lora_optimizer.LoRAOptimizerInline()
        node._normalize_stack(self._virtual_stack(_QWEN_ACE_AMBIG_KEY))
        self.assertEqual(node._detected_arch, "acestep")

    def test_qwen_hint_overrides_ambiguous_keys(self):
        node = lora_optimizer.LoRAOptimizerInline()
        node._normalize_stack(self._virtual_stack(_QWEN_ACE_AMBIG_KEY),
                              _arch_hint="qwen_image")
        self.assertEqual(node._detected_arch, "qwen_image")
        # The whole point: qwen_image -> 'llm' preset, NOT 'acestep_dit'.
        self.assertEqual(
            lora_optimizer._resolve_arch_preset("auto", node._detected_arch)[0],
            "llm")

    def test_acestep_hint_keeps_acestep(self):
        node = lora_optimizer.LoRAOptimizerInline()
        node._normalize_stack(self._virtual_stack(_QWEN_ACE_AMBIG_KEY),
                              _arch_hint="acestep")
        self.assertEqual(node._detected_arch, "acestep")

    def test_unknown_hint_falls_back_to_key_based(self):
        # A None/"unknown" hint must never override key-based virtual detection.
        node = lora_optimizer.LoRAOptimizerInline()
        node._normalize_stack(self._virtual_stack(_LTX_KEY), _arch_hint="unknown")
        self.assertEqual(node._detected_arch, "ltx")

    def test_hint_not_consulted_for_file_based_stack(self):
        # Regression: a file-based (non-virtual) stack keeps pure key-based
        # detection even when a hint is present — the hint is captured-only.
        file_item = {
            "name": "sdxl_style.safetensors",
            "lora": {"lora_te1_text_model_encoder_layers_0_self_attn_q_proj"
                     ".lora_up.weight": torch.zeros(4, 4)},
            "strength": 1.0,
        }
        node = lora_optimizer.LoRAOptimizerInline()
        node._normalize_stack([file_item], _arch_hint="qwen_image")
        self.assertEqual(node._detected_arch, "sdxl")   # hint ignored for files

    def test_optimize_merge_resolves_llm_for_captured_qwen_via_model_class(self):
        # End-to-end: optimize_merge computes the arch hint from the MODEL class
        # and threads it into _normalize_stack, so an ambiguous captured Qwen
        # chain resolves to the 'llm' preset, NOT 'acestep_dit'. A spy on
        # _resolve_arch_preset captures the resolved preset then short-circuits
        # the (irrelevant) heavy merge.
        class QwenImage:
            pass
        model = _FakePatcher()
        model.model = QwenImage()
        node = lora_optimizer.LoRAOptimizerInline()
        stack = self._virtual_stack(_QWEN_ACE_AMBIG_KEY)

        class _StopMerge(Exception):
            pass
        captured = {}
        real = lora_optimizer._resolve_arch_preset

        def _spy(override, detected):
            captured["result"] = real(override, detected)
            raise _StopMerge()

        with mock.patch.object(lora_optimizer, "_resolve_arch_preset", _spy):
            with self.assertRaises(_StopMerge):
                node.optimize_merge(model, stack, 1.0)
        self.assertEqual(node._detected_arch, "qwen_image")
        self.assertEqual(captured["result"][0], "llm")
        self.assertNotEqual(captured["result"][0], "acestep_dit")


class TestInlineFingerprintArch(unittest.TestCase):
    """execute_inline must detect the architecture from the captured
    model-space keys and surface it in the fingerprint, instead of always
    warning 'architecture: unknown (inline capture)'."""

    def _model(self, patches):
        m = _FakePatcher(patches)
        m.model = types.SimpleNamespace()
        return m

    @staticmethod
    def _out(result):
        return result["result"] if isinstance(result, dict) else result

    def test_ltx_chain_reports_detected_arch(self):
        model = self._model(_chain_patches((0.8, {_LTX_KEY: _adapter()})))
        node = lora_optimizer.LoRAOptimizerInline()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "engine report",
                                                     None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertIn("architecture: ltx", report)
        self.assertIn("detected from captured keys", report)
        self.assertNotIn("architecture: unknown (inline capture)", report)

    def test_unknown_chain_still_warns(self):
        # Regression net: an unrecognizable key with no pinned preset still
        # emits the "set a preset" guidance.
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = lora_optimizer.LoRAOptimizerInline()
        node.optimize_merge = lambda m, s, o, **kw: (m, None, "r", None, None)
        report = self._out(node.execute_inline(model, output_strength=1.0))[2]
        self.assertIn("architecture: unknown (inline capture)", report)
        self.assertIn("architecture_preset", report)


class TestInlineRankReporting(unittest.TestCase):
    """The inline analysis path must read each captured adapter's TRUE rank
    (mat_down.shape[0]) instead of counting every precomputed diff as rank 1.
    A rank-1 undercount makes avg_rank -> sum_rank -> compress_rank floor to
    64, over-compressing higher-rank captured LoRAs."""

    KEY = "layer.weight"

    # ---- unit: the _payload_rank helper, now on the shared base class ----

    def test_payload_rank_reads_adapter_rank(self):
        got = lora_optimizer._LoRAMergeBase._payload_rank(
            _adapter(rank=32, out_dim=16, in_dim=16))
        self.assertEqual(got, 32)

    def test_payload_rank_none_for_dense_diff_tuple(self):
        self.assertIsNone(lora_optimizer._LoRAMergeBase._payload_rank(
            ("diff", (torch.randn(8, 8),))))

    def test_payload_rank_none_for_bare_tensor(self):
        self.assertIsNone(
            lora_optimizer._LoRAMergeBase._payload_rank(torch.randn(8, 8)))

    def test_payload_rank_none_for_lokr(self):
        lokr = lora_optimizer.LoKrAdapter(
            loaded_keys=set(),
            weights=(torch.randn(4, 4), torch.randn(2, 2), None, None,
                     None, None, None, None))
        self.assertIsNone(lora_optimizer._LoRAMergeBase._payload_rank(lokr))

    def test_helper_not_left_on_inline_subclass(self):
        # Moved to the base, so the subclass must NOT redeclare it (DRY: one
        # implementation shared by the fingerprint AND _prepare_group_diffs).
        self.assertNotIn("_payload_rank",
                         vars(lora_optimizer.LoRAOptimizerInline))
        self.assertIn("_payload_rank",
                      vars(lora_optimizer._LoRAMergeBase))

    # ---- _prepare_group_diffs rank accounting on virtual items ----

    def _rank_sums(self, payloads, dim=16):
        """Run _prepare_group_diffs over a virtual (_precomputed_diffs) stack
        of the given per-LoRA payloads on one shared model key; return the
        resolved rank_sums dict."""
        model = _pipeline_model({}, {}, dim=dim)     # .model has layer.weight
        active = [{"name": f"chain lora #{i + 1}", "strength": 1.0,
                   "_precomputed_diffs": True, "lora": {self.KEY: p}}
                  for i, p in enumerate(payloads)]
        target_group = {"target_key": self.KEY, "is_clip": False,
                        "aliases": [self.KEY], "label_prefix": self.KEY}
        node = lora_optimizer.LoRAOptimizerInline()
        prepared = node._prepare_group_diffs(
            target_group, active, model, None, torch.device("cpu"))
        return prepared["rank_sums"]

    def test_prepare_group_diffs_reads_true_adapter_rank(self):
        rank_sums = self._rank_sums([_adapter(rank=32, out_dim=16, in_dim=16)])
        self.assertEqual(rank_sums[0], 32)       # was 1 before the fix

    def test_prepare_group_diffs_dense_diff_stays_rank_1(self):
        # A formula-submerge-style dense virtual payload has no meaningful rank
        # -> _payload_rank is None -> fall back to += 1 (unchanged behavior).
        rank_sums = self._rank_sums([("diff", (torch.randn(16, 16),))])
        self.assertEqual(rank_sums[0], 1)

    def test_prepare_group_diffs_bare_tensor_stays_rank_1(self):
        rank_sums = self._rank_sums([torch.randn(16, 16)])
        self.assertEqual(rank_sums[0], 1)

    # ---- end-to-end: avg_rank -> sum_rank -> compress_rank ----

    def _run_inline(self, chain, settings):
        applied = {}
        model = _pipeline_model(_chain_patches(*chain), applied)
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": self.KEY}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            result = node.execute_inline(model, output_strength=1.0,
                                         settings=settings)
        return result["result"] if isinstance(result, dict) else result

    def test_report_shows_true_avg_rank_not_one(self):
        chain = (
            (1.0, {self.KEY: _adapter(rank=32, out_dim=16, in_dim=16)}),
            (0.7, {self.KEY: _adapter(rank=32, out_dim=16, in_dim=16)}),
        )
        settings = _advanced_settings(optimization_mode="global",
                                      merge_strategy_override="weighted_sum")
        report = self._run_inline(chain, settings)[2]
        self.assertIn("Avg rank: 32", report)
        self.assertNotIn("Avg rank: 1\n", report)   # the old always-1 bug

    def test_compress_rank_not_floored_for_high_rank_chain(self):
        # The user's actual regression: rank-64 + rank-128 captured LoRAs.
        # True sum_rank = 64 + 128 = 192 -> compress_rank = max(192, 64) = 192.
        # The bug reported avg_rank 1 each -> sum_rank 2 -> floored to 64,
        # over-compressing the rank-128 LoRA.
        chain = (
            (1.0, {self.KEY: _adapter(rank=64, out_dim=16, in_dim=16)}),
            (0.7, {self.KEY: _adapter(rank=128, out_dim=16, in_dim=16)}),
        )
        settings = _advanced_settings(optimization_mode="global",
                                      merge_strategy_override="weighted_sum",
                                      patch_compression="smart")
        lora_data = self._run_inline(chain, settings)[4]
        self.assertEqual(lora_data["sum_rank"], 192)   # NOT floored to 64


def _autotuner_settings(**overrides):
    """Complete OPTIMIZER_SETTINGS autotuner-mode dict, mirroring what the
    AutoTuner Settings node (build_settings) emits — with cheap, deterministic
    knobs. execute_inline reads the bracket-keyed fields to build the auto_tune
    call, so a partial dict would KeyError before auto_tune is even reached."""
    settings = {
        "mode": "autotuner",
        "top_n": 1,
        "scoring_svd": "disabled",
        "scoring_device": "cpu",
        "scoring_speed": "full",
        "scoring_formula": "v2",
        "output_mode": "merge",
        "smooth_slerp_gate": False,
        "normalize_keys": "enabled",     # must be overridden to disabled
        "architecture_preset": "auto",
        "auto_strength_floor": -1.0,
        "decision_smoothing": 0.25,
        "vram_budget": 0.0,
        "cache_patches": "disabled",
        "star_eta": 100.0,
        "tame_layers": 0.0,
        "tame_threshold": 0.3,
        "diff_cache_mode": "disabled",
        "diff_cache_ram_pct": 0.5,
        "community_cache": "disabled",
        "evaluator": None,
        "memory_mode": "disabled",
        "selection": 1,
        "record_dataset": "disabled",
    }
    settings.update(overrides)
    return settings


class TestInlineAutoTunerDelegation(unittest.TestCase):
    """Commit B: mode='autotuner' settings now delegate to the real AutoTuner
    on the inline-prepared STRIPPED model/clip + VIRTUAL stack, forcing
    normalize_keys off, instead of falling back to optimizer defaults."""

    def _node(self):
        return lora_optimizer.LoRAOptimizerInline()

    def _model(self, patches):
        m = _FakePatcher(patches)
        m.model = types.SimpleNamespace()
        return m

    @staticmethod
    def _out(result):
        return result["result"] if isinstance(result, dict) else result

    def _stub_delegate(self, node, seen):
        """Pre-seed node._autotuner_delegate with a recorder so execute_inline
        reuses it instead of building a real LoRAAutoTuner."""
        def fake_auto_tune(model, stack, output_strength, **kw):
            seen["model"] = model
            seen["stack"] = stack
            seen["output_strength"] = output_strength
            seen.update(kw)
            return (model, kw.get("clip"), "tuner report",
                    "analysis report", {"td": 1}, {"ld": 2})
        node._autotuner_delegate = types.SimpleNamespace(auto_tune=fake_auto_tune)

    def test_autotune_threads_model_class_arch_hint(self):
        """auto_tune must pass the model-class arch hint to _normalize_stack so
        captured chains score candidates under the right preset (was hintless →
        attention-only Qwen scored under the ACE-Step preset)."""
        class QwenImage:  # mirrors comfy.model_base.QwenImage
            pass
        model = _FakePatcher({})
        model.model = QwenImage()
        delegate = lora_optimizer.LoRAAutoTuner()
        seen = {}

        class _Stop(Exception):
            pass

        def rec_normalize(stack, normalize_keys="disabled", _arch_hint=None):
            seen["hint"] = _arch_hint
            raise _Stop

        delegate._normalize_stack = rec_normalize
        stack = [{"name": "chain lora #1 [x]", "lora": {"k": _adapter()},
                  "strength": 1.0, "_precomputed_diffs": True}]
        with self.assertRaises(_Stop):
            delegate.auto_tune(model, stack, 1.0)
        self.assertEqual(seen.get("hint"), "qwen_image")

    def test_delegates_with_virtual_stack_and_stripped_model(self):
        model = self._model(_chain_patches(
            (0.8, {"a": _adapter()}), (0.5, {"a": _adapter()})))
        node = self._node()
        node.optimize_merge = lambda *a, **k: self.fail(
            "optimize_merge must not run in autotuner mode")
        seen = {}
        self._stub_delegate(node, seen)
        node.execute_inline(model, output_strength=1.0,
                            settings=_autotuner_settings())
        # VIRTUAL stack (precomputed diffs) ...
        self.assertEqual(len(seen["stack"]), 2)
        self.assertTrue(all(i["_precomputed_diffs"] for i in seen["stack"]))
        # ... STRIPPED model (patches removed, not the raw input) ...
        self.assertEqual(seen["model"].patches, {})
        self.assertIsNot(seen["model"], model)
        # ... normalize_keys forced off despite settings "enabled"
        self.assertEqual(seen["normalize_keys"], "disabled")
        self.assertAlmostEqual(seen["output_strength"], 1.0)
        self.assertTrue(
            node._autotuner_delegate._force_explicit_clip_multiplier)

    def test_return_is_inline_5_tuple_with_fingerprint_prepended(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter(), "b": _adapter()})))
        node = self._node()
        seen = {}
        self._stub_delegate(node, seen)
        out = self._out(node.execute_inline(model, output_strength=1.0,
                                            settings=_autotuner_settings()))
        self.assertEqual(len(out), 5)      # inline 5-tuple, not auto_tune's 6
        report = out[2]
        self.assertIn("Detected loader chain", report)          # fingerprint
        self.assertLess(report.index("Detected loader chain"),
                        report.index("tuner report"))            # prepended
        self.assertIn("ANALYSIS REPORT", report)                # analysis folded in
        self.assertIn("analysis report", report)
        self.assertEqual(out[3], {"td": 1})                     # tuner_data
        self.assertEqual(out[4], {"ld": 2})                     # lora_data

    def test_forwards_settings_fields_to_auto_tune(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._stub_delegate(node, seen)
        node.execute_inline(model, output_strength=1.0,
                            settings=_autotuner_settings(
                                top_n=3, memory_mode="auto",
                                experimental_options={"np_lora": True, "ct_merge": False},
                                community_cache="upload_and_download",
                                architecture_preset="dit"))
        self.assertEqual(seen["top_n"], 3)
        self.assertEqual(seen["memory_mode"], "auto")
        self.assertEqual(seen["community_cache"], "upload_and_download")
        self.assertEqual(seen["architecture_preset"], "dit")
        self.assertEqual(seen["experimental_options"], {"np_lora": True, "ct_merge": False})

    def test_cache_patches_pinned_disabled_for_delegate(self):
        # cache_patches is NOT forwarded from settings — the delegate's in-node
        # cache can never usefully hit for inline and reintroduces id()-reuse
        # staleness, so it is pinned off regardless of the settings value.
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._stub_delegate(node, seen)
        node.execute_inline(model, output_strength=1.0,
                            settings=_autotuner_settings(cache_patches="enabled"))
        self.assertEqual(seen["cache_patches"], "disabled")

    def test_passes_stripped_clip_original_untouched(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        clip = _FakeCLIP(_FakePatcher(_chain_patches((0.6, {"te.a": _adapter()}))))
        node = self._node()
        seen = {}
        self._stub_delegate(node, seen)
        node.execute_inline(model, output_strength=1.0, clip=clip,
                            settings=_autotuner_settings())
        self.assertIsNot(seen["clip"], clip)                 # stripped clone
        self.assertEqual(seen["clip"].patcher.patches, {})
        self.assertEqual(len(clip.patcher.patches["te.a"]), 1)  # original intact

    def test_old_not_supported_message_gone(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._stub_delegate(node, seen)
        report = self._out(node.execute_inline(
            model, output_strength=1.0, settings=_autotuner_settings()))[2]
        self.assertNotIn("AutoTuner settings are not supported inline", report)
        self.assertNotIn("using optimizer defaults", report)

    def test_pinned_preset_suppresses_arch_unknown_warning(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        self._stub_delegate(node, seen)
        report = self._out(node.execute_inline(
            model, output_strength=1.0,
            settings=_autotuner_settings(architecture_preset="dit")))[2]
        self.assertNotIn("architecture: unknown", report)

    def test_delegate_created_lazily_and_reused(self):
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        self.assertFalse(hasattr(node, "_autotuner_delegate"))
        with mock.patch.object(lora_optimizer.LoRAAutoTuner, "auto_tune",
                               return_value=(model, None, "r", "", None, None)) as m:
            node.execute_inline(model, output_strength=1.0,
                                settings=_autotuner_settings())
            delegate = node._autotuner_delegate
            self.assertIsInstance(delegate, lora_optimizer.LoRAAutoTuner)
            node.execute_inline(model, output_strength=1.0,
                                settings=_autotuner_settings())
        self.assertIs(node._autotuner_delegate, delegate)   # reused, not recreated
        self.assertEqual(m.call_count, 2)

    def test_advanced_mode_still_uses_optimize_merge(self):
        # Regression: advanced mode is untouched — it must still route through
        # optimize_merge (not the AutoTuner) with normalize_keys pinned off.
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        def fake_merge(m, stack, output_strength, **kw):
            seen.update(kw)
            seen["stack"] = stack
            return (m, kw.get("clip"), "engine report", None, None)
        node.optimize_merge = fake_merge
        node._autotuner_delegate = types.SimpleNamespace(
            auto_tune=lambda *a, **k: self.fail("auto_tune called in advanced mode"))
        node.execute_inline(model, output_strength=1.0, settings=_advanced_settings())
        self.assertEqual(seen["normalize_keys"], "disabled")
        self.assertIn("stack", seen)

    def test_no_settings_path_still_uses_optimize_merge(self):
        # Regression: no settings -> plain optimize_merge, no AutoTuner.
        model = self._model(_chain_patches((0.8, {"a": _adapter()})))
        node = self._node()
        seen = {}
        def fake_merge(m, stack, output_strength, **kw):
            seen.update(kw)
            return (m, kw.get("clip"), "engine report", None, None)
        node.optimize_merge = fake_merge
        node._autotuner_delegate = types.SimpleNamespace(
            auto_tune=lambda *a, **k: self.fail("auto_tune called with no settings"))
        node.execute_inline(model, output_strength=1.0)
        self.assertEqual(seen["normalize_keys"], "disabled")


class TestInlineAutoTunerEndToEnd(unittest.TestCase):
    """The REAL AutoTuner runs a tiny search over a 2-captured-LoRA chain and
    returns a merged, re-applied model — proving auto_tune tolerates virtual
    (_precomputed_diffs) items end to end, with no file reload."""

    KEY = "layer.weight"

    def test_both_experimental_modes_apply_on_captured_chains(self):
        for mode in ("np_lora", "ct_merge"):
            with self.subTest(mode=mode):
                chain = ((1., {self.KEY: _adapter(out_dim=16, in_dim=16)}),
                         (.7, {self.KEY: _adapter(out_dim=16, in_dim=16)}))
                applied = {}
                model = _pipeline_model(_chain_patches(*chain), applied)
                original = {k: list(v) for k, v in model.patches.items()}
                node = lora_optimizer.LoRAOptimizerInline()
                node._get_model_keys = lambda model: {"alias_layer": self.KEY}
                node._autotuner_delegate = lora_optimizer.LoRAAutoTuner()
                node._autotuner_delegate._get_model_keys = lambda model: {"alias_layer": self.KEY}
                def choose(_e, _m, _c, _ld, config, _summary):
                    return {"score": 1. if config["merge_mode"] == mode else 0.}
                with mock.patch.object(lora_optimizer, "_run_autotuner_evaluator", side_effect=choose):
                    result = node.execute_inline(model, 1., settings=_autotuner_settings(
                        architecture_preset="dit", experimental_options={"np_lora": True, "ct_merge": True},
                        evaluator={"combine_mode": "external_only"}))
                out = result["result"] if isinstance(result, dict) else result
                self.assertEqual(out[4]["merge_metadata"]["mode"], mode)
                self.assertIn(self.KEY, applied["patches"])
                self.assertTrue(torch.isfinite(lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
                    applied["patches"][self.KEY])).all())
                self.assertEqual(model.patches[self.KEY], original[self.KEY])

    def test_two_lora_chain_autotunes_and_reapplies(self):
        up_a = torch.linspace(-1.0, 1.0, 16 * 4).reshape(16, 4)
        down_a = torch.linspace(0.5, -0.5, 4 * 16).reshape(4, 16)
        up_b = torch.linspace(-0.8, 1.2, 16 * 4).reshape(16, 4)
        down_b = torch.linspace(0.3, -0.7, 4 * 16).reshape(4, 16)
        chain = (
            (1.0, {self.KEY: _adapter(up=up_a, down=down_a)}),
            (0.7, {self.KEY: _adapter(up=up_b, down=down_b)}),
        )
        applied = {}
        model = _pipeline_model(_chain_patches(*chain), applied)
        orig_entries = {k: list(v) for k, v in model.patches.items()}
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": self.KEY}
        # The merge runs on the delegate (a separate LoRAAutoTuner), so the
        # key-alias override the fake model needs must live there too. In
        # production both share the real _get_model_keys against a real model.
        node._autotuner_delegate = lora_optimizer.LoRAAutoTuner()
        node._autotuner_delegate._get_model_keys = lambda m: {"alias_layer": self.KEY}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            result = node.execute_inline(
                model, output_strength=1.0,
                settings=_autotuner_settings(architecture_preset="dit"))
        out = result["result"] if isinstance(result, dict) else result
        # a real AutoTuner ran the search
        self.assertIsInstance(node._autotuner_delegate,
                              lora_optimizer.LoRAAutoTuner)
        self.assertEqual(calls, [])                   # no single-LoRA fast-path
        # merged patch re-applied on a clone for the shared key
        self.assertIn("patches", applied)
        self.assertEqual(set(applied["patches"]), {self.KEY})
        got = lora_optimizer._LoRAMergeBase._expand_patch_to_diff(
            applied["patches"][self.KEY])
        self.assertEqual(tuple(got.shape), (16, 16))
        self.assertTrue(torch.isfinite(got).all())
        self.assertGreater(got.abs().max().item(), 0.0)
        # inline 5-tuple with the fingerprint prepended
        self.assertEqual(len(out), 5)
        self.assertIn("Detected loader chain", out[2])
        # input model untouched (pre-run snapshot)
        self.assertEqual(model.patches[self.KEY], orig_entries[self.KEY])


class TestNormalizedVirtualIdentity(unittest.TestCase):
    def test_resolved_filename_and_memoized_content_hash_survive_normalization(self):
        # Inline stamps are resolved before optimize_merge normalizes the
        # virtual stack.  Dropping either identity field there disconnects the
        # inline path from file-based memory/community cache entries.
        item = _virtual_item({"layer.weight": _adapter()}, name="captured")
        item["_resolved_file_name"] = "styles/example.safetensors"
        item["_content_hash"] = "0123456789abcdef"

        normalized = lora_optimizer.LoRAOptimizer()._normalize_stack(
            [item], normalize_keys="disabled")

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["_resolved_file_name"],
                         "styles/example.safetensors")
        self.assertEqual(normalized[0]["_content_hash"],
                         "0123456789abcdef")


class TestCapturedContentHash(unittest.TestCase):
    """Commit A: _lora_content_hash must produce a STABLE 16-hex content hash
    for captured chain items (adapter objects, tuple keys, diff tuples),
    keyed on the factor VALUES not object identity, so inline chains get a
    persistent memory/community identity across sessions and machines. The
    file-based path and plain-tensor path stay byte-identical."""

    def _hash(self, item):
        # Captured items have no file on disk -> force the in-memory fallback.
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            return lora_optimizer.LoRAAutoTuner._lora_content_hash(item)

    def test_captured_adapter_item_hashes_to_stable_16hex(self):
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        h = self._hash({"name": "chain lora #1 [x]",
                        "lora": {"k": _adapter(up=up, down=down)}})
        self.assertIsNotNone(h)
        self.assertEqual(len(h), 16)
        int(h, 16)   # valid hex

    def test_hash_identical_for_fresh_adapters_with_same_values(self):
        # Two DIFFERENT adapter OBJECTS holding the SAME tensor values must
        # hash identically -> content-based, not identity-based. Pre-fix this
        # failed: adapters have no .detach, so repr(v) (memory address) was
        # hashed and differed per object.
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        a1 = _adapter(up=up.clone(), down=down.clone())
        a2 = _adapter(up=up.clone(), down=down.clone())
        self.assertIsNot(a1, a2)
        h1 = self._hash({"name": "chain lora #1 [aaa]", "lora": {"k": a1}})
        h2 = self._hash({"name": "chain lora #1 [bbb]", "lora": {"k": a2}})
        self.assertEqual(h1, h2)

    def test_hash_differs_when_a_weight_value_changes(self):
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        h1 = self._hash({"name": "n",
                         "lora": {"k": _adapter(up=up.clone(), down=down.clone())}})
        down2 = down.clone()
        down2[0, 0] += 1.0
        h2 = self._hash({"name": "n",
                         "lora": {"k": _adapter(up=up.clone(), down=down2)}})
        self.assertNotEqual(h1, h2)

    def test_hash_stable_when_alpha_changes_is_reflected(self):
        # alpha is a non-tensor weight element (repr-hashed) -> a different
        # alpha must change the hash.
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        h1 = self._hash({"name": "n",
                         "lora": {"k": _adapter(up=up.clone(), down=down.clone(),
                                                alpha=4.0)}})
        h2 = self._hash({"name": "n",
                         "lora": {"k": _adapter(up=up.clone(), down=down.clone(),
                                                alpha=2.0)}})
        self.assertNotEqual(h1, h2)

    def test_tuple_keys_hash_without_crashing_and_stably(self):
        # Fused-QKV captures key the virtual dict by (str_key, offset) TUPLES.
        # Pre-fix k.encode() crashed on a tuple; str(k) must be used.
        off = (0, 0, 4)
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        item1 = {"name": "n",
                 "lora": {("k", off): _adapter(up=up.clone(), down=down.clone())}}
        item2 = {"name": "n",
                 "lora": {("k", off): _adapter(up=up.clone(), down=down.clone())}}
        h1 = self._hash(item1)          # must not raise
        self.assertEqual(len(h1), 16)
        self.assertEqual(h1, self._hash(item2))

    def test_diff_tuple_payload_hashes_stably_and_value_sensitively(self):
        t = torch.randn(4, 4)
        item1 = {"name": "n", "lora": {"k": ("diff", (t.clone(),))}}
        item2 = {"name": "n", "lora": {"k": ("diff", (t.clone(),))}}
        h1 = self._hash(item1)
        self.assertEqual(len(h1), 16)
        self.assertEqual(h1, self._hash(item2))
        t2 = t.clone()
        t2[0, 0] += 1.0
        self.assertNotEqual(h1, self._hash({"name": "n",
                                            "lora": {"k": ("diff", (t2,))}}))

    def test_plain_tensor_dict_still_hashes_and_is_stable(self):
        # Regression: LoRAExtractFromModel-style plain-tensor dicts still hash.
        sd = {"blk.lora_up.weight": torch.ones(4, 2),
              "blk.lora_down.weight": torch.ones(2, 4) * 0.5}
        h1 = self._hash({"name": "<x>", "lora": sd})
        self.assertIsNotNone(h1)
        self.assertEqual(len(h1), 16)
        self.assertEqual(h1, self._hash({"name": "<x>", "lora": dict(sd)}))

    def test_plain_tensor_hash_byte_identical_to_prefix_reference(self):
        # The plain-tensor code path must be UNCHANGED (existing
        # LoRAExtractFromModel community entries must not shift). Reproduce the
        # exact pre-fix per-key hashing and assert equality.
        import hashlib
        sd = {"b.up": torch.ones(4, 2), "b.down": torch.ones(2, 4) * 0.5}
        h = hashlib.sha256()
        for k in sorted(sd.keys()):
            v = sd[k]
            h.update(k.encode())
            t = v.detach().to("cpu", torch.float32).contiguous()
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
        expected = h.hexdigest()[:16]
        self.assertEqual(self._hash({"name": "<x>", "lora": sd}), expected)

    def test_two_different_captured_loras_differ(self):
        a = {"name": "n", "lora": {"k": _adapter(up=torch.ones(8, 4),
                                                 down=torch.ones(4, 8))}}
        b = {"name": "n", "lora": {"k": _adapter(up=torch.zeros(8, 4),
                                                 down=torch.ones(4, 8))}}
        self.assertNotEqual(self._hash(a), self._hash(b))


class TestCapturedPersistentIdentity(unittest.TestCase):
    """Critical #1: the persistent memory + analysis-cache keys must derive
    from captured CONTENT (session-stable), not the per-session salted item
    name — otherwise inline memory saves under a fresh key every ComfyUI
    session and never reads back. File-based items are unchanged."""

    KEY = "layer.weight"

    def _captured_item(self, salt, up, down):
        # One session's virtual stack item: identical content, session salt.
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"k": _adapter(up=up.clone(), down=down.clone())})))
        return lora_optimizer.LoRAOptimizerInline._chain_groups_to_stack(
            mg, [], [_slot()], "simple", name_salt=f" [{salt}]")[0]

    def test_persistent_key_stable_across_sessions_for_captured(self):
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            s1 = self._captured_item("aaaaaaaa", up, down)
            s2 = self._captured_item("bbbbbbbb", up, down)
            self.assertNotEqual(s1["name"], s2["name"])   # salted names differ
            k1 = lora_optimizer.LoRAAutoTuner._persistent_lora_key(s1)
            k2 = lora_optimizer.LoRAAutoTuner._persistent_lora_key(s2)
            self.assertTrue(k1.startswith("captured:"))
            self.assertEqual(k1, k2)      # content identity is session-stable

    def test_persistent_key_is_name_for_file_items(self):
        # Regression: file-based items (no _precomputed_diffs) key on name.
        item = {"name": "style.safetensors", "strength": 1.0, "lora": {}}
        self.assertEqual(
            lora_optimizer.LoRAAutoTuner._persistent_lora_key(item),
            "style.safetensors")

    def test_identity_hash_stable_across_sessions_for_captured(self):
        # The per-LoRA/pair analysis DISK cache key (_lora_identity_hash) must
        # also be session-stable for captured items.
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            h1 = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                self._captured_item("aaaaaaaa", up, down))
            h2 = lora_optimizer.LoRAAutoTuner._lora_identity_hash(
                self._captured_item("bbbbbbbb", up, down))
            self.assertEqual(len(h1), 16)
            self.assertEqual(h1, h2)

    def test_identity_hash_unchanged_for_file_items(self):
        # Regression: file item identity hash == the pre-fix name+0+0 formula.
        import hashlib, json
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            item = {"name": "x.safetensors", "strength": 1.0, "lora": {}}
            got = lora_optimizer.LoRAAutoTuner._lora_identity_hash(item)
        expected = hashlib.sha256(json.dumps(
            ("x.safetensors", 0, 0), separators=(",", ":")).encode()).hexdigest()[:16]
        self.assertEqual(got, expected)

    def test_names_only_hash_stable_across_sessions_for_captured(self):
        # The whole-stack analysis cache key must be session-stable too.
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            s1 = [self._captured_item("aaaaaaaa", up, down)]
            s2 = [self._captured_item("bbbbbbbb", up, down)]
            h1, _ = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(s1)
            h2, _ = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(s2)
            self.assertEqual(h1, h2)

    def test_per_lora_merge_signature_stable_across_sessions_for_captured(self):
        # Feeds the memory settings_hash (part of the memory file path) — must
        # not vary by salt or cross-session hits break.
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            sig1 = lora_optimizer.LoRAAutoTuner._per_lora_merge_signature(
                [self._captured_item("aaaaaaaa", up, down)])
            sig2 = lora_optimizer.LoRAAutoTuner._per_lora_merge_signature(
                [self._captured_item("bbbbbbbb", up, down)])
            self.assertEqual(sig1, sig2)

    def test_memo_content_hash_computed_once(self):
        # Memoized on the item dict so re-hashing the (expensive) captured
        # factors doesn't happen per candidate / per memory+community lookup.
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        item = self._captured_item("aaaaaaaa", up, down)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            with mock.patch.object(
                    lora_optimizer.LoRAAutoTuner, "_lora_content_hash",
                    wraps=lora_optimizer.LoRAAutoTuner._lora_content_hash) as spy:
                a = lora_optimizer.LoRAAutoTuner._persistent_lora_key(item)
                b = lora_optimizer.LoRAAutoTuner._memo_content_hash(item)
                c = lora_optimizer.LoRAAutoTuner._persistent_lora_key(item)
        self.assertEqual(a, c)
        self.assertEqual(a, f"captured:{b}")
        self.assertEqual(spy.call_count, 1)   # computed exactly once


class TestInlineMemoryCrossSession(unittest.TestCase):
    """The headline: a memory entry saved in 'session 1' (one salt) is found
    in 'session 2' (different salt, identical captured content) — driving the
    REAL auto_tune through execute_inline. Fails on name-based keys (no hit),
    passes on content-based keys."""

    KEY = "layer.weight"

    @staticmethod
    def _chain():
        up_a = torch.linspace(-1.0, 1.0, 16 * 4).reshape(16, 4)
        down_a = torch.linspace(0.5, -0.5, 4 * 16).reshape(4, 16)
        up_b = torch.linspace(-0.8, 1.2, 16 * 4).reshape(16, 4)
        down_b = torch.linspace(0.3, -0.7, 4 * 16).reshape(4, 16)
        return (
            (1.0, {TestInlineMemoryCrossSession.KEY: _adapter(up=up_a, down=down_a)}),
            (0.7, {TestInlineMemoryCrossSession.KEY: _adapter(up=up_b, down=down_b)}),
        )

    def _run_session(self, salt_uuid):
        applied = {}
        model = _pipeline_model(_chain_patches(*self._chain()), applied)
        model.patches_uuid = salt_uuid          # drives the per-session name salt
        node = lora_optimizer.LoRAOptimizerInline()
        node._get_model_keys = lambda m: {"alias_layer": self.KEY}
        node._autotuner_delegate = lora_optimizer.LoRAAutoTuner()
        node._autotuner_delegate._get_model_keys = lambda m: {"alias_layer": self.KEY}
        calls = []
        with mock.patch.object(lora_optimizer.comfy.sd, "load_lora_for_models",
                               _realistic_load_lora_for_models(calls)):
            result = node.execute_inline(
                model, output_strength=1.0,
                settings=_autotuner_settings(architecture_preset="dit",
                                             memory_mode="auto", top_n=1))
        return result["result"] if isinstance(result, dict) else result

    def test_memory_saved_in_session1_is_found_in_session2(self):
        import tempfile
        with tempfile.TemporaryDirectory() as memdir:
            # captured content has no file -> force in-memory content hashing
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", memdir), \
                 mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                                   return_value=None):
                out1 = self._run_session(uuid.uuid4())
                self.assertNotIn("MEMORY HIT", out1[2])     # session 1: full sweep
                out2 = self._run_session(uuid.uuid4())      # different salt
                self.assertIn("MEMORY HIT", out2[2])        # session 2: cross-session hit
                self.assertEqual(len(out2), 5)


class TestInlineAnalysisCacheCrossSession(unittest.TestCase):
    """FIX #14: the whole-stack analysis cache is KEYED (names_only_hash) on
    captured content, so the cache FILE is found cross-session — but the stored
    source_loras and the _remap_analysis_indices name comparison used the raw,
    per-session salted name, so validation mismatched -> miss -> Pass 1
    recomputed. Routing both through _persistent_lora_key (session-stable) makes
    the analysis cache HIT cross-session for inline captured chains. File-based
    items are unchanged (_persistent_lora_key returns their name)."""

    def _captured_item(self, salt, up, down):
        mg = lora_optimizer._LoRAMergeBase._reconstruct_chain_groups(
            _chain_patches((0.8, {"k": _adapter(up=up.clone(), down=down.clone())})))
        return lora_optimizer.LoRAOptimizerInline._chain_groups_to_stack(
            mg, [], [_slot()], "simple", name_salt=f" [{salt}]")[0]

    # ---- _cache_source_loras: what gets STORED in the analysis cache ----

    def test_cache_source_loras_persistent_for_captured(self):
        # Two sessions, same captured content, different salted names -> the
        # stored source-LoRA keys are IDENTICAL (session-stable), not salted.
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            s1 = self._captured_item("aaaaaaaa", up, down)
            s2 = self._captured_item("bbbbbbbb", up, down)
            self.assertNotEqual(s1["name"], s2["name"])       # salted names differ
            src1 = lora_optimizer.LoRAAutoTuner._cache_source_loras([s1])
            src2 = lora_optimizer.LoRAAutoTuner._cache_source_loras([s2])
            self.assertTrue(src1[0]["name"].startswith("captured:"))
            self.assertEqual(src1, src2)                      # session-stable

    def test_cache_source_loras_is_name_for_file(self):
        # Regression: file items store their name, byte-identical to the old
        # [{"name": item["name"]}] formula.
        item = {"name": "style.safetensors", "strength": 1.0, "lora": {}}
        self.assertEqual(
            lora_optimizer.LoRAAutoTuner._cache_source_loras([item]),
            [{"name": "style.safetensors"}])

    # ---- _remap_analysis_indices: how the cache is VALIDATED on load ----

    def test_remap_matches_captured_across_salt(self):
        # Session 1 stored persistent-keyed source_loras; session 2 loads with a
        # differently-salted captured item of the SAME content. Pre-fix the
        # remap compared raw l["name"] (salted, mismatched) -> None (miss);
        # post-fix it compares persistent keys -> match -> returns per_prefix.
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        per_prefix = {"blk": {"per_lora_norm_sq": {"0": 1.0}}}
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            s1 = self._captured_item("aaaaaaaa", up, down)
            s2 = self._captured_item("bbbbbbbb", up, down)
            # session-1's stored source_loras (persistent-keyed); built here
            # without the helper so this pins the remap change in isolation.
            cached_src = [{"name":
                           lora_optimizer.LoRAAutoTuner._persistent_lora_key(s1)}]
            out = lora_optimizer.LoRAAutoTuner._remap_analysis_indices(
                per_prefix, cached_src, [s2])
        self.assertEqual(out, per_prefix)                     # HIT, not None

    def test_remap_file_items_unchanged(self):
        # Regression: file items still validate by name (same order -> as-is).
        per_prefix = {"blk": {"per_lora_norm_sq": {"0": 1.0}}}
        cached_src = [{"name": "a.safetensors"}]
        active = [{"name": "a.safetensors", "strength": 1.0, "lora": {}}]
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            out = lora_optimizer.LoRAAutoTuner._remap_analysis_indices(
                per_prefix, cached_src, active)
        self.assertEqual(out, per_prefix)

    # ---- integration: real save -> load round-trip HITS cross-session ----

    def test_analysis_cache_hits_cross_session_for_captured(self):
        import tempfile
        up, down = torch.randn(8, 4), torch.randn(4, 8)
        per_prefix = {"blk": {"per_lora_norm_sq": {"0": 1.0}}}
        with tempfile.TemporaryDirectory() as memdir:
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", memdir), \
                 mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                                   return_value=None):
                s1 = self._captured_item("aaaaaaaa", up, down)
                s2 = self._captured_item("bbbbbbbb", up, down)   # same content
                h1, _ = lora_optimizer.LoRAAutoTuner._compute_names_only_hash([s1])
                src = lora_optimizer.LoRAAutoTuner._cache_source_loras([s1])
                lora_optimizer.LoRAAutoTuner._analysis_cache_save(
                    h1, per_prefix, src)
                # Session 2: content-based cache FILE key already matches...
                h2, _ = lora_optimizer.LoRAAutoTuner._compute_names_only_hash([s2])
                self.assertEqual(h1, h2)
                # ...and now the stored/validated identities match too -> HIT.
                got = lora_optimizer.LoRAAutoTuner._analysis_cache_load(
                    h2, active_loras=[s2])
                self.assertIsNotNone(got)
                self.assertIn("blk", got)


class _AttachHost:
    """Minimal object exposing the ModelPatcher attachment API."""
    def __init__(self):
        self._att = {}

    def get_attachment(self, key):
        return self._att.get(key, None)

    def set_attachments(self, key, val):
        self._att[key] = val


class _ClipWrap:
    """CLIP-like wrapper whose attachments live on its .patcher (real CLIP
    objects don't expose get/set_attachments — only clip.patcher does)."""
    def __init__(self):
        self.patcher = _AttachHost()


class TestLoraNameStampInstaller(unittest.TestCase):
    """Commit 1: _install_lora_name_stamp wraps a LoraLoader-like class so
    every load_lora call appends {name, strength_model, strength_clip} to the
    'loraopt_chain_names' attachment on the returned model (and clip)."""

    ATTACH = "loraopt_chain_names"

    def _fresh_loader_cls(self):
        class _L:
            def load_lora(self, model, clip, lora_name,
                          strength_model, strength_clip):
                return (model, clip)
        return _L

    def test_default_target_never_raises(self):
        # With/without comfy's top-level `nodes` importable, the no-arg call
        # must degrade gracefully (never raise at package import).
        try:
            result = lora_optimizer._install_lora_name_stamp()
        except Exception as e:  # pragma: no cover
            self.fail(f"installer raised: {e}")
        self.assertIn(result, (True, False))

    def test_entries_accumulate_in_chain_order(self):
        L = self._fresh_loader_cls()
        self.assertTrue(lora_optimizer._install_lora_name_stamp(L))
        inst = L()
        m = _AttachHost()
        inst.load_lora(m, None, "styles/a.safetensors", 0.8, 0.7)
        inst.load_lora(m, None, "chars/b.safetensors", 0.5, 0.5)
        stamps = m.get_attachment(self.ATTACH)
        self.assertEqual([s["name"] for s in stamps],
                         ["styles/a.safetensors", "chars/b.safetensors"])
        self.assertEqual(stamps[0]["strength_model"], 0.8)
        self.assertEqual(stamps[0]["strength_clip"], 0.7)
        self.assertEqual(stamps[1]["strength_model"], 0.5)

    def test_builds_new_list_never_mutates_upstream(self):
        # The accumulator MUST build a NEW list each call: ModelPatcher.clone
        # copies the attachment list by REFERENCE, so appending in place would
        # corrupt the upstream (pre-clone) model's chain.
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        inst.load_lora(m, None, "a.safetensors", 1.0, 1.0)
        first = m.get_attachment(self.ATTACH)
        inst.load_lora(m, None, "b.safetensors", 1.0, 1.0)
        second = m.get_attachment(self.ATTACH)
        self.assertIsNot(first, second)      # brand-new list
        self.assertEqual(len(first), 1)      # the earlier list is untouched
        self.assertEqual(len(second), 2)

    def test_double_install_is_idempotent(self):
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        lora_optimizer._install_lora_name_stamp(L)   # second call is a no-op
        inst = L()
        m = _AttachHost()
        inst.load_lora(m, None, "a.safetensors", 1.0, 1.0)
        # Only one wrapper -> one entry per call (double-wrap would give two).
        self.assertEqual(len(m.get_attachment(self.ATTACH)), 1)

    def test_stamping_exception_is_swallowed(self):
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()

        class _Bad:
            def get_attachment(self, key):
                return None

            def set_attachments(self, key, val):
                raise RuntimeError("boom")

        bad = _Bad()
        out = inst.load_lora(bad, None, "a.safetensors", 1.0, 1.0)
        # Original result still returned; loading is never broken by stamping.
        self.assertEqual(out, (bad, None))

    def test_clip_none_handled(self):
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        out = inst.load_lora(m, None, "a.safetensors", 1.0, 0.0)
        self.assertIsNone(out[1])
        self.assertEqual(len(m.get_attachment(self.ATTACH)), 1)

    def test_clip_stamped_via_patcher(self):
        # clip_out is a CLIP wrapper -> stamp lands on clip.patcher, which is
        # exactly where the read path (clip.patcher.get_attachment) looks.
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        c = _ClipWrap()
        inst.load_lora(m, c, "a.safetensors", 0.8, 0.6)
        cs = c.patcher.get_attachment(self.ATTACH)
        self.assertEqual(cs[0]["name"], "a.safetensors")
        self.assertEqual(cs[0]["strength_clip"], 0.6)

    def test_falsy_lora_name_not_stamped(self):
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        inst.load_lora(m, None, "", 1.0, 1.0)
        self.assertIsNone(m.get_attachment(self.ATTACH))

    def test_returns_original_result_object(self):
        # The wrapper must return exactly what the original returned (same
        # model/clip objects) so downstream nodes see the real patched models.
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        c = _ClipWrap()
        out = inst.load_lora(m, c, "a.safetensors", 1.0, 1.0)
        self.assertIs(out[0], m)
        self.assertIs(out[1], c)


class _AttachModel(_FakePatcher):
    """ModelPatcher stand-in that also carries a loraopt_chain_names
    attachment and copies it (shallow) on clone, like the real one."""
    def __init__(self, patches=None, stamps=None):
        super().__init__(patches)
        self._att = {}
        if stamps is not None:
            self._att[lora_optimizer.LORAOPT_CHAIN_NAMES_ATTACH] = stamps
        self.model = types.SimpleNamespace()

    def get_attachment(self, key):
        return self._att.get(key, None)

    def set_attachments(self, key, val):
        self._att[key] = val

    def clone(self):
        n = _AttachModel({k: v[:] for k, v in self.patches.items()})
        n._att = dict(self._att)
        return n


class _AttachClip:
    """CLIP stand-in whose stamps + patches live on its .patcher."""
    def __init__(self, patcher):
        self.patcher = patcher
        self.cond_stage_model = types.SimpleNamespace()

    def clone(self):
        return _AttachClip(self.patcher.clone())


class TestInlineStampMatcher(unittest.TestCase):
    """Commit 2: _resolve_stamp_names aligns stamped filenames to chain groups
    by ORDER and STRENGTH (per branch)."""

    @staticmethod
    def _m(groups, stamps, field):
        return lora_optimizer.LoRAOptimizerInline._resolve_stamp_names(
            groups, stamps, field)

    def test_matches_two_groups_by_strength_model(self):
        groups = [{"strength": 0.8, "entries": {}}, {"strength": 0.5, "entries": {}}]
        stamps = [{"name": "a", "strength_model": 0.8, "strength_clip": 0.3},
                  {"name": "b", "strength_model": 0.5, "strength_clip": 0.4}]
        self.assertEqual(self._m(groups, stamps, "strength_model"), ["a", "b"])

    def test_matches_by_clip_strength_field(self):
        groups = [{"strength": 0.6, "entries": {}}]
        stamps = [{"name": "te", "strength_model": 0.0, "strength_clip": 0.6}]
        self.assertEqual(self._m(groups, stamps, "strength_clip"), ["te"])

    def test_strength_mismatch_leaves_group_unnamed(self):
        groups = [{"strength": 0.9, "entries": {}}]
        stamps = [{"name": "a", "strength_model": 0.2, "strength_clip": 1.0}]
        self.assertEqual(self._m(groups, stamps, "strength_model"), [None])

    def test_fewer_stamps_than_groups_degrades(self):
        groups = [{"strength": 0.8, "entries": {}}, {"strength": 0.5, "entries": {}}]
        stamps = [{"name": "a", "strength_model": 0.8, "strength_clip": 1.0}]
        self.assertEqual(self._m(groups, stamps, "strength_model"), ["a", None])

    def test_empty_stamps_all_unnamed(self):
        groups = [{"strength": 0.8, "entries": {}}]
        self.assertEqual(self._m(groups, [], "strength_model"), [None])

    def test_malformed_stamp_entry_skipped(self):
        groups = [{"strength": 0.8, "entries": {}}]
        self.assertEqual(self._m(groups, ["not a dict"], "strength_model"), [None])


class TestInlineStampReadPath(unittest.TestCase):
    """Commit 2: execute_inline consumes stamped filenames -> real names in the
    fingerprint + file identity on the virtual items (reconciling with the
    file-based Stack community/memory dataset). Unstamped -> captured fallback."""

    def _node(self):
        return lora_optimizer.LoRAOptimizerInline()

    @staticmethod
    def _out(result):
        return result["result"] if isinstance(result, dict) else result

    def _capture_merge(self, node, seen):
        def fake_merge(m, stack, output_strength, **kw):
            seen["model"] = m
            seen["stack"] = stack
            seen.update(kw)
            return (m, kw.get("clip"), "engine report", None, None)
        node.optimize_merge = fake_merge

    @staticmethod
    def _resolves(kind, name):
        return "/loras/" + name

    def test_two_stamps_named_and_get_file_identity(self):
        patches = _chain_patches((0.8, {"a": _adapter()}), (0.5, {"b": _adapter()}))
        stamps = [{"name": "styles/a.safetensors", "strength_model": 0.8,
                   "strength_clip": 0.7},
                  {"name": "chars/b.safetensors", "strength_model": 0.5,
                   "strength_clip": 0.5}]
        model = _AttachModel(patches, stamps)
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               side_effect=self._resolves):
            result = node.execute_inline(model, output_strength=1.0)
        report = self._out(result)[2]
        # Real names surface in the fingerprint.
        self.assertIn("styles/a.safetensors", report)
        self.assertIn("chars/b.safetensors", report)
        stack = seen["stack"]
        self.assertEqual(stack[0]["_resolved_file_name"], "styles/a.safetensors")
        self.assertEqual(stack[1]["_resolved_file_name"], "chars/b.safetensors")
        # Salted display/merge-cache name is kept SEPARATE from the identity.
        self.assertTrue(stack[0]["name"].startswith("chain lora #1"))
        # Persistent identity is the FILE key (reconciles with Stack), not captured:.
        k0 = lora_optimizer.LoRAAutoTuner._persistent_lora_key(stack[0])
        self.assertEqual(k0, "styles/a.safetensors")
        self.assertFalse(k0.startswith("captured:"))

    def test_fewer_stamps_than_groups_leaves_extra_unnamed(self):
        patches = _chain_patches((0.8, {"a": _adapter()}), (0.5, {"b": _adapter()}))
        stamps = [{"name": "only/first.safetensors", "strength_model": 0.8,
                   "strength_clip": 1.0}]
        model = _AttachModel(patches, stamps)
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               side_effect=self._resolves):
            node.execute_inline(model, output_strength=1.0)
        stack = seen["stack"]
        self.assertEqual(stack[0]["_resolved_file_name"], "only/first.safetensors")
        self.assertNotIn("_resolved_file_name", stack[1])   # unmatched -> unnamed

    def test_strength_mismatch_group_stays_unnamed(self):
        patches = _chain_patches((0.8, {"a": _adapter()}))
        stamps = [{"name": "wrong.safetensors", "strength_model": 0.2,
                   "strength_clip": 1.0}]
        model = _AttachModel(patches, stamps)
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               side_effect=self._resolves):
            result = node.execute_inline(model, output_strength=1.0)
        report = self._out(result)[2]
        self.assertNotIn("wrong.safetensors", report)
        self.assertNotIn("_resolved_file_name", seen["stack"][0])

    def test_clip_only_leftover_group_gets_clip_name(self):
        # TE-only LoRA: no model group, one clip group named by its clip stamp.
        model = _AttachModel({}, [])
        clip = _AttachClip(_AttachModel(
            _chain_patches((0.6, {"te.a": _adapter()})),
            [{"name": "te/style.safetensors", "strength_model": 0.0,
              "strength_clip": 0.6}]))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               side_effect=self._resolves):
            result = node.execute_inline(model, output_strength=1.0, clip=clip)
        report = self._out(result)[2]
        self.assertIn("te/style.safetensors", report)
        self.assertEqual(seen["stack"][0]["_resolved_file_name"],
                         "te/style.safetensors")

    def test_stamps_retain_model_only_and_clip_only_holes(self):
        # Loader #1 contributes MODEL only; loader #2 contributes CLIP only.
        # Compacting each branch independently would hybridize A's MODEL with
        # B's CLIP. The longer stamped history supplies the missing slots.
        stamps = [
            {"name": "model/a.safetensors", "strength_model": 0.8,
             "strength_clip": 0.0},
            {"name": "clip/b.safetensors", "strength_model": 0.0,
             "strength_clip": 0.6},
        ]
        model = _AttachModel(
            _chain_patches((0.8, {"model.a": _adapter()})), stamps)
        clip = _AttachClip(_AttachModel(
            _chain_patches((0.6, {"te.b": _adapter()})), [stamps[1]]))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               side_effect=self._resolves):
            node.execute_inline(model, output_strength=1.0, clip=clip)

        self.assertEqual(len(seen["stack"]), 2)
        first, second = seen["stack"]
        self.assertEqual(set(first["lora"]), {"model.a"})
        self.assertEqual(first["clip_strength"], 0.0)
        self.assertEqual(first["_resolved_file_name"], "model/a.safetensors")
        self.assertTrue(first["name"].startswith("chain lora #1"))
        self.assertEqual(set(second["lora"]), {"te.b"})
        self.assertEqual(second["_resolved_file_name"], "clip/b.safetensors")
        self.assertTrue(second["name"].startswith("chain lora #2 (clip-only)"))

    def test_conflicting_branch_histories_pass_through_instead_of_hybridizing(self):
        model_stamp = {"name": "model/a.safetensors", "strength_model": 0.8,
                       "strength_clip": 0.0}
        clip_stamp = {"name": "clip/b.safetensors", "strength_model": 0.0,
                      "strength_clip": 0.6}
        model = _AttachModel(
            _chain_patches((0.8, {"model.a": _adapter()})), [model_stamp])
        clip = _AttachClip(_AttachModel(
            _chain_patches((0.6, {"te.b": _adapter()})), [clip_stamp]))
        node = self._node()
        node.optimize_merge = mock.Mock(
            side_effect=AssertionError("mismatched files must not be merged"))

        result = node.execute_inline(model, output_strength=1.0, clip=clip)
        out = self._out(result)
        self.assertIs(out[0], model)
        self.assertIs(out[1], clip)
        self.assertIn("MODEL/CLIP loader identity mismatch", out[2])
        node.optimize_merge.assert_not_called()

    def test_partial_blocked_slot_withholds_full_file_identity(self):
        stamp = {"name": "style/a.safetensors", "strength_model": 0.8,
                 "strength_clip": 0.6}
        model = _AttachModel(_chain_patches((
            0.8, {"model.dora": _adapter(dora_scale=torch.ones(8, 1))})),
            [stamp])
        clip = _AttachClip(_AttachModel(
            _chain_patches((0.6, {"te.a": _adapter()})), [stamp]))
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        with mock.patch.object(
                lora_optimizer.folder_paths, "get_full_path",
                side_effect=lambda kind, name: (
                    "/loras/style/a.safetensors"
                    if name == "style/a.safetensors" else None)):
            result = node.execute_inline(model, output_strength=1.0, clip=clip)
            item = seen["stack"][0]
            self.assertNotIn("_resolved_file_name", item)
            key = lora_optimizer.LoRAAutoTuner._persistent_lora_key(item)
        self.assertTrue(key.startswith("captured:"))
        self.assertIn("style/a.safetensors", self._out(result)[2])

    def test_no_attachment_keeps_generic_names_and_captured_identity(self):
        # Regression: unstamped loader (fake patcher lacks the attachment API)
        # -> generic "chain lora #N" name + captured: content-hash identity.
        patches = _chain_patches((0.8, {"a": _adapter()}))
        model = _FakePatcher(patches)
        model.model = types.SimpleNamespace()
        node = self._node()
        seen = {}
        self._capture_merge(node, seen)
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            result = node.execute_inline(model, output_strength=1.0)
            report = self._out(result)[2]
            self.assertIn("#1: 1 keys", report)      # generic fingerprint line
            stack = seen["stack"]
            self.assertNotIn("_resolved_file_name", stack[0])
            self.assertTrue(stack[0]["name"].startswith("chain lora #1"))
            # captured identity requires get_full_path -> None (in-memory hash)
            key = lora_optimizer.LoRAAutoTuner._persistent_lora_key(stack[0])
        self.assertTrue(key.startswith("captured:"))


class TestInlineNamedReconciliation(unittest.TestCase):
    """Commit 2: a NAMED inline item shares the memory/community identity with
    a Stack run of the same file (file-bytes content hash + name key)."""

    def test_persistent_key_equals_stack_key(self):
        real = "styles/a.safetensors"
        named = {"name": "chain lora #1 [zzz]", "_precomputed_diffs": True,
                 "_resolved_file_name": real, "strength": 0.8, "lora": {}}
        stack_item = {"name": real, "strength": 0.8, "lora": {}}
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               side_effect=lambda kind, name: "/loras/" + name):
            k_named = lora_optimizer.LoRAAutoTuner._persistent_lora_key(named)
            k_stack = lora_optimizer.LoRAAutoTuner._persistent_lora_key(stack_item)
        self.assertEqual(k_named, k_stack)
        self.assertEqual(k_named, real)

    def test_content_hash_equals_stack_file_hash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as memdir:
            path = os.path.join(memdir, "a.safetensors")
            with open(path, "wb") as f:
                f.write(b"fake lora file bytes " * 64)
            real = "styles/a.safetensors"
            named = {"name": "chain lora #1 [q]", "_precomputed_diffs": True,
                     "_resolved_file_name": real, "strength": 0.8,
                     "lora": {"k": _adapter()}}
            stack_item = {"name": real, "strength": 0.8, "lora": {}}
            with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", memdir), \
                    mock.patch.object(lora_optimizer.folder_paths,
                                      "get_full_path",
                                      side_effect=lambda kind, name: path):
                h_named = lora_optimizer.LoRAAutoTuner._lora_content_hash(named)
                h_stack = lora_optimizer.LoRAAutoTuner._lora_content_hash(
                    stack_item)
            # Named inline item hashes the FILE bytes (via _resolved_file_name),
            # so it reconciles with the Stack path's file-bytes hash.
            self.assertIsNotNone(h_named)
            self.assertEqual(h_named, h_stack)

    def test_unnamed_captured_still_hashes_in_memory(self):
        # Regression: an item WITHOUT _resolved_file_name keeps the captured
        # in-memory content hash (unchanged fallback).
        item = {"name": "chain lora #1 [s]", "_precomputed_diffs": True,
                "strength": 0.8, "lora": {"k": _adapter()}}
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               return_value=None):
            h = lora_optimizer.LoRAAutoTuner._lora_content_hash(item)
            key = lora_optimizer.LoRAAutoTuner._persistent_lora_key(item)
        self.assertIsNotNone(h)
        self.assertTrue(key.startswith("captured:"))


class TestInlineConservativeMatching(unittest.TestCase):
    """Important 2: _resolve_stamp_names must be conservative + order-
    independent. A wrong _resolved_file_name pollutes the shared file-based
    dataset, so name a group ONLY on an unambiguous unique-strength match."""

    @staticmethod
    def _m(groups, stamps, field):
        return lora_optimizer.LoRAOptimizerInline._resolve_stamp_names(
            groups, stamps, field)

    @staticmethod
    def _g(*strengths):
        return [{"strength": s, "entries": {}} for s in strengths]

    @staticmethod
    def _s(*pairs):
        return [{"name": n, "strength_model": sm, "strength_clip": sm}
                for (n, sm) in pairs]

    def test_same_strength_groups_both_unnamed(self):
        # Two groups at the SAME strength -> ambiguous -> BOTH unnamed
        # (better anonymous than a wrong file identity).
        names = self._m(self._g(0.8, 0.8),
                        self._s(("a", 0.8), ("b", 0.8)), "strength_model")
        self.assertEqual(names, [None, None])

    def test_distinct_strengths_reversed_order_named_correctly(self):
        # Reconstruction reversed the group order vs the stamp order: unique
        # strengths must still map correctly by VALUE, not index.
        names = self._m(self._g(0.5, 0.8),
                        self._s(("a", 0.8), ("b", 0.5)), "strength_model")
        self.assertEqual(names, ["b", "a"])

    def test_count_mismatch_unique_named_colliding_unnamed(self):
        # An unstamped loader adds a 3rd group at 0.5: the unique 0.8 group is
        # named; the two colliding 0.5 groups stay unnamed.
        names = self._m(self._g(0.8, 0.5, 0.5),
                        self._s(("a", 0.8), ("b", 0.5)), "strength_model")
        self.assertEqual(names, ["a", None, None])

    def test_duplicate_stamp_strength_leaves_group_unnamed(self):
        # Two stamps share a strength -> the matching group can't be resolved.
        names = self._m(self._g(0.5),
                        self._s(("a", 0.5), ("b", 0.5)), "strength_model")
        self.assertEqual(names, [None])

    def test_near_equal_strengths_within_tol_treated_as_collision(self):
        # 0.8 vs 0.80005 are within tol -> fuzzy-unique fails -> both unnamed
        # (closes the 0.8/0.80001 false-positive window).
        names = self._m(self._g(0.8, 0.80005),
                        self._s(("a", 0.8), ("b", 0.80005)), "strength_model")
        self.assertEqual(names, [None, None])

    def test_wrong_name_never_assigned_in_ambiguous_case(self):
        names = self._m(self._g(0.8, 0.8),
                        self._s(("a", 0.8), ("b", 0.8)), "strength_model")
        self.assertNotIn("a", names)
        self.assertNotIn("b", names)


class TestInlineAmbiguousNoPollution(unittest.TestCase):
    """Integration: an ambiguous (same-strength) captured chain must get NO
    file identity -> falls back to captured identity, so it can never write a
    wrong file's stats into the shared dataset."""

    def _node(self):
        return lora_optimizer.LoRAOptimizerInline()

    @staticmethod
    def _out(result):
        return result["result"] if isinstance(result, dict) else result

    def test_same_strength_chain_gets_no_resolved_file_name(self):
        patches = _chain_patches((0.8, {"a": _adapter()}),
                                 (0.8, {"b": _adapter()}))
        stamps = [{"name": "styles/a.safetensors", "strength_model": 0.8,
                   "strength_clip": 0.8},
                  {"name": "styles/b.safetensors", "strength_model": 0.8,
                   "strength_clip": 0.8}]
        model = _AttachModel(patches, stamps)
        node = self._node()
        seen = {}

        def fake_merge(m, stack, output_strength, **kw):
            seen["stack"] = stack
            return (m, kw.get("clip"), "engine report", None, None)
        node.optimize_merge = fake_merge
        with mock.patch.object(lora_optimizer.folder_paths, "get_full_path",
                               side_effect=lambda kind, name: "/loras/" + name):
            result = node.execute_inline(model, output_strength=1.0)
        report = self._out(result)[2]
        # No real name displayed, no file identity attached.
        self.assertNotIn("styles/a.safetensors", report)
        self.assertNotIn("styles/b.safetensors", report)
        for item in seen["stack"]:
            self.assertNotIn("_resolved_file_name", item)


class TestLoraNameStampHardening(unittest.TestCase):
    """Review hardening: 0/0 early-return guard, signature-agnostic wrapper,
    functools.wraps + __wrapped__-chain idempotency."""

    ATTACH = "loraopt_chain_names"

    def _fresh_loader_cls(self):
        class _L:
            def load_lora(self, model, clip, lora_name,
                          strength_model, strength_clip):
                return (model, clip)
        return _L

    # --- Important 1: strength 0/0 loads must NOT stamp / mutate input ---

    def test_zero_zero_load_adds_no_stamp(self):
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        inst.load_lora(m, None, "a.safetensors", 0.0, 0.0)
        self.assertIsNone(m.get_attachment(self.ATTACH))   # no phantom stamp

    def test_zero_zero_load_does_not_mutate_input_attachment(self):
        # Stock LoraLoader early-returns the INPUT (model, clip) UNCHANGED on a
        # 0/0 load (no clone) -> stamping would mutate a shared upstream model.
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        existing = [{"name": "up.safetensors", "strength_model": 1.0,
                     "strength_clip": 1.0}]
        m.set_attachments(self.ATTACH, existing)
        inst.load_lora(m, None, "a.safetensors", 0.0, 0.0)
        after = m.get_attachment(self.ATTACH)
        self.assertIs(after, existing)     # exact same object, untouched
        self.assertEqual(len(after), 1)    # nothing appended

    def test_nonzero_model_only_load_still_stamps(self):
        # LoraLoaderModelOnly -> strength_clip=0 but strength_model!=0: NOT a
        # 0/0 load, so it must still stamp.
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        inst.load_lora(m, None, "a.safetensors", 0.8, 0.0)
        self.assertEqual(len(m.get_attachment(self.ATTACH)), 1)

    # --- Important 3: signature-agnostic wrapper ---

    def test_stamps_when_called_with_keyword_args(self):
        # ComfyUI invokes node FUNCTION with keyword args from the inputs dict.
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        inst = L()
        m = _AttachHost()
        inst.load_lora(model=m, clip=None, lora_name="a.safetensors",
                       strength_model=0.8, strength_clip=0.6)
        s = m.get_attachment(self.ATTACH)
        self.assertEqual(s[0]["name"], "a.safetensors")
        self.assertEqual(s[0]["strength_model"], 0.8)
        self.assertEqual(s[0]["strength_clip"], 0.6)

    def test_unknown_trailing_arg_does_not_break_loading_or_stamping(self):
        # A future comfy adds a trailing param. The wrapper must pass it
        # through (loading unbroken) AND still stamp from the leading args.
        class _Future:
            def load_lora(self, model, clip, lora_name, sm, sc, extra=None):
                return (model, clip)
        lora_optimizer._install_lora_name_stamp(_Future)
        inst = _Future()
        m = _AttachHost()
        out = inst.load_lora(m, None, "a.safetensors", 0.8, 0.6, "new_param")
        self.assertEqual(out, (m, None))                    # loading unbroken
        self.assertEqual(m.get_attachment(self.ATTACH)[0]["name"],
                         "a.safetensors")

    def test_extraction_failure_skips_stamp_but_not_load(self):
        # An exotic signature we can't extract from must never break loading.
        class _Sig:
            def load_lora(self, model, clip, lora_name):
                return (model, clip)
        lora_optimizer._install_lora_name_stamp(_Sig)
        inst = _Sig()
        m = _AttachHost()
        out = inst.load_lora(m, None, "a.safetensors")
        self.assertEqual(out, (m, None))                    # no crash
        self.assertIsNone(m.get_attachment(self.ATTACH))    # stamp skipped

    # --- Minors: functools.wraps + __wrapped__-chain idempotency ---

    def test_wrapper_preserves_metadata_via_functools_wraps(self):
        class _Named:
            def load_lora(self, model, clip, lora_name, sm, sc):
                "orig docstring"
                return (model, clip)
        orig = _Named.load_lora
        lora_optimizer._install_lora_name_stamp(_Named)
        self.assertEqual(_Named.load_lora.__name__, "load_lora")
        self.assertEqual(_Named.load_lora.__doc__, "orig docstring")
        self.assertIs(_Named.load_lora.__wrapped__, orig)

    def test_no_double_stamp_when_reinstalled_over_third_party_wrap(self):
        L = self._fresh_loader_cls()
        lora_optimizer._install_lora_name_stamp(L)
        ours = L.load_lora

        def third_party(self, *a, **k):
            return ours(self, *a, **k)
        third_party.__wrapped__ = ours        # chain set, no _loraopt_stamped
        L.load_lora = third_party
        self.assertFalse(getattr(third_party, "_loraopt_stamped", False))
        # Re-install (e.g. HotReload re-import) must find us in the chain and
        # refuse to re-wrap.
        self.assertTrue(lora_optimizer._install_lora_name_stamp(L))
        self.assertIs(L.load_lora, third_party)
        inst = L()
        m = _AttachHost()
        inst.load_lora(m, None, "a.safetensors", 1.0, 1.0)
        self.assertEqual(len(m.get_attachment(self.ATTACH)), 1)   # single stamp


if __name__ == "__main__":
    unittest.main()
