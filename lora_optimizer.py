"""
ComfyUI-LoRA-Optimizer
Auto-optimizer node for combining multiple LoRAs via diff-based merging
with TIES conflict resolution and automatic parameter selection.
"""

import torch
import logging
import math
import os
import json
import hashlib
import time
import re
import concurrent.futures
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
from comfy.weight_adapter.lora import LoRAAdapter
from safetensors.torch import save_file


class LoRAStack:
    """
    Node for creating a LoRA stack (input format for LoRAOptimizer).
    Chain multiple Stack nodes to build a list of any length.
    """
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (lora_list, {"tooltip": "Pick a LoRA file to add to the stack. LoRAs are style/character/concept add-ons trained on top of a base model."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "conflict_mode": (["all", "low_conflict", "high_conflict"], {
                    "default": "all",
                    "tooltip": "Filter where this LoRA applies based on conflicts with other LoRAs. "
                               "'all': apply everywhere (default). "
                               "'low_conflict': only apply where this LoRA agrees with the majority — safe, avoids contested weights. "
                               "'high_conflict': only apply where this LoRA disagrees — forces this LoRA to dominate contested regions."
                }),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Connect another LoRA Stack node here to chain multiple LoRAs together."}),
            }
        }
    
    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "add_to_stack"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Adds a LoRA to the stack for use with LoRA Optimizer"
    
    def add_to_stack(self, lora_name, strength, conflict_mode="all", lora_stack=None):
        lora_list = list(lora_stack) if lora_stack else []

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        lora_list.append({
            "name": lora_name,
            "lora": lora,
            "strength": strength,
            "conflict_mode": conflict_mode,
        })
        
        return (lora_list,)


class LoRAStackDynamic:
    """
    Dynamic LoRA stacker — single node with adjustable slot count.
    Simple mode: one strength per LoRA (applies to both model and CLIP).
    Advanced mode: separate model_strength and clip_strength per LoRA.
    Outputs standard (name, model_str, clip_str) tuples.
    """

    MAX_LORAS = 10

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "mode": (["simple", "advanced"], {
                    "tooltip": "Simple: one strength slider per LoRA (controls both image and text). "
                               "Advanced: separate model_strength and clip_strength sliders for fine-tuning how each LoRA affects image generation vs prompt understanding."
                }),
                "lora_count": ("INT", {"default": 3, "min": 1, "max": cls.MAX_LORAS, "step": 1,
                                       "tooltip": "How many LoRA slots to show. Increase to add more LoRAs."}),
            }
        }
        for i in range(1, cls.MAX_LORAS + 1):
            inputs["required"][f"lora_name_{i}"] = (loras, {
                "tooltip": f"LoRA #{i} — pick a LoRA file or leave as 'None' to skip this slot."
            })
            inputs["required"][f"strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                "tooltip": f"How strongly LoRA #{i} affects the output. 1.0 = full effect."
            })
            inputs["required"][f"model_strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                "tooltip": f"How strongly LoRA #{i} affects image generation (visual style, composition)."
            })
            inputs["required"][f"clip_strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                "tooltip": f"How strongly LoRA #{i} affects text understanding (prompt interpretation)."
            })
            inputs["required"][f"conflict_mode_{i}"] = (["all", "low_conflict", "high_conflict"], {
                "default": "all",
                "tooltip": f"LoRA #{i} conflict filter. "
                           f"'all': apply everywhere (default). "
                           f"'low_conflict': only where this LoRA agrees with the majority. "
                           f"'high_conflict': only where this LoRA disagrees."
            })
        inputs["optional"] = {
            "lora_stack": ("LORA_STACK", {"tooltip": "Connect another LoRA Stack node here to add even more LoRAs to the list."}),
        }
        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Dynamic LoRA stacker with adjustable slot count and optional per-LoRA CLIP strength"

    def build_stack(self, mode, lora_count, lora_stack=None, **kwargs):
        loras = []
        for i in range(1, lora_count + 1):
            name = kwargs.get(f"lora_name_{i}", "None")
            if name == "None":
                continue
            conflict_mode = kwargs.get(f"conflict_mode_{i}", "all")
            if mode == "simple":
                wt = kwargs.get(f"strength_{i}", 1.0)
                loras.append((name, wt, wt, conflict_mode))
            else:
                model_str = kwargs.get(f"model_strength_{i}", 1.0)
                clip_str = kwargs.get(f"clip_strength_{i}", 1.0)
                loras.append((name, model_str, clip_str, conflict_mode))
        if lora_stack is not None:
            for l in lora_stack:
                if isinstance(l, dict):
                    if l.get("name", "None") != "None":
                        s = l.get("strength", 1.0)
                        cm = l.get("conflict_mode", "all")
                        loras.append((l["name"], s, s, cm))
                elif isinstance(l, (tuple, list)):
                    if l[0] != "None":
                        loras.append(tuple(l))
                else:
                    loras.append(l)
        return (loras,)


class _LoRAMergeBase:
    """
    Base class for diff-based LoRA merging.

    Computes full weight diffs (Up @ Down x alpha) for LoRAs of any rank,
    then merges the diffs. Supports TIES-Merging (NeurIPS 2023) for
    resolving sign conflicts.

    Not a registered ComfyUI node — subclassed by LoRAOptimizer.
    """

    def __init__(self):
        self.loaded_loras = {}

    @staticmethod
    def _get_compute_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _detect_architecture(lora_sd):
        """
        Detect model architecture from LoRA key patterns.
        Returns: 'zimage', 'flux', 'wan', 'sdxl', 'ltx', 'qwen_image', or 'unknown'.
        """
        keys = list(lora_sd.keys())
        keys_str = ' '.join(k.lower() for k in keys)

        # Z-Image Turbo (Lumina2): layers.N with attention patterns
        # Handles: diffusion_model.layers.N, single_transformer_blocks.N (non-FLUX),
        #          lora_unet_layers_N (Musubi Tuner)
        if any('diffusion_model.layers.' in k and ('attention' in k or 'adaln' in k.lower())
               for k in keys):
            return 'zimage'
        if any('lora_unet_layers_' in k and 'attention' in k.lower() for k in keys):
            return 'zimage'
        # single_transformer_blocks WITHOUT transformer. or Kohya transformer_ prefix = Z-Image
        if any('single_transformer_blocks' in k
               and 'transformer.single_transformer_blocks' not in k
               and 'transformer_single_transformer_blocks' not in k
               for k in keys):
            return 'zimage'

        # FLUX: double/single blocks in various trainer formats
        if any('transformer.single_transformer_blocks' in k or 'transformer.transformer_blocks' in k
               for k in keys):
            return 'flux'
        if any('transformer_single_transformer_blocks' in k or 'transformer_double_blocks' in k
               for k in keys):
            return 'flux'
        if any('double_blocks' in k or 'single_blocks' in k for k in keys):
            return 'flux'

        # Wan: blocks.N with self_attn/cross_attn/ffn
        if any(('blocks.' in k or 'blocks_' in k) and
               any(x in k for x in ['self_attn', 'cross_attn', 'ffn'])
               for k in keys):
            return 'wan'

        # LTX Video: transformer_blocks with attn1/attn2 and adaln_single
        if any('adaln_single' in k for k in keys):
            return 'ltx'
        has_img_mlp = any('img_mlp' in k for k in keys)
        if not has_img_mlp and any(
                'transformer_blocks' in k and ('attn1' in k or 'attn2' in k)
                and 'input_blocks' not in k and 'output_blocks' not in k
                and 'down_blocks' not in k and 'up_blocks' not in k
                and 'middle_block' not in k and 'mid_block' not in k
                for k in keys):
            return 'ltx'

        # Qwen-Image: transformer_blocks with img_mlp/txt_mlp/img_mod/txt_mod
        if any('transformer_blocks' in k and
               any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod', 'add_q_proj'])
               for k in keys):
            return 'qwen_image'

        # SDXL: text encoders or UNet block patterns
        if 'lora_te1_' in keys_str or 'lora_te2_' in keys_str:
            return 'sdxl'
        if any('input_blocks' in k or 'output_blocks' in k for k in keys):
            return 'sdxl'
        if any('down_blocks' in k or 'up_blocks' in k for k in keys):
            return 'sdxl'

        return 'unknown'

    @staticmethod
    def _normalize_keys_zimage(lora_sd):
        """
        Normalize Z-Image Turbo (Lumina2) LoRA keys to a canonical format.

        Handles:
        1. Split fused QKV (attention.qkv) into separate to_q/to_k/to_v
           for per-component conflict analysis during merge.
        2. Remap attention.out -> attention.to_out.0 (diffusers convention).
        3. Standardize Musubi Tuner format (lora_unet_layers_N_...).
        4. Ensure diffusion_model.layers.N prefix.

        Returns new dict with normalized keys. Original dict is not modified.
        """
        normalized = {}
        processed = set()

        # First pass: standardize prefixes (Musubi Tuner -> canonical)
        prefix_fixed = {}
        for k, v in lora_sd.items():
            new_k = k
            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]
            # Musubi Tuner: lora_unet_layers_N_attention_... -> diffusion_model.layers.N.attention...
            if new_k.startswith('lora_unet_'):
                new_k = new_k.replace('lora_unet_', 'diffusion_model.')
                # Convert underscore-separated to dot-separated for known patterns
                new_k = re.sub(r'layers_(\d+)_', r'layers.\1.', new_k)
                new_k = re.sub(r'attention_', 'attention.', new_k)
                new_k = re.sub(r'feed_forward_', 'feed_forward.', new_k)
            # Ensure diffusion_model. prefix
            if new_k.startswith('layers.'):
                new_k = 'diffusion_model.' + new_k
            prefix_fixed[new_k] = v

        # Second pass: split fused QKV and remap output projection
        # Find all layer indices
        layer_pattern = re.compile(r'((?:diffusion_model\.)?layers\.(\d+)\.attention)\.')
        layers_seen = set()
        for k in prefix_fixed:
            m = layer_pattern.search(k)
            if m:
                layers_seen.add((m.group(1), int(m.group(2))))

        for base, layer_idx in layers_seen:
            # --- Split fused QKV ---
            # In fused QKV LoRAs: lora_down/A is [rank, in_features] (shared),
            # lora_up/B is [3*out_features, rank] (fused). Only the up/B
            # matrix needs splitting; the down/A matrix is copied to all three.
            for lora_fmt in [('.lora_A.weight', '.lora_B.weight'),
                             ('.lora_down.weight', '.lora_up.weight'),
                             ('.lora.down.weight', '.lora.up.weight')]:
                down_suffix, up_suffix = lora_fmt
                qkv_down_key = f"{base}.qkv{down_suffix}"
                qkv_up_key = f"{base}.qkv{up_suffix}"

                if qkv_down_key in prefix_fixed and qkv_up_key in prefix_fixed:
                    qkv_down = prefix_fixed[qkv_down_key]
                    qkv_up = prefix_fixed[qkv_up_key]

                    # Only the up/B matrix is fused [3*out, rank] — split it
                    out_dim = qkv_up.shape[0]
                    if out_dim % 3 != 0:
                        continue  # Not valid fused QKV, try next format
                    q_up, k_up, v_up = torch.chunk(qkv_up, 3, dim=0)

                    # Down/A is shared [rank, in] — copy to all three
                    for comp, comp_up in [('to_q', q_up),
                                          ('to_k', k_up),
                                          ('to_v', v_up)]:
                        normalized[f"{base}.{comp}{down_suffix}"] = qkv_down
                        normalized[f"{base}.{comp}{up_suffix}"] = comp_up

                    # Copy alpha (same for all three components)
                    alpha_key = f"{base}.qkv.alpha"
                    if alpha_key in prefix_fixed:
                        for comp in ('to_q', 'to_k', 'to_v'):
                            normalized[f"{base}.{comp}.alpha"] = prefix_fixed[alpha_key]
                        processed.add(alpha_key)

                    processed.add(qkv_down_key)
                    processed.add(qkv_up_key)
                    break  # Found QKV format, don't try others

            # --- Remap output projection: attention.out -> attention.to_out.0 ---
            for lora_fmt in [('.lora_A.weight', '.lora_B.weight'),
                             ('.lora_up.weight', '.lora_down.weight'),
                             ('.lora_B.weight', '.lora_A.weight'),
                             ('.lora.up.weight', '.lora.down.weight')]:
                sfx_a, sfx_b = lora_fmt
                out_a = f"{base}.out{sfx_a}"
                out_b = f"{base}.out{sfx_b}"
                if out_a in prefix_fixed and out_b in prefix_fixed:
                    normalized[f"{base}.to_out.0{sfx_a}"] = prefix_fixed[out_a]
                    normalized[f"{base}.to_out.0{sfx_b}"] = prefix_fixed[out_b]
                    processed.add(out_a)
                    processed.add(out_b)

                    out_alpha = f"{base}.out.alpha"
                    if out_alpha in prefix_fixed:
                        normalized[f"{base}.to_out.0.alpha"] = prefix_fixed[out_alpha]
                        processed.add(out_alpha)
                    break

        # Pass through all unprocessed keys
        for k, v in prefix_fixed.items():
            if k not in processed:
                normalized[k] = v

        return normalized

    # Compound component names in FLUX models where underscores must be preserved
    _FLUX_COMPOUND_NAMES = sorted([
        'img_attn', 'txt_attn', 'img_mlp', 'txt_mlp',
        'img_mod', 'txt_mod', 'img_norm1', 'img_norm2',
        'txt_norm1', 'txt_norm2', 'query_norm', 'key_norm',
        'lora_up', 'lora_down', 'lora_A', 'lora_B',
        'lora_mid', 'redux_up', 'redux_down',
    ], key=len, reverse=True)  # longest first to avoid partial matches

    @classmethod
    def _flux_kohya_underscore_to_dot(cls, rest):
        """Convert Kohya underscore-separated key to dot-separated,
        preserving compound component names like img_attn, lora_up."""
        protected = []
        for i, name in enumerate(cls._FLUX_COMPOUND_NAMES):
            placeholder = f'\x00{i}\x00'
            if name in rest:
                rest = rest.replace(name, placeholder)
                protected.append((placeholder, name))
        rest = rest.replace('_', '.')
        for placeholder, name in protected:
            rest = rest.replace(placeholder, name)
        return rest

    @classmethod
    def _normalize_keys_flux(cls, lora_sd):
        """
        Normalize FLUX LoRA keys from various trainer formats to canonical format.

        Canonical format: diffusion_model.double_blocks.N.* / diffusion_model.single_blocks.N.*

        Handles:
        - AI-Toolkit: transformer.transformer_blocks.N -> double_blocks.N
                      transformer.single_transformer_blocks.N -> single_blocks.N
        - Kohya: lora_transformer_double_blocks_N -> double_blocks.N
                 lora_transformer_single_transformer_blocks_N -> single_blocks.N
        - Standard: double_blocks.N / single_blocks.N (ensure prefix)
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # AI-Toolkit format
            # transformer.single_transformer_blocks.N -> diffusion_model.single_blocks.N
            new_k = re.sub(
                r'^transformer\.single_transformer_blocks\.(\d+)\.',
                r'diffusion_model.single_blocks.\1.', new_k)
            # transformer.transformer_blocks.N -> diffusion_model.double_blocks.N
            new_k = re.sub(
                r'^transformer\.transformer_blocks\.(\d+)\.',
                r'diffusion_model.double_blocks.\1.', new_k)

            # Kohya underscore format — smart replacement preserving compound names
            m = re.match(r'^lora_transformer_single_transformer_blocks_(\d+)_(.*)', new_k)
            if m:
                block_num = m.group(1)
                rest = cls._flux_kohya_underscore_to_dot(m.group(2))
                new_k = f"diffusion_model.single_blocks.{block_num}.{rest}"
            m = re.match(r'^lora_transformer_double_blocks_(\d+)_(.*)', new_k)
            if m:
                block_num = m.group(1)
                rest = cls._flux_kohya_underscore_to_dot(m.group(2))
                new_k = f"diffusion_model.double_blocks.{block_num}.{rest}"

            # Ensure diffusion_model. prefix for standard format
            if new_k.startswith('double_blocks.') or new_k.startswith('single_blocks.'):
                new_k = 'diffusion_model.' + new_k

            # Generic transformer. prefix -> diffusion_model.
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)

            normalized[new_k] = v
        return normalized

    @staticmethod
    def _normalize_keys_wan(lora_sd):
        """
        Normalize Wan LoRA keys from various trainer formats to canonical format.

        Canonical format: diffusion_model.blocks.N.{self_attn,cross_attn,ffn}.*

        Handles LyCORIS, diffusers, Fun LoRA, finetrainer formats.
        Also applies RS-LoRA alpha compensation if detected.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # LyCORIS/aitoolkit format
            if new_k.startswith('lycoris_blocks_'):
                new_k = new_k.replace('lycoris_blocks_', 'blocks.')
                # Add dot separator after block number
                new_k = re.sub(r'^blocks\.(\d+)_', r'blocks.\1.', new_k)
                # Use regex to match both underscore and dot as leading separator
                # (dot comes from the block number fix above)
                new_k = re.sub(r'[._]cross_attn[._]', '.cross_attn.', new_k)
                new_k = re.sub(r'[._]self_attn[._]', '.self_attn.', new_k)
                new_k = re.sub(r'[._]ffn_net_0_proj', '.ffn.0', new_k)
                new_k = re.sub(r'[._]ffn_net_2', '.ffn.2', new_k)
                new_k = new_k.replace('to_out_0', 'o')

            # Diffusers format prefixes
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)
            if new_k.startswith('pipe.dit.'):
                new_k = new_k.replace('pipe.dit.', 'diffusion_model.', 1)
            if new_k.startswith('blocks.'):
                new_k = 'diffusion_model.' + new_k
            if new_k.startswith('vace_blocks.'):
                new_k = 'diffusion_model.' + new_k

            # Common diffusers cleanup
            new_k = new_k.replace('.default.', '.')
            new_k = new_k.replace('.diff_m', '.modulation.diff')

            # Fun LoRA format: lora_unet__blocks_N_...
            if new_k.startswith('lora_unet__'):
                parts = new_k.split('.')
                main_part = parts[0]
                weight_type = '.'.join(parts[1:]) if len(parts) > 1 else None

                if 'blocks_' in main_part:
                    components = main_part[len('lora_unet__'):].split('_')
                    rebuilt = 'diffusion_model'

                    if components[0] == 'blocks':
                        rebuilt += f".blocks.{components[1]}"
                        idx = 2
                        if idx < len(components):
                            if (components[idx] == 'self' and idx + 1 < len(components)
                                    and components[idx + 1] == 'attn'):
                                rebuilt += '.self_attn'
                                idx += 2
                            elif (components[idx] == 'cross' and idx + 1 < len(components)
                                  and components[idx + 1] == 'attn'):
                                rebuilt += '.cross_attn'
                                idx += 2
                            elif components[idx] == 'ffn':
                                rebuilt += '.ffn'
                                idx += 1
                        if idx < len(components):
                            component = components[idx]
                            idx += 1
                            if idx < len(components) and components[idx] == 'img':
                                component += '_img'
                                idx += 1
                            rebuilt += f'.{component}'
                        # Append any remaining components with dot separators
                        while idx < len(components):
                            rebuilt += f'.{components[idx]}'
                            idx += 1

                    if weight_type:
                        if weight_type == 'alpha':
                            rebuilt += '.alpha'
                        elif weight_type in ('lora_down.weight', 'lora_down'):
                            rebuilt += '.lora_A.weight'
                        elif weight_type in ('lora_up.weight', 'lora_up'):
                            rebuilt += '.lora_B.weight'
                        else:
                            rebuilt += f'.{weight_type}'
                            if not rebuilt.endswith('.weight'):
                                rebuilt += '.weight'
                    new_k = rebuilt
                else:
                    new_k = main_part.replace('lora_unet__', 'diffusion_model.')
                    new_k = new_k.replace('_', '.')
                    if weight_type:
                        if weight_type == 'alpha':
                            new_k += '.alpha'
                        elif weight_type in ('lora_down.weight', 'lora_down'):
                            new_k += '.lora_A.weight'
                        elif weight_type in ('lora_up.weight', 'lora_up'):
                            new_k += '.lora_B.weight'
                        else:
                            new_k += f'.{weight_type}'
                            if not new_k.endswith('.weight'):
                                new_k += '.weight'

            # Finetrainer format
            new_k = new_k.replace('.attn1.to_q.', '.self_attn.q.')
            new_k = new_k.replace('.attn1.to_k.', '.self_attn.k.')
            new_k = new_k.replace('.attn1.to_v.', '.self_attn.v.')
            new_k = new_k.replace('.attn1.to_out.0.', '.self_attn.o.')
            new_k = new_k.replace('.attn2.to_q.', '.cross_attn.q.')
            new_k = new_k.replace('.attn2.to_k.', '.cross_attn.k.')
            new_k = new_k.replace('.attn2.to_v.', '.cross_attn.v.')
            new_k = new_k.replace('.attn2.to_out.0.', '.cross_attn.o.')

            normalized[new_k] = v

        # RS-LoRA compensation: detect and fix alpha scaling.
        # RS-LoRA files omit ALL alpha keys and rely on sqrt(rank) scaling.
        # Only apply compensation if the file has zero alpha keys (reliable
        # heuristic — standard LoRAs either include explicit alphas or use
        # rank as default alpha, which _get_lora_key_info handles).
        has_any_alpha = any(nk.endswith('.alpha') for nk in normalized)
        has_any_lora = any(nk.endswith('.lora_A.weight') for nk in normalized)
        if not has_any_alpha and has_any_lora:
            # Find rank from any lora_A weight
            sample_key = next(nk for nk in normalized if nk.endswith('.lora_A.weight'))
            rank = normalized[sample_key].shape[0]
            # alpha/rank = rank^1.5/rank = sqrt(rank), matching RS-LoRA scaling
            corrected_alpha = torch.tensor(rank * (rank ** 0.5))
            for nk in list(normalized.keys()):
                if nk.endswith('.lora_A.weight'):
                    alpha_key = nk.replace('.lora_A.weight', '.alpha')
                    normalized[alpha_key] = corrected_alpha

        return normalized

    @staticmethod
    def _normalize_keys_sdxl(lora_sd):
        """
        Normalize SDXL LoRA keys to canonical format.

        Canonical format: lora_unet_* / lora_te1_* / lora_te2_* (Kohya convention).

        Handles diffusers-format keys (down_blocks.N, up_blocks.N, mid_block).
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # Diffusers format: text_encoder.* -> lora_te1_*, text_encoder_2.* -> lora_te2_*
            if new_k.startswith('text_encoder_2.'):
                new_k = 'lora_te2_' + new_k[len('text_encoder_2.'):].replace('.', '_')
            elif new_k.startswith('text_encoder.'):
                new_k = 'lora_te1_' + new_k[len('text_encoder.'):].replace('.', '_')

            # Diffusers UNet: unet.* -> lora_unet_*
            if new_k.startswith('unet.'):
                new_k = 'lora_unet_' + new_k[len('unet.'):].replace('.', '_')

            normalized[new_k] = v
        return normalized

    @staticmethod
    def _normalize_keys_ltx(lora_sd):
        """
        Normalize LTX Video LoRA keys to canonical format.

        Canonical format: diffusion_model.transformer_blocks.N.attn1/attn2.to_q/to_k/to_v.*

        LTX uses standard separate Q/K/V — only prefix standardization needed.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # Kohya format: lora_unet_transformer_blocks_N_... -> diffusion_model.transformer_blocks.N...
            if new_k.startswith('lora_unet_'):
                new_k = new_k.replace('lora_unet_', 'diffusion_model.')
                # Convert underscores back to dots for known structural segments
                new_k = re.sub(r'transformer_blocks_(\d+)_', r'transformer_blocks.\1.', new_k)
                new_k = re.sub(r'attn(\d)_', r'attn\1.', new_k)
                new_k = re.sub(r'to_(q|k|v|out)_', r'to_\1.', new_k)
                new_k = re.sub(r'ff_net_(\d+)_', r'ff.net.\1.', new_k)

            # Diffusers format: unet.* -> diffusion_model.*
            if new_k.startswith('unet.'):
                new_k = new_k.replace('unet.', 'diffusion_model.', 1)

            # Ensure diffusion_model. prefix
            if new_k.startswith('transformer_blocks.'):
                new_k = 'diffusion_model.' + new_k

            normalized[new_k] = v
        return normalized

    @staticmethod
    def _normalize_keys_qwen_image(lora_sd):
        """
        Normalize Qwen-Image LoRA keys to canonical format.

        Canonical format: diffusion_model.transformer_blocks.N.*

        Qwen-Image uses separate Q/K/V with dual-stream (image+text) attention.
        Supports transformer.*, lycoris_*, and direct key formats.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Strip PEFT prefix first so subsequent checks work
            if new_k.startswith('base_model.model.'):
                new_k = new_k[len('base_model.model.'):]

            # LyCORIS format: lycoris_transformer_blocks_N_... -> diffusion_model.transformer_blocks.N...
            if new_k.startswith('lycoris_'):
                new_k = new_k.replace('lycoris_', 'diffusion_model.')
                new_k = re.sub(r'transformer_blocks_(\d+)_', r'transformer_blocks.\1.', new_k)
                # Restore dots for known component names
                for comp in ['to_q', 'to_k', 'to_v', 'add_q_proj', 'add_k_proj', 'add_v_proj',
                             'img_mlp', 'txt_mlp', 'img_mod', 'txt_mod', 'img_norm1', 'img_norm2',
                             'txt_norm1', 'txt_norm2']:
                    new_k = new_k.replace(f'_{comp}_', f'.{comp}.')
                    if new_k.endswith(f'_{comp}'):
                        new_k = new_k[:-len(f'_{comp}')] + f'.{comp}'

            # transformer.* -> diffusion_model.*
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)

            # Ensure diffusion_model. prefix
            if new_k.startswith('transformer_blocks.'):
                new_k = 'diffusion_model.' + new_k

            normalized[new_k] = v
        return normalized

    @classmethod
    def _normalize_keys(cls, lora_sd, architecture):
        """
        Dispatch to architecture-specific key normalizer.
        Returns a new dict with normalized keys.
        """
        if architecture == 'zimage':
            return cls._normalize_keys_zimage(lora_sd)
        elif architecture == 'flux':
            return cls._normalize_keys_flux(lora_sd)
        elif architecture == 'wan':
            return cls._normalize_keys_wan(lora_sd)
        elif architecture == 'sdxl':
            return cls._normalize_keys_sdxl(lora_sd)
        elif architecture == 'ltx':
            return cls._normalize_keys_ltx(lora_sd)
        elif architecture == 'qwen_image':
            return cls._normalize_keys_qwen_image(lora_sd)
        return lora_sd  # unknown — pass through unchanged

    @staticmethod
    def _refuse_zimage_patches(patches):
        """
        Re-fuse split to_q/to_k/to_v patches back into fused QKV patches
        for Z-Image Turbo models. Also remaps to_out.0 -> out.

        Called after merging, before applying patches to the model.
        Returns a new dict with fused patches.
        """
        fused = {}
        qkv_groups = {}  # base -> {comp: (key, patch)}

        for key, patch in patches.items():
            # Handle both string keys and tuple keys
            if isinstance(key, tuple):
                key_str = key[0]
            else:
                key_str = key

            # Detect to_q/to_k/to_v patterns in the key
            m = re.search(r'(layers\.\d+\.attention)\.to_(q|k|v)(?:\.|$)', key_str)
            if m:
                base = m.group(1)
                comp = m.group(2)
                if base not in qkv_groups:
                    qkv_groups[base] = {}
                qkv_groups[base][comp] = (key, patch)
                continue

            # Detect to_out.0 -> out remap
            m_out = re.search(r'(layers\.\d+\.attention)\.to_out\.0(?:\.|$)', key_str)
            if m_out:
                new_key_str = key_str.replace('.to_out.0', '.out')
                if isinstance(key, tuple):
                    new_key = (new_key_str,) + key[1:]
                else:
                    new_key = new_key_str
                fused[new_key] = patch
                continue

            # Not a QKV or out key — pass through
            fused[key] = patch

        # Fuse QKV groups
        for base, comps in qkv_groups.items():
            if len(comps) == 3 and 'q' in comps and 'k' in comps and 'v' in comps:
                # All three components present — fuse
                q_key, q_patch = comps['q']
                k_key, k_patch = comps['k']
                v_key, v_patch = comps['v']

                # Build the fused key name
                if isinstance(q_key, tuple):
                    fused_key_str = re.sub(r'\.to_q(?=\.|$)', '.qkv', q_key[0])
                    fused_key = (fused_key_str,) + q_key[1:]
                else:
                    fused_key_str = re.sub(r'\.to_q(?=\.|$)', '.qkv', q_key)
                    fused_key = fused_key_str

                # Handle different patch formats
                if isinstance(q_patch, tuple) and q_patch[0] == "diff":
                    # Full-rank diff patch: ("diff", (tensor,))
                    q_diff = q_patch[1][0]
                    k_diff = k_patch[1][0]
                    v_diff = v_patch[1][0]
                    fused_diff = torch.cat([q_diff, k_diff, v_diff], dim=0)
                    fused[fused_key] = ("diff", (fused_diff,))
                elif isinstance(q_patch, LoRAAdapter):
                    q_data = q_patch.weights
                    k_data = k_patch.weights
                    v_data = v_patch.weights
                    # weights = (mat_up, mat_down, alpha, mid, dora_scale, reshape)

                    # Check if down matrices are shared (true for original LoRA,
                    # false for independently SVD-compressed patches)
                    if q_data[1] is k_data[1] and k_data[1] is v_data[1]:
                        # Shared down: concatenate ups, keep one down copy
                        fused_up = torch.cat([q_data[0], k_data[0], v_data[0]], dim=0)
                        fused_down = q_data[1]
                        fused_alpha = q_data[2]
                        fused_patch = LoRAAdapter(set(), (fused_up, fused_down, fused_alpha, None, None, None))
                        fused[fused_key] = fused_patch
                    else:
                        # Independent decompositions (e.g., SVD-compressed) —
                        # expand to full-rank diffs, then fuse as a single diff patch
                        parts = []
                        for comp_data in [q_data, k_data, v_data]:
                            alpha = comp_data[2] if comp_data[2] is not None else float(comp_data[1].shape[0])
                            rank = comp_data[1].shape[0]
                            diff = torch.mm(comp_data[0].float(), comp_data[1].float()) * (alpha / rank)
                            parts.append(diff)
                        fused_diff = torch.cat(parts, dim=0)
                        fused[fused_key] = ("diff", (fused_diff,))
                else:
                    # Unknown patch format — pass through unfused
                    for comp_key, comp_patch in [(q_key, q_patch), (k_key, k_patch), (v_key, v_patch)]:
                        fused[comp_key] = comp_patch
            else:
                # Incomplete QKV group — pass through individual components
                for comp, (comp_key, comp_patch) in comps.items():
                    fused[comp_key] = comp_patch

        return fused

    def _load_lora(self, lora_name):
        """Loads LoRA file with caching"""
        if lora_name == "None" or lora_name is None:
            return None
        if lora_name in self.loaded_loras:
            return self.loaded_loras[lora_name]
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_loras[lora_name] = lora
        return lora
    
    def _get_lora_key_info(self, lora_dict, key_prefix):
        """
        Extracts LoRA information for the given key.
        Returns (mat_up, mat_down, alpha, mid) or None.
        """
        # LoRA key formats
        formats = [
            ("{}.lora_up.weight", "{}.lora_down.weight"),           # regular
            ("{}_lora.up.weight", "{}_lora.down.weight"),           # diffusers
            ("{}.lora_B.weight", "{}.lora_A.weight"),               # diffusers2
            ("{}.lora.up.weight", "{}.lora.down.weight"),           # diffusers3
        ]
        
        for up_fmt, down_fmt in formats:
            up_key = up_fmt.format(key_prefix)
            down_key = down_fmt.format(key_prefix)
            
            if up_key in lora_dict and down_key in lora_dict:
                mat_up = lora_dict[up_key]
                mat_down = lora_dict[down_key]
                
                # Alpha
                alpha_key = "{}.alpha".format(key_prefix)
                alpha = lora_dict.get(alpha_key, None)
                if alpha is not None:
                    alpha = alpha.item()
                else:
                    alpha = mat_down.shape[0]  # rank as default
                
                # Mid (for LoCon)
                mid_key = "{}.lora_mid.weight".format(key_prefix)
                mid = lora_dict.get(mid_key, None)
                
                return (mat_up, mat_down, alpha, mid)
        
        return None
    
    def _compute_lora_diff(self, mat_up, mat_down, alpha, mid, target_shape, device=None):
        """
        Computes full diff for a single LoRA.
        diff = mat_up @ mat_down × (alpha / rank)
        When device is given, matrices are moved there for faster matmul,
        then the result is returned on CPU to avoid VRAM accumulation.
        """
        rank = mat_down.shape[0]
        scale = alpha / rank

        if device is not None:
            mat_up = mat_up.to(device)
            mat_down = mat_down.to(device)
            if mid is not None:
                mid = mid.to(device)

        if mid is not None:
            # LoCon with mid matrix (rare)
            final_shape = [mat_down.shape[1], mat_down.shape[0], mid.shape[2], mid.shape[3]]
            mat_down = (
                torch.mm(
                    mat_down.transpose(0, 1).flatten(start_dim=1).float(),
                    mid.transpose(0, 1).flatten(start_dim=1).float(),
                )
                .reshape(final_shape)
                .transpose(0, 1)
            )

        # Compute diff
        diff = torch.mm(
            mat_up.flatten(start_dim=1).float(),
            mat_down.flatten(start_dim=1).float()
        )

        # Try to reshape to target shape
        try:
            diff = diff.reshape(target_shape)
        except RuntimeError:
            # If shape doesn't match, skip
            return None

        diff = diff * scale
        if device is not None and device.type != "cpu":
            return diff.cpu()
        return diff

    @staticmethod
    def _ties_trim(tensor, density):
        """
        TIES Step 1: Trim — keep only the top-k% values by absolute magnitude.
        Everything else is zeroed out (noise removal).
        """
        flat = tensor.flatten()
        n = flat.numel()
        k = max(1, int(n * density))
        if k >= n:
            return tensor.clone()
        _, indices = torch.topk(flat.abs(), k)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[indices] = True
        return (flat * mask).reshape(tensor.shape)

    @staticmethod
    def _dare_sparsify(tensor, density, generator=None):
        """
        DARE sparsification: randomly drop parameters and rescale survivors.
        Each element is kept with probability `density`, then rescaled by 1/density.
        """
        if density >= 1.0:
            return tensor
        mask = torch.bernoulli(
            torch.full(tensor.shape, density, dtype=tensor.dtype, device=tensor.device),
            generator=generator
        )
        return tensor * mask * (1.0 / density)

    @staticmethod
    def _della_sparsify(tensor, density, epsilon=0.3, generator=None):
        """
        DELLA sparsification: magnitude-aware dropout.
        Low-magnitude elements are dropped with higher probability.
        Survivors are rescaled by 1/(1-p_i) to preserve expected value.
        """
        if density >= 1.0:
            return tensor
        original_shape = tensor.shape
        mat = tensor.unsqueeze(0) if tensor.dim() < 2 else tensor.reshape(tensor.shape[0], -1)
        nrows, ncols = mat.shape
        p_min = max((1.0 - density) - epsilon / 2.0, 0.0)
        # Double-argsort gives ascending magnitude ranks; invert so low-magnitude
        # elements get HIGH drop probability (rank 0 = highest magnitude → p_min)
        asc_ranks = mat.abs().argsort(dim=1).argsort(dim=1).float()
        ranks = (ncols - 1) - asc_ranks
        drop_probs = (p_min + (epsilon / ncols) * ranks).clamp(0.0, 1.0)
        keep_probs = 1.0 - drop_probs
        mask = torch.bernoulli(keep_probs, generator=generator)
        rescale = torch.where(mask > 0, 1.0 / keep_probs.clamp(min=1e-6), torch.zeros_like(keep_probs))
        return (mat * mask * rescale).reshape(original_shape)

    @staticmethod
    def _compute_conflict_mask(diffs_with_weights):
        """
        Boolean mask: True where 2+ diffs have opposing signs (actual interference).
        Uses sign-corrected diffs (weight sign applied).
        """
        has_positive = torch.zeros_like(diffs_with_weights[0][0], dtype=torch.bool)
        has_negative = torch.zeros_like(has_positive)
        for diff, weight in diffs_with_weights:
            effective = diff if weight >= 0 else -diff
            nonzero = effective != 0
            has_positive |= (nonzero & (effective > 0))
            has_negative |= (nonzero & (effective < 0))
        return has_positive & has_negative

    @staticmethod
    def _dare_sparsify_conflict(tensor, conflict_mask, density, generator=None):
        if density >= 1.0:
            return tensor
        rand_mask = torch.bernoulli(
            torch.full(tensor.shape, density, dtype=tensor.dtype, device=tensor.device),
            generator=generator
        )
        return torch.where(conflict_mask, tensor * rand_mask * (1.0 / density), tensor)

    @staticmethod
    def _della_sparsify_conflict(tensor, conflict_mask, density, epsilon=0.3, generator=None):
        if density >= 1.0:
            return tensor
        della_result = _LoRAMergeBase._della_sparsify(tensor, density, epsilon, generator)
        return torch.where(conflict_mask, della_result, tensor)

    @staticmethod
    def _ties_elect_sign(trimmed_diffs, method="frequency"):
        """
        TIES Step 2: Elect Sign — determine majority sign direction per weight position.

        Args:
            trimmed_diffs: list of trimmed diff tensors (same shape)
            method: "frequency" (count votes) or "total" (sum magnitudes)

        Returns:
            majority_sign: tensor of +1/-1 per position
        """
        # Iterate instead of torch.stack to avoid allocating [N, *shape] tensor
        ref = trimmed_diffs[0]
        total = torch.zeros_like(ref, dtype=torch.float32)
        if method == "total":
            for d in trimmed_diffs:
                total.add_(d.to(dtype=torch.float32))
        else:
            # sign() returns -1/0/+1: zeros don't vote, matching original behavior
            for d in trimmed_diffs:
                total.add_(d.sign())
        # +1 where majority is positive or tied, -1 where majority is negative
        majority_sign = torch.where(total >= 0,
                                    torch.tensor(1.0, device=total.device, dtype=total.dtype),
                                    torch.tensor(-1.0, device=total.device, dtype=total.dtype))
        return majority_sign

    @staticmethod
    def _ties_disjoint_merge(trimmed_diffs, weights, majority_sign):
        """
        TIES Step 3: Disjoint Merge — average only contributors that agree
        with the elected majority sign at each position.

        Args:
            trimmed_diffs: list of trimmed diff tensors
            weights: list of scalar weights corresponding to each diff
            majority_sign: tensor of +1/-1 per position

        Returns:
            merged tensor
        """
        result = torch.zeros_like(trimmed_diffs[0], dtype=torch.float32)
        contributor_count = torch.zeros_like(result)

        for diff, weight in zip(trimmed_diffs, weights):
            diff_f = diff.to(dtype=torch.float32)
            # Positions where diff agrees with elected majority sign (and is non-zero)
            sign_match = (diff_f * majority_sign) > 0
            result.add_(torch.where(sign_match, diff_f * weight,
                                    torch.tensor(0.0, device=result.device, dtype=result.dtype)))
            contributor_count.add_(sign_match.float())

        # Average by number of contributors (avoid div-by-zero)
        contributor_count.clamp_(min=1.0)
        return result.div_(contributor_count)

    @staticmethod
    @torch.no_grad()
    def _compress_to_lowrank(diff, rank, svd_device=None):
        """
        Re-compress a full-rank diff tensor to low-rank via truncated SVD.
        Returns ("lora", (mat_up, mat_down, alpha=rank, None)) so ComfyUI
        computes up @ down * (rank/rank) = up @ down (no extra scaling).

        svd_device: where to run SVD. GPU is ~10-50x faster. CPU if None.
        For a [4096, 4096] diff at rank 128: 64MB → 2MB (~32x reduction).
        """
        original_shape = diff.shape
        # Reshape to 2D for SVD: [out_features, in_features]
        mat = diff.reshape(original_shape[0], -1).float()
        rank = min(rank, min(mat.shape))

        # Move to requested device for SVD (GPU is much faster for matmul-heavy randomized SVD)
        if svd_device is not None and mat.device != svd_device:
            mat = mat.to(svd_device)
        U, S, V = torch.svd_lowrank(mat, q=rank)
        del mat
        # U: [out, rank], S: [rank], V: [in, rank]
        # Reconstruct as: mat_up = U * sqrt(S), mat_down = sqrt(S) * V^T
        # Return on CPU for storage (ComfyUI moves to device when applying)
        sqrt_S = S.sqrt()
        mat_up = (U * sqrt_S.unsqueeze(0)).cpu()    # [out, rank]
        mat_down = ((V * sqrt_S.unsqueeze(0)).T).cpu()  # [rank, in]
        del U, S, V, sqrt_S
        # alpha=rank so ComfyUI computes: up @ down * (rank/rank) = up @ down
        return LoRAAdapter(set(), (mat_up, mat_down, float(rank), None, None, None))

    @torch.no_grad()
    def _merge_diffs(self, diffs_with_weights, mode, density=0.5, majority_sign_method="frequency",
                     compute_device=None, sparsification="disabled",
                     sparsification_density=0.7, sparsification_generator=None):
        """
        Merges a list of diffs with their weights.
        When compute_device is given, tensors are moved there for faster ops,
        then the result is returned on CPU.
        """
        if len(diffs_with_weights) == 0:
            return None

        if len(diffs_with_weights) == 1:
            diff, weight = diffs_with_weights[0]
            result = diff * weight
            if compute_device is not None and compute_device.type != "cpu" and result.is_cuda:
                return result.cpu()
            return result

        # All diffs should have the same shape (verified during computation)
        ref_diff = diffs_with_weights[0][0]
        dtype = ref_diff.dtype
        dev = compute_device if compute_device is not None else ref_diff.device
        to_cpu = compute_device is not None and compute_device.type != "cpu"

        # DARE/DELLA preprocessing for non-TIES modes
        # (TIES replaces its trim step instead — handled in the ties branch)
        if sparsification != "disabled" and mode != "ties":
            is_conflict = sparsification in ("dare_conflict", "della_conflict")

            if is_conflict:
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    diffs_with_weights[idx] = (diff.to(device=dev, dtype=torch.float32), weight)
                conflict_mask = self._compute_conflict_mask(diffs_with_weights)

                sparsify_fn = (self._dare_sparsify_conflict
                               if sparsification == "dare_conflict"
                               else self._della_sparsify_conflict)
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    diff = sparsify_fn(diff, conflict_mask, sparsification_density,
                                       generator=sparsification_generator)
                    diffs_with_weights[idx] = (diff.to(dtype), weight)
                del conflict_mask
            else:
                sparsify_fn = (self._dare_sparsify if sparsification == "dare"
                               else self._della_sparsify)
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    diff = diff.to(device=dev, dtype=torch.float32)
                    diff = sparsify_fn(diff, sparsification_density,
                                       generator=sparsification_generator)
                    diffs_with_weights[idx] = (diff.to(dtype), weight)

        if mode == "weighted_average":
            result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            total_weight = sum(abs(w) for _, w in diffs_with_weights)
            if total_weight == 0:
                return result.to(dtype).cpu() if to_cpu else result.to(dtype)
            for idx in range(len(diffs_with_weights)):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                result.add_(diff.to(device=dev, dtype=torch.float32) * (weight / total_weight))
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "weighted_sum":
            result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            for idx in range(len(diffs_with_weights)):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                result.add_(diff.to(device=dev, dtype=torch.float32) * weight)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "normalize":
            # Normalization by "energy" (sum of squared weights)
            weights = [w for _, w in diffs_with_weights]
            sum_sq = sum(w*w for w in weights)
            if sum_sq == 0:
                z = torch.zeros(ref_diff.shape, device=dev)
                return z.cpu() if to_cpu else z
            scale = 1.0 / math.sqrt(sum_sq)

            result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            for idx in range(len(diffs_with_weights)):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                result.add_(diff.to(device=dev, dtype=torch.float32) * weight * scale)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "slerp":
            # Spherical Linear Interpolation — magnitude-preserving blend for 2 diffs.
            # result = sin((1-t)*θ)/sin(θ) * v1 + sin(t*θ)/sin(θ) * v2
            # where θ is the angle between v1 and v2, t is derived from weights.
            if len(diffs_with_weights) != 2:
                # Fallback to weighted_average for != 2 diffs
                mode = "weighted_average"
                result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
                total_weight = sum(abs(w) for _, w in diffs_with_weights)
                if total_weight == 0:
                    return result.to(dtype).cpu() if to_cpu else result.to(dtype)
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    diffs_with_weights[idx] = None
                    result.add_(diff.to(device=dev, dtype=torch.float32) * (weight / total_weight))
                result = result.to(dtype)
                return result.cpu() if to_cpu else result

            d1, w1 = diffs_with_weights[0]
            d2, w2 = diffs_with_weights[1]
            diffs_with_weights[0] = None
            diffs_with_weights[1] = None
            v1 = d1.to(device=dev, dtype=torch.float32).flatten()
            del d1
            v2 = d2.to(device=dev, dtype=torch.float32).flatten()
            del d2
            # Negative weights negate the diff direction (same as TIES/weighted_avg)
            if w1 < 0:
                v1 = -v1
            if w2 < 0:
                v2 = -v2

            # t = interpolation factor from weights (0 = all v1, 1 = all v2)
            total_w = abs(w1) + abs(w2)
            t = abs(w2) / total_w if total_w > 0 else 0.5

            # Compute angle between vectors
            dot = torch.dot(v1, v2)
            norm1 = v1.norm()
            norm2 = v2.norm()
            denom = norm1 * norm2
            if denom > 0:
                cos_theta = (dot / denom).clamp(-1.0, 1.0)
            else:
                cos_theta = torch.tensor(1.0, device=dev)
            theta = torch.acos(cos_theta)

            # If vectors are nearly parallel, fall back to linear interpolation
            if theta.item() < 1e-6:
                result = ((1.0 - t) * v1 + t * v2).reshape(ref_diff.shape)
            else:
                sin_theta = torch.sin(theta)
                a = torch.sin((1.0 - t) * theta) / sin_theta
                b = torch.sin(t * theta) / sin_theta
                result = (a * v1 + b * v2).reshape(ref_diff.shape)
            del v1, v2
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "ties":
            # TIES-Merging: Trim, Elect Sign, Disjoint Merge
            # Pre-multiply diffs by sign(weight) so negative strengths vote correctly,
            # then use abs(weight) for magnitude in disjoint merge.
            # Memory-optimized: free input diffs after trimming to reduce peak VRAM.
            trimmed = []
            abs_weights = []
            is_conflict = sparsification in ("dare_conflict", "della_conflict")

            if is_conflict:
                signed_diffs = []
                for idx in range(len(diffs_with_weights)):
                    d, w = diffs_with_weights[idx]
                    diffs_with_weights[idx] = None
                    d_f = d.to(device=dev, dtype=torch.float32)
                    del d
                    if w < 0:
                        d_f = -d_f
                    signed_diffs.append(d_f)
                    abs_weights.append(abs(w))

                conflict_mask = self._compute_conflict_mask(
                    [(d, 1.0) for d in signed_diffs])

                sparsify_fn = (self._dare_sparsify_conflict
                               if sparsification == "dare_conflict"
                               else self._della_sparsify_conflict)
                for d_f in signed_diffs:
                    trimmed.append(sparsify_fn(d_f, conflict_mask, sparsification_density,
                                               generator=sparsification_generator))
                del signed_diffs, conflict_mask
            else:
                for idx in range(len(diffs_with_weights)):
                    d, w = diffs_with_weights[idx]
                    diffs_with_weights[idx] = None  # Free input diff early
                    d_f = d.to(device=dev, dtype=torch.float32)
                    del d
                    if w < 0:
                        d_f = -d_f
                    # DARE/DELLA replaces TIES trim step when enabled
                    if sparsification == "dare":
                        trimmed.append(self._dare_sparsify(d_f, sparsification_density, generator=sparsification_generator))
                    elif sparsification == "della":
                        trimmed.append(self._della_sparsify(d_f, sparsification_density, generator=sparsification_generator))
                    else:
                        trimmed.append(self._ties_trim(d_f, density))
                    abs_weights.append(abs(w))

            # Step 2: Elect majority sign
            majority_sign = self._ties_elect_sign(trimmed, majority_sign_method)

            # Step 3: Disjoint merge
            result = self._ties_disjoint_merge(trimmed, abs_weights, majority_sign)
            del trimmed, majority_sign
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        return None


class LoRAOptimizer(_LoRAMergeBase):
    """
    Auto-optimizer that analyzes a LoRA stack (sign conflicts, magnitude
    distributions, overlap) and automatically selects the best merge mode
    and parameters, then performs the merge.

    Outputs the merged model/clip plus an analysis report explaining
    what was chosen and why.

    Two-pass streaming architecture:
      Pass 1 — Analysis: computes diffs per weight prefix, samples conflict
        and magnitude statistics, then discards diffs immediately. Only
        lightweight scalars and small sample tensors are kept in memory.
      Pass 2 — Merge: recomputes diffs per prefix and merges with the
        auto-selected strategy. Each prefix's diffs are freed after merging.
    Peak memory is ~one prefix's diffs at a time (~260MB) regardless of
    the total number of LoRAs or weight prefixes.

    Limitation: the optimizer only analyzes LoRAs in its own stack. It has
    no visibility into LoRA patches already applied to the model by upstream
    nodes (via Load LoRA, etc.). Those patches stack additively on top of
    the optimizer's output, which could cause overexposure. Fully baked
    merges (safetensors checkpoints) are indistinguishable from base weights
    and cannot be detected at all.
    """

    def __init__(self):
        self.loaded_loras = {}
        self._merge_cache = {}  # single-entry: {cache_key: (model_patches, clip_patches, report, clip_strength_out)}
        self._detected_arch = None

    def _normalize_stack(self, lora_stack, normalize_keys="disabled"):
        """
        Normalize a LoRA stack into a consistent list of dicts.

        Accepts two formats:
        - Standard tuples: [(lora_name, model_strength, clip_strength), ...]
          Used by Efficiency Nodes, Comfyroll, and other popular node packs.
          LoRAs are loaded from disk (cached in self.loaded_loras).
        - LoRAStack dicts: [{"name": str, "lora": dict, "strength": float}, ...]
          Already loaded, clip_strength defaults to None (use global multiplier).

        Returns list of dicts with keys: name, lora, strength, clip_strength.
        clip_strength is None when the global multiplier should be used.
        """
        if not lora_stack:
            return []

        first = lora_stack[0]
        normalized = []

        if isinstance(first, (tuple, list)):
            # Standard format: (lora_name, model_strength, clip_strength[, conflict_mode])
            for entry in lora_stack:
                if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                    logging.warning("[LoRA Optimizer] Skipping malformed tuple entry (expected 3 elements)")
                    continue
                lora_name, model_str, clip_str = entry[0], entry[1], entry[2]
                conflict_mode = entry[3] if len(entry) > 3 else "all"

                # Load LoRA with caching
                if lora_name in self.loaded_loras:
                    lora_dict = self.loaded_loras[lora_name]
                else:
                    try:
                        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                        lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        self.loaded_loras[lora_name] = lora_dict
                    except Exception as e:
                        logging.warning(f"[LoRA Optimizer] Failed to load LoRA '{lora_name}': {e}")
                        continue

                normalized.append({
                    "name": lora_name,
                    "lora": lora_dict,
                    "strength": model_str,
                    "clip_strength": clip_str,
                    "conflict_mode": conflict_mode,
                })

        elif isinstance(first, dict):
            # LoRAStack format: already loaded dicts
            for item in lora_stack:
                if not isinstance(item, dict) or "lora" not in item or "strength" not in item or "name" not in item:
                    logging.warning("[LoRA Optimizer] Skipping malformed dict entry (expected keys: name, lora, strength)")
                    continue
                normalized.append({
                    "name": item["name"],
                    "lora": item["lora"],
                    "strength": item["strength"],
                    "clip_strength": None,  # use global multiplier
                    "conflict_mode": item.get("conflict_mode", "all"),
                })

        else:
            logging.warning("[LoRA Optimizer] Unrecognized stack format")
            return []

        # Architecture-aware key normalization
        if normalize_keys == "enabled" and len(normalized) > 0:
            # Detect from first LoRA that yields a known architecture
            arch = "unknown"
            for item in normalized:
                detected = self._detect_architecture(item["lora"])
                if detected != "unknown":
                    arch = detected
                    break
            if arch != "unknown":
                logging.info(f"[LoRA Optimizer] Architecture detected: {arch}")
                logging.info(f"[LoRA Optimizer] Normalizing keys for {len(normalized)} LoRAs...")
                for item in normalized:
                    item["lora"] = self._normalize_keys(item["lora"], arch)
                self._detected_arch = arch
            else:
                logging.info("[LoRA Optimizer] Architecture: unknown (no key normalization applied)")
                self._detected_arch = None
        else:
            self._detected_arch = None

        return normalized

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Your base model (e.g. SDXL, Flux). The merged LoRAs will be applied to it."}),
                "lora_stack": ("LORA_STACK", {"tooltip": "Connect a LoRA Stack node here. This is the list of LoRAs you want to merge together."}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                                              "tooltip": "Master volume for the merged result. 1.0 = full effect, 0.5 = half, 0 = no effect. Start at 1.0 and lower if the result looks too strong."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "The text encoder. Connect this so LoRAs can also affect how your prompts are understood. Leave empty for video models that don't use CLIP."}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                                                       "tooltip": "How strongly LoRAs affect text understanding, relative to output_strength. At 1.0, CLIP uses the same strength as the model. Lower values reduce LoRA influence on prompt interpretation while keeping the visual effect."}),
                "auto_strength": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Automatically turns down individual LoRA strengths when combining many LoRAs to avoid oversaturated or distorted results. Useful when stacking 3+ LoRAs."
                }),
                "free_vram_between_passes": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Frees GPU memory between processing steps. Enable if you're running out of VRAM. Barely affects speed."
                }),
                "optimization_mode": (["per_prefix", "global", "weighted_sum_only"], {
                    "default": "per_prefix",
                    "tooltip": "How the optimizer decides to combine LoRAs. 'per_prefix' (recommended): automatically picks the best method for each layer. 'global': uses one method everywhere. 'weighted_sum_only': simple addition, no TIES trimming — use this if your stack includes edit, distillation, or DPO LoRAs whose weights must be preserved exactly."
                }),
                "cache_patches": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": "Keep the merge result in memory so re-running the workflow is instant (no re-merge needed). Disable to save RAM — recommended for large video models like Wan or LTX."
                }),
                "compress_patches": (["non_ties", "all", "disabled"], {
                    "default": "non_ties",
                    "tooltip": "Shrink the merged result to use less memory. 'non_ties' (recommended): compresses most layers with no quality loss. 'all': compresses everything, slightly lossy but saves the most memory. 'disabled': no compression, uses more RAM. Enable for large models."
                }),
                "svd_device": (["gpu", "cpu"], {
                    "default": "gpu",
                    "tooltip": "Where to run compression math. GPU is much faster (10-50x). Switch to CPU only if you get out-of-memory errors during the merge."
                }),
                "normalize_keys": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Makes LoRAs from different training tools compatible with each other. Enable if your LoRAs were trained with different software (e.g. Kohya vs AI-Toolkit vs PEFT) or if merging fails without it."
                }),
                "sparsification": (["disabled", "dare", "della", "dare_conflict", "della_conflict"], {
                    "default": "disabled",
                    "tooltip": "Reduces interference between LoRAs by sparsifying weights before merging. "
                               "DARE: random dropout everywhere. DELLA: magnitude-aware dropout everywhere. "
                               "Conflict variants (recommended): same algorithms but ONLY applied where LoRAs "
                               "push in opposite directions — unique contributions are preserved untouched."
                }),
                "sparsification_density": ("FLOAT", {
                    "default": 0.7, "min": 0.01, "max": 1.0, "step": 0.05,
                    "tooltip": "What percentage of weights to keep (0.7 = keep 70%, drop 30%). Lower values drop more weights — reduces interference but may lose detail. "
                               "At 1.0, no weights are dropped (equivalent to disabled). "
                               "Note: in TIES mode, sparsification replaces the trim step — setting density to 1.0 disables both sparsification AND trimming."
                }),
                "merge_strategy_override": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Connect the merge_strategy output from a LoRA Conflict Editor to override the optimizer's auto-detected strategy."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "LORA_DATA")
    RETURN_NAMES = ("model", "clip", "analysis_report", "lora_data")
    FUNCTION = "optimize_merge"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Auto-analyzes LoRA stack and selects optimal merge strategy per weight group. Outputs merged model + analysis report. Best for style/character LoRAs — apply edit, distillation (LCM/Turbo/Hyper), or DPO LoRAs via a standard Load LoRA node instead."

    @staticmethod
    def _compute_cache_key(lora_stack, output_strength, clip_strength_multiplier, auto_strength, optimization_mode="per_prefix", compress_patches="non_ties", svd_device="gpu", normalize_keys="disabled", sparsification="disabled", sparsification_density=0.7, merge_strategy_override=""):
        """
        Build a deterministic SHA-256 hash (16 hex chars) from the stack
        configuration. Used by IS_CHANGED to let ComfyUI skip re-execution
        when nothing changed.
        """
        h = hashlib.sha256()
        if lora_stack:
            first = lora_stack[0] if len(lora_stack) > 0 else None
            entries = []
            if isinstance(first, (tuple, list)):
                for entry in lora_stack:
                    cm = entry[3] if len(entry) > 3 else "all"
                    entries.append((str(entry[0]), float(entry[1]), float(entry[2]), cm))
            elif isinstance(first, dict):
                for item in lora_stack:
                    cm = item.get("conflict_mode", "all")
                    entries.append((str(item.get("name", "")), float(item.get("strength", 0)), cm))
            entries.sort()
            h.update(json.dumps(entries).encode())
        h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}|om={optimization_mode}|cp={compress_patches}|sd={svd_device}|nk={normalize_keys}|sp={sparsification}|spd={sparsification_density}|mso={merge_strategy_override}".encode())
        return h.hexdigest()[:16]

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, clip=None,
                   clip_strength_multiplier=1.0, auto_strength="disabled",
                   free_vram_between_passes="disabled", optimization_mode="per_prefix",
                   cache_patches="enabled", compress_patches="non_ties",
                   svd_device="gpu", normalize_keys="disabled",
                   sparsification="disabled", sparsification_density=0.7,
                   merge_strategy_override=""):
        return cls._compute_cache_key(lora_stack, output_strength,
                                      clip_strength_multiplier, auto_strength,
                                      optimization_mode, compress_patches,
                                      svd_device, normalize_keys,
                                      sparsification, sparsification_density,
                                      merge_strategy_override)

    def _save_report_to_disk(self, cache_key, lora_combo, auto_strength, report, selected_params):
        """
        Persist the analysis report as JSON for later reference.
        Saved to {user_dir}/lora_optimizer_reports/{cache_key}.json.
        Failures are silently logged — never blocks the merge.
        """
        try:
            user_dir = folder_paths.get_user_directory()
            report_dir = os.path.join(user_dir, "lora_optimizer_reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"{cache_key}.json")
            data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "lora_combo": lora_combo,
                "auto_strength": auto_strength,
                "report": report,
                "selected_params": selected_params,
            }
            with open(report_path, "w") as f:
                json.dump(data, f, indent=2)
            return report_path
        except Exception as e:
            logging.warning(f"[LoRA Optimizer] Failed to save report: {e}")
            return None

    def _sample_conflict(self, diff_a, diff_b, device=None):
        """
        Compute sign conflict ratio and cosine similarity components between
        two diff tensors. Samples up to 100k positions for large tensors.
        Returns (n_overlap, n_conflict, dot, norm_a_sq, norm_b_sq).
        """
        # Avoid unnecessary .float() copies — diffs from _analyze_prefix are already float32
        flat_a = diff_a.flatten()
        flat_b = diff_b.flatten()
        if device is not None:
            flat_a = flat_a.to(device=device, dtype=torch.float32)
            flat_b = flat_b.to(device=device, dtype=torch.float32)
        elif flat_a.dtype != torch.float32:
            flat_a = flat_a.float()
            flat_b = flat_b.float()

        if flat_a.numel() != flat_b.numel():
            return (0, 0, 0.0, 0.0, 0.0)

        n = flat_a.numel()
        if n > 100000:
            target_device = flat_a.device
            g = torch.Generator(device=target_device).manual_seed(42)
            indices = torch.randperm(n, device=target_device, generator=g)[:100000]
            flat_a = flat_a[indices]
            flat_b = flat_b[indices]

        # Only consider positions where both are non-zero
        mask = (flat_a != 0) & (flat_b != 0)
        n_overlap = mask.sum().item()
        if n_overlap == 0:
            return (0, 0, 0.0, 0.0, 0.0)

        a_overlap = flat_a[mask]
        b_overlap = flat_b[mask]
        n_conflict = (a_overlap.sign() != b_overlap.sign()).sum().item()

        # Cosine similarity components (on overlap region)
        dot = (a_overlap * b_overlap).sum().item()
        norm_a_sq = (a_overlap * a_overlap).sum().item()
        norm_b_sq = (b_overlap * b_overlap).sum().item()

        return (n_overlap, n_conflict, dot, norm_a_sq, norm_b_sq)

    def _process_prefix(self, lora_prefix, active_loras, model_keys, clip_keys,
                        model, clip, device):
        """
        Process a single LoRA key prefix: resolve target key, compute diffs for
        each active LoRA against the target shape. Thread-safe — all writes go
        to local variables, reads from shared immutable data.

        Returns (lora_prefix, diffs_for_key, lora_diffs_for_key, partial_stats,
                 target_info, skip_count) or None if prefix should be skipped.
        partial_stats is a list of (lora_index, rank, l2_norm) tuples.
        """
        target_key = None
        is_clip = False

        if lora_prefix in model_keys:
            target_key = model_keys[lora_prefix]
        elif lora_prefix in clip_keys:
            target_key = clip_keys[lora_prefix]
            is_clip = True

        if target_key is None:
            return None

        # Handle tuple keys (for sliced weights, e.g. Flux linear1_qkv)
        offset = None
        if isinstance(target_key, tuple):
            actual_key = target_key[0]
            if len(target_key) > 1:
                offset = target_key[1]
        else:
            actual_key = target_key

        # Get target weight shape (use sliced shape when offset present)
        try:
            if is_clip:
                target_weight = comfy.utils.get_attr(clip.cond_stage_model, actual_key)
            else:
                target_weight = comfy.utils.get_attr(model.model, actual_key)
            target_shape = list(target_weight.shape)
            if offset is not None:
                target_shape[offset[0]] = offset[2]
            target_shape = torch.Size(target_shape)
        except (AttributeError, RuntimeError, IndexError):
            return None

        diffs_for_key = []
        lora_diffs_for_key = {}
        partial_stats = []
        skip_count = 0
        for i, item in enumerate(active_loras):
            lora_info = self._get_lora_key_info(item["lora"], lora_prefix)
            if lora_info is None:
                continue

            mat_up, mat_down, alpha, mid = lora_info
            rank = mat_down.shape[0]
            diff = self._compute_lora_diff(mat_up, mat_down, alpha, mid, target_shape, device=device)

            if diff is not None:
                # For CLIP keys, use per-LoRA clip_strength when available
                if is_clip and item["clip_strength"] is not None:
                    eff_strength = item["clip_strength"]
                else:
                    eff_strength = item["strength"]
                diffs_for_key.append((diff, eff_strength, i))
                # Store sign-corrected diff for conflict analysis;
                # negative strength inverts the LoRA's direction
                lora_diffs_for_key[i] = diff if eff_strength >= 0 else -diff
                # Always use model strength for l2_norms so _compute_auto_strengths
                # can correctly undo the weighting (it divides by model strength)
                l2 = diff.float().norm().item() * abs(item["strength"])
                partial_stats.append((i, rank, l2))
            else:
                skip_count += 1

        if len(diffs_for_key) == 0:
            if skip_count > 0:
                return (lora_prefix, [], {}, [], (target_key, is_clip), skip_count)
            return None

        return (lora_prefix, diffs_for_key, lora_diffs_for_key, partial_stats,
                (target_key, is_clip), skip_count)

    @torch.no_grad()
    def _analyze_prefix(self, lora_prefix, active_loras, model_keys, clip_keys,
                        model, clip, device, n_magnitude_samples=1000):
        """
        Pass 1 per-prefix analysis: compute diffs, sample conflicts and
        magnitudes, then discard diffs. Returns lightweight scalars/samples
        so the caller never accumulates full-rank diff tensors.

        All GPU work (diff computation, conflict sampling, magnitude sampling)
        stays on GPU until the final small results are copied to CPU, avoiding
        the GPU→CPU→GPU bounce that kills pipelining.

        Returns (lora_prefix, partial_stats, pair_conflicts, magnitude_samples,
                 target_info, skip_count) or None if prefix should be skipped.

        partial_stats: list of (lora_index, rank, l2_norm)
        pair_conflicts: dict mapping (i, j) -> (overlap, conflict, dot, norm_a_sq, norm_b_sq)
        magnitude_samples: list of small 1D CPU float tensors
        """
        # --- Resolve target key and shape (same as _process_prefix) ---
        target_key = None
        is_clip = False

        if lora_prefix in model_keys:
            target_key = model_keys[lora_prefix]
        elif lora_prefix in clip_keys:
            target_key = clip_keys[lora_prefix]
            is_clip = True

        if target_key is None:
            return None

        offset = None
        if isinstance(target_key, tuple):
            actual_key = target_key[0]
            if len(target_key) > 1:
                offset = target_key[1]
        else:
            actual_key = target_key

        try:
            if is_clip:
                target_weight = comfy.utils.get_attr(clip.cond_stage_model, actual_key)
            else:
                target_weight = comfy.utils.get_attr(model.model, actual_key)
            target_shape = list(target_weight.shape)
            if offset is not None:
                target_shape[offset[0]] = offset[2]
            target_shape = torch.Size(target_shape)
        except (AttributeError, RuntimeError, IndexError):
            return None

        # --- Compute diffs for all active LoRAs ---
        # Keep diffs on GPU (device) for conflict + magnitude analysis.
        # _compute_lora_diff normally returns CPU when device=cuda, so we
        # call it with device=None and manually move matrices to GPU.
        use_gpu = device is not None and device.type != "cpu"
        diffs = {}  # lora_index -> diff tensor (on device)
        eff_strengths = {}  # lora_index -> effective strength
        partial_stats = []
        skip_count = 0

        for i, item in enumerate(active_loras):
            lora_info = self._get_lora_key_info(item["lora"], lora_prefix)
            if lora_info is None:
                continue

            mat_up, mat_down, alpha, mid = lora_info
            rank = mat_down.shape[0]

            # Compute diff on GPU but DON'T copy back to CPU yet
            if use_gpu:
                mat_up = mat_up.to(device)
                mat_down = mat_down.to(device)
                if mid is not None:
                    mid = mid.to(device)
            scale = alpha / rank

            if mid is not None:
                final_shape = [mat_down.shape[1], mat_down.shape[0], mid.shape[2], mid.shape[3]]
                mat_down = (
                    torch.mm(
                        mat_down.transpose(0, 1).flatten(start_dim=1).float(),
                        mid.transpose(0, 1).flatten(start_dim=1).float(),
                    )
                    .reshape(final_shape)
                    .transpose(0, 1)
                )

            diff = torch.mm(
                mat_up.flatten(start_dim=1).float(),
                mat_down.flatten(start_dim=1).float()
            )
            del mat_up, mat_down  # Free LoRA matrices from GPU
            try:
                diff = diff.reshape(target_shape)
            except RuntimeError:
                skip_count += 1
                continue

            diff = diff * scale

            if is_clip and item["clip_strength"] is not None:
                eff_strength = item["clip_strength"]
            else:
                eff_strength = item["strength"]
            diffs[i] = diff
            eff_strengths[i] = eff_strength
            l2 = diff.float().norm().item() * abs(item["strength"])
            partial_stats.append((i, rank, l2))

        if len(diffs) == 0:
            if skip_count > 0:
                return (lora_prefix, [], {}, [], (target_key, is_clip), skip_count)
            return None

        # --- Pairwise conflict analysis (sign-corrected, stays on device) ---
        pair_conflicts = {}
        lora_indices = sorted(diffs.keys())
        for ai in range(len(lora_indices)):
            for bi in range(ai + 1, len(lora_indices)):
                i, j = lora_indices[ai], lora_indices[bi]
                diff_i = diffs[i] if eff_strengths[i] >= 0 else -diffs[i]
                diff_j = diffs[j] if eff_strengths[j] >= 0 else -diffs[j]
                ov, conf, dot, na_sq, nb_sq = self._sample_conflict(diff_i, diff_j, device=device)
                pair_conflicts[(i, j)] = (ov, conf, dot, na_sq, nb_sq)

        # --- Magnitude sampling (sample on device, free each diff eagerly) ---
        magnitude_samples = []
        seed = hash(lora_prefix) & 0xFFFFFFFF
        sample_dev = diffs[lora_indices[0]].device
        mag_g = torch.Generator(device=sample_dev).manual_seed(seed)
        for i in lora_indices:
            flat = diffs[i].flatten().abs().float() * abs(eff_strengths[i])
            del diffs[i]  # Free this diff's GPU memory immediately
            n = flat.numel()
            if n > n_magnitude_samples:
                indices = torch.randint(0, n, (n_magnitude_samples,),
                                        device=sample_dev, generator=mag_g)
                flat = flat[indices]
            magnitude_samples.append(flat.cpu())
        return (lora_prefix, partial_stats, pair_conflicts, magnitude_samples,
                (target_key, is_clip), skip_count)

    def _estimate_density(self, all_key_diffs):
        """
        Estimate TIES density parameter from magnitude distribution.
        Uses fraction of values above 10% of the max magnitude as a
        sparsity proxy. Returns float in [0.1, 0.9].
        """
        samples = []
        max_samples_per_key = 1000
        g = torch.Generator().manual_seed(42)

        for key, diffs_list in all_key_diffs.items():
            for entry in diffs_list:
                diff, strength = entry[0], entry[1]
                flat = diff.flatten().abs().float().cpu() * abs(strength)
                n = flat.numel()
                if n > max_samples_per_key:
                    indices = torch.randperm(n, generator=g)[:max_samples_per_key]
                    flat = flat[indices]
                samples.append(flat)

        if len(samples) == 0:
            return 0.5

        all_samples = torch.cat(samples)
        if all_samples.numel() == 0:
            return 0.5

        max_val = all_samples.max().item()
        if max_val <= 0:
            return 0.5

        # Fraction of values above 10% of max magnitude
        noise_floor = max_val * 0.1
        above_noise = (all_samples > noise_floor).float().mean().item()

        return max(0.1, min(0.9, above_noise))

    def _estimate_density_from_samples(self, magnitude_samples):
        """
        Estimate TIES density from pre-sampled magnitude tensors.
        Takes a list of 1D CPU float tensors (from _analyze_prefix).
        Returns float in [0.1, 0.9].
        """
        if len(magnitude_samples) == 0:
            return 0.5

        all_samples = torch.cat(magnitude_samples)
        if all_samples.numel() == 0:
            return 0.5

        max_val = all_samples.max().item()
        if max_val <= 0:
            return 0.5

        noise_floor = max_val * 0.1
        above_noise = (all_samples > noise_floor).float().mean().item()

        return max(0.1, min(0.9, above_noise))

    def _compute_auto_strengths(self, active_loras, lora_stats, pairwise_similarities=None):
        """
        Compute reduced per-LoRA strengths using interference-aware energy
        normalization. Uses pairwise cosine similarity to account for
        directional alignment between LoRAs:
        - Aligned LoRAs (cos≈1) → more aggressive scaling (they reinforce)
        - Orthogonal LoRAs (cos≈0) → same as classic L2 norm
        - Opposing LoRAs (cos≈-1) → less scaling (they cancel out)

        Returns (new_strengths, reasoning_lines) where new_strengths is a list
        of floats (one per active LoRA) and reasoning_lines is a list of strings.
        """
        n = len(active_loras)
        original_strengths = [item["strength"] for item in active_loras]
        reasoning = []

        # Compute raw (un-strength-weighted) L2 norms per LoRA
        raw_l2 = []
        for i, stat in enumerate(lora_stats):
            s = abs(original_strengths[i])
            l2 = stat["l2_mean"]
            if s > 0 and l2 > 0:
                raw_l2.append(l2 / s)
            else:
                raw_l2.append(0.0)

        # Effective contribution: abs(strength) * raw_l2 = l2_mean (the roundtrip
        # cancels out), but keeping the decomposition clarifies intent: raw_l2 is
        # intrinsic LoRA magnitude, strength is the user-set scaling.
        effective = [abs(original_strengths[i]) * raw_l2[i] for i in range(n)]

        # Filter out zero contributions
        nonzero = [e for e in effective if e > 0]
        if len(nonzero) <= 1:
            # 0 or 1 contributing LoRAs: no reduction needed
            reasoning.append("Single contributing LoRA or none — no strength adjustment needed")
            return (list(original_strengths), reasoning)

        # Interference-aware combined energy using the vector-sum formula:
        #   ||sum(v_i)||^2 = sum(||v_i||^2) + 2 * sum_{i<j}(||v_i|| * ||v_j|| * cos(v_i, v_j))
        # This is the exact squared magnitude of the sum of vectors given their
        # pairwise angles. When cos=0 (orthogonal), reduces to sum of squares.
        energy_sq = sum(e * e for e in effective)
        orthogonal_energy_sq = energy_sq  # save for reporting

        if pairwise_similarities:
            for (i, j), cos_sim in pairwise_similarities.items():
                if effective[i] > 0 and effective[j] > 0:
                    energy_sq += 2.0 * effective[i] * effective[j] * cos_sim

        # Clamp to avoid sqrt of negative (possible if strong opposing LoRAs)
        energy_sq = max(energy_sq, 0.0)
        current_energy = math.sqrt(energy_sq)

        # Reference energy: what the strongest single LoRA contributes alone
        reference_energy = max(effective)

        # Scale factor
        if current_energy > 0:
            scale = reference_energy / current_energy
        else:
            scale = 1.0

        # Apply scale to all strengths
        new_strengths = []
        for i in range(n):
            if effective[i] > 0:
                new_strengths.append(original_strengths[i] * scale)
            else:
                new_strengths.append(original_strengths[i])

        # Build reasoning
        reasoning.append(f"Scale factor: {scale:.4f}")
        if pairwise_similarities:
            avg_cos = sum(pairwise_similarities.values()) / len(pairwise_similarities)
            orthogonal_energy = math.sqrt(orthogonal_energy_sq)
            if avg_cos > 0.1:
                alignment_desc = "mostly aligned (reinforcing)"
            elif avg_cos < -0.1:
                alignment_desc = "mostly opposing (cancelling)"
            else:
                alignment_desc = "mostly orthogonal (independent)"
            reasoning.append(f"Method: interference-aware energy normalization")
            reasoning.append(f"  Avg pairwise cosine similarity: {avg_cos:.3f} ({alignment_desc})")
            reasoning.append(f"  Interference-aware energy: {current_energy:.4f} (orthogonal assumption: {orthogonal_energy:.4f})")
        else:
            reasoning.append("Method: L2-aware energy normalization (no similarity data)")
        for i in range(n):
            if effective[i] > 0 and abs(scale - 1.0) > 1e-9:
                reasoning.append(f"  {active_loras[i]['name']}: {original_strengths[i]} -> {new_strengths[i]:.4f}")
            else:
                reasoning.append(f"  {active_loras[i]['name']}: {original_strengths[i]} (unchanged)")

        return (new_strengths, reasoning)

    @staticmethod
    def _extract_block_name(prefix):
        """
        Extract a human-readable block name from a LoRA key prefix.
        Handles common architectures: SD1.5, SDXL, Flux, Wan, etc.

        Examples:
          lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_q -> input_blocks.4
          lora_unet_double_blocks_12_img_attn_proj -> double_blocks.12
          lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k -> down_blocks.2
          diffusion_model.joint_blocks.5.x_block.attn.qkv -> joint_blocks.5
          transformer.blocks.8.attn1.to_q -> blocks.8

        Falls back to the first two meaningful segments if no pattern matches.
        """
        import re
        # Normalize separators: lora_unet_input_blocks_4 -> input_blocks.4
        # Strip common prefixes
        p = prefix
        for strip in ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_",
                       "diffusion_model.", "transformer.", "model."]:
            if p.startswith(strip):
                p = p[len(strip):]
                break

        # Replace underscores with dots for pattern matching
        p_dots = re.sub(r'_', '.', p)

        # Match: word.number (e.g., input.blocks.4, double.blocks.12, down.blocks.2)
        m = re.match(r'([a-z]+(?:\.[a-z]+)*?)\.(\d+)', p_dots)
        if m:
            block_type = m.group(1).replace('.', '_')
            block_num = m.group(2)
            return f"{block_type}.{block_num}"

        # Fallback: first segment
        parts = re.split(r'[._]', prefix)
        meaningful = [p for p in parts if p not in ("lora", "unet", "te", "te1", "te2",
                                                     "diffusion", "model", "transformer")]
        if len(meaningful) >= 2:
            return f"{meaningful[0]}.{meaningful[1]}"
        elif meaningful:
            return meaningful[0]
        return prefix[:30]

    def _auto_select_params(self, avg_conflict_ratio, magnitude_ratio, all_key_diffs=None, magnitude_samples=None):
        """
        Decision logic for auto-selecting merge parameters.
        Returns (mode, density, sign_method, reasoning_lines).

        Density can be estimated from either all_key_diffs (legacy bulk path)
        or magnitude_samples (streaming path).
        """
        reasoning = []

        # Select mode based on sign conflict
        if avg_conflict_ratio > 0.25:
            mode = "ties"
            reasoning.append(f"Sign conflict ratio {avg_conflict_ratio:.1%} > 25% threshold -> TIES mode selected")
            reasoning.append("  TIES resolves sign conflicts via trim + elect sign + disjoint merge")
        else:
            mode = "weighted_average"
            reasoning.append(f"Sign conflict ratio {avg_conflict_ratio:.1%} <= 25% threshold -> weighted_average mode selected")
            reasoning.append("  Low conflict means LoRAs are mostly compatible, simple averaging works well")

        # Auto-density (TIES only)
        if mode == "ties":
            if magnitude_samples is not None:
                density = self._estimate_density_from_samples(magnitude_samples)
            else:
                density = self._estimate_density(all_key_diffs)
            reasoning.append(f"Auto-density estimated at {density:.2f} from magnitude distribution")
        else:
            density = 0.5  # unused but set for completeness

        # Sign method (only relevant for TIES mode)
        if mode == "ties":
            if magnitude_ratio > 2.0:
                sign_method = "total"
                reasoning.append(f"Magnitude ratio {magnitude_ratio:.2f}x > 2x -> 'total' sign method (magnitude-weighted voting)")
                reasoning.append("  Stronger LoRA gets more influence in sign election")
            else:
                sign_method = "frequency"
                reasoning.append(f"Magnitude ratio {magnitude_ratio:.2f}x <= 2x -> 'frequency' sign method (equal voting)")
                reasoning.append("  Similar-strength LoRAs get equal votes")
        else:
            sign_method = "frequency"  # unused, default for completeness

        return (mode, density, sign_method, reasoning)

    def _build_report(self, lora_stats, pairwise_conflicts, collection_stats,
                      mode, density, sign_method, reasoning, merge_summary,
                      auto_strength_info=None, strategy_counts=None, optimization_mode="global",
                      prefix_decisions=None, detected_arch=None, normalize_keys="disabled",
                      sparsification="disabled", sparsification_density=0.7):
        """Format analysis as a multi-line report string."""
        lines = []
        lines.append("=" * 50)
        lines.append("LORA OPTIMIZER - ANALYSIS REPORT")
        lines.append("=" * 50)

        # Architecture info (when normalization is enabled)
        if normalize_keys == "enabled" and detected_arch:
            arch_names = {
                'zimage': 'Z-Image Turbo (Lumina2)',
                'flux': 'FLUX',
                'wan': 'Wan 2.1/2.2',
                'sdxl': 'SDXL',
                'ltx': 'LTX Video',
                'qwen_image': 'Qwen-Image',
            }
            lines.append(f"Architecture: {arch_names.get(detected_arch, detected_arch)} (auto-detected)")
            lines.append(f"Key normalization: enabled")
            lines.append("")

        # Per-LoRA Analysis
        lines.append("")
        lines.append("--- Per-LoRA Analysis ---")
        for stat in lora_stats:
            lines.append(f"  {stat['name']}:")
            lines.append(f"    Strength: {stat.get('original_strength', stat['strength'])}")
            lines.append(f"    Keys: {stat['key_count']}")
            if stat['key_count'] > 0:
                lines.append(f"    Avg rank: {stat['avg_rank']:.0f}")
                lines.append(f"    L2 norm (mean): {stat['l2_mean']:.4f}")
            else:
                lines.append(f"    Avg rank: N/A (no compatible keys)")
                lines.append(f"    L2 norm (mean): N/A")
            if stat.get("conflict_mode", "all") != "all":
                lines.append(f"    Conflict mode: {stat['conflict_mode']}")

        # Auto-Strength Adjustment (between Per-LoRA and Pairwise)
        if auto_strength_info is not None:
            lines.append("")
            lines.append("--- Auto-Strength Adjustment ---")
            for i, name in enumerate(auto_strength_info["names"]):
                orig = auto_strength_info["original_strengths"][i]
                new = auto_strength_info["new_strengths"][i]
                lines.append(f"  {name}: {orig} -> {new:.4f}")
            for r in auto_strength_info["reasoning"]:
                lines.append(f"  {r}")

        # Pairwise Analysis
        if pairwise_conflicts:
            lines.append("")
            lines.append("--- Pairwise Analysis ---")
            for pc in pairwise_conflicts:
                lines.append(f"  {pc['pair']}:")
                lines.append(f"    Overlapping positions: {pc['overlap']}")
                lines.append(f"    Sign conflicts: {pc['conflicts']} ({pc['ratio']:.1%})")
                if 'cosine_sim' in pc:
                    lines.append(f"    Cosine similarity: {pc['cosine_sim']:.3f}")

        # Collection Statistics
        lines.append("")
        lines.append("--- Collection Statistics ---")
        lines.append(f"  Total LoRAs: {collection_stats['n_loras']}")
        lines.append(f"  Total unique keys: {collection_stats['total_keys']}")
        lines.append(f"  Avg sign conflict ratio: {collection_stats['avg_conflict']:.1%}")
        lines.append(f"  Magnitude ratio (max/min L2): {collection_stats['magnitude_ratio']:.2f}x")

        # Auto-Selected Parameters
        lines.append("")
        lines.append("--- Auto-Selected Parameters ---")
        if optimization_mode == "weighted_sum_only":
            lines.append(f"  Merge mode: weighted_sum (forced by weighted_sum_only)")
            lines.append(f"  Auto-detected mode: {mode} (overridden)")
        else:
            lines.append(f"  Merge mode: {mode}")
        if mode == "ties":
            lines.append(f"  Density: {density:.2f}")
            lines.append(f"  Sign method: {sign_method}")
        if optimization_mode == "per_prefix":
            lines.append("  (global fallback — each prefix uses its own parameters)")
        if sparsification != "disabled":
            display_name = {
                "dare": "DARE", "della": "DELLA",
                "dare_conflict": "DARE (conflict-aware)",
                "della_conflict": "DELLA (conflict-aware)",
            }.get(sparsification, sparsification.upper())
            lines.append(f"  Sparsification: {display_name}")
            lines.append(f"  Sparsification density: {sparsification_density:.2f} (keep rate)")
            if optimization_mode == "per_prefix":
                lines.append(f"  For TIES prefixes: replaces trim step; others: preprocessing")
            elif optimization_mode == "weighted_sum_only":
                lines.append(f"  Applied as preprocessing before weighted_sum")
            elif mode == "ties":
                lines.append(f"  Note: {display_name} replaces TIES trim step")
            else:
                lines.append(f"  Applied as preprocessing before {mode}")

        # Per-Prefix Strategy breakdown (only in per_prefix mode)
        if optimization_mode == "per_prefix" and strategy_counts:
            lines.append("")
            lines.append("--- Per-Prefix Strategy ---")
            total_pf = sum(strategy_counts.values())
            if total_pf > 0:
                if strategy_counts.get("weighted_sum", 0) > 0:
                    n = strategy_counts["weighted_sum"]
                    lines.append(f"  weighted_sum (single LoRA):      {n:>4} prefixes ({n/total_pf:.0%})")
                if strategy_counts.get("slerp", 0) > 0:
                    n = strategy_counts["slerp"]
                    lines.append(f"  slerp (2 LoRAs, low conflict):   {n:>4} prefixes ({n/total_pf:.0%})")
                if strategy_counts.get("weighted_average", 0) > 0:
                    n = strategy_counts["weighted_average"]
                    lines.append(f"  weighted_average (low conflict):  {n:>4} prefixes ({n/total_pf:.0%})")
                if strategy_counts.get("ties", 0) > 0:
                    n = strategy_counts["ties"]
                    lines.append(f"  ties (high conflict):            {n:>4} prefixes ({n/total_pf:.0%})")
                lines.append(f"  Total:                           {total_pf:>4} prefixes")

        # Block Strategy Map (per_prefix mode only)
        if optimization_mode == "per_prefix" and prefix_decisions:
            # Group prefixes by block name
            block_data = {}  # block_name -> list of (mode, conflict, n_loras)
            for prefix, pf_mode, conflict, n_loras in prefix_decisions:
                block_name = self._extract_block_name(prefix)
                if block_name not in block_data:
                    block_data[block_name] = []
                block_data[block_name].append((pf_mode, conflict, n_loras))

            # Aggregate per block: dominant strategy, avg conflict, max n_loras
            # Priority-based dominant: ties > slerp > avg > sum (show most interesting)
            mode_priority = {"ties": 3, "slerp": 2, "weighted_average": 1, "weighted_sum": 0}
            block_summary = []
            for block_name, entries in block_data.items():
                modes = [e[0] for e in entries]
                conflicts = [e[1] for e in entries]
                n_loras_max = max(e[2] for e in entries)
                mode_counts = {}
                for m in modes:
                    mode_counts[m] = mode_counts.get(m, 0) + 1
                # Pick highest-priority mode present (not most frequent)
                dominant = max(mode_counts, key=lambda m: mode_priority.get(m, -1))
                # Avg conflict only over multi-LoRA prefixes (sum prefixes have 0 conflict)
                multi_conflicts = [c for m, c in zip(modes, conflicts) if m != "weighted_sum"]
                avg_conflict = sum(multi_conflicts) / len(multi_conflicts) if multi_conflicts else 0
                n_prefixes = len(entries)
                block_summary.append((block_name, dominant, avg_conflict, n_loras_max, n_prefixes, mode_counts))

            # Sort by block name for consistent ordering
            block_summary.sort(key=lambda x: x[0])

            lines.append("")
            lines.append("--- Block Strategy Map ---")
            symbols = {"weighted_sum": "====", "slerp": "~~~~", "weighted_average": "----", "ties": "####"}
            labels = {"weighted_sum": "sum", "slerp": "slrp", "weighted_average": "avg", "ties": "TIES"}
            # Find max block name length for alignment
            max_name = max(len(b[0]) for b in block_summary) if block_summary else 10
            for block_name, dominant, avg_conflict, n_loras_max, n_prefixes, mode_counts in block_summary:
                sym = symbols.get(dominant, "????")
                lbl = labels.get(dominant, dominant)
                if len(mode_counts) == 1 and dominant == "weighted_sum":
                    detail = "1 LoRA"
                else:
                    # Show breakdown when block has mixed strategies
                    parts = []
                    for m in ("weighted_sum", "weighted_average", "slerp", "ties"):
                        if mode_counts.get(m, 0) > 0:
                            parts.append(f"{mode_counts[m]} {labels.get(m, m)}")
                    detail = f"{avg_conflict:.0%} conflict ({', '.join(parts)})"
                count_str = f"({n_prefixes}x)" if n_prefixes > 1 else ""
                lines.append(f"  {block_name:<{max_name}}  {sym}  {lbl:<5} {detail} {count_str}")
            lines.append(f"  Legend: ==== sum  ~~~~ slerp  ---- avg  #### TIES")

        # Reasoning
        lines.append("")
        lines.append("--- Reasoning ---")
        for r in reasoning:
            lines.append(f"  {r}")

        # Merge Summary
        lines.append("")
        lines.append("--- Merge Summary ---")
        lines.append(f"  Keys processed: {merge_summary['keys_processed']}")
        lines.append(f"  Model patches: {merge_summary['model_patches']}")
        lines.append(f"  CLIP patches: {merge_summary['clip_patches']}")
        if merge_summary.get('skipped_keys', 0) > 0:
            lines.append(f"  Skipped keys: {merge_summary['skipped_keys']} (shape mismatch, e.g. sliced weights)")
        lines.append(f"  Output strength: {merge_summary['output_strength']}")
        lines.append(f"  CLIP strength: {merge_summary['clip_strength']}")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def optimize_merge(self, model, lora_stack, output_strength, clip=None, clip_strength_multiplier=1.0, auto_strength="disabled", free_vram_between_passes="disabled", optimization_mode="per_prefix", cache_patches="enabled", compress_patches="non_ties", svd_device="gpu", normalize_keys="disabled", sparsification="disabled", sparsification_density=0.7, merge_strategy_override=""):
        """
        Main entry point. Two-pass streaming architecture:
        Pass 1: Compute diffs per-prefix, sample conflicts + magnitudes, discard diffs
        Decision: Finalize stats, auto-select params from lightweight accumulators
        Pass 2: Recompute diffs per-prefix, merge immediately, discard
        Peak memory: ~260MB (one prefix's diffs at a time) vs ~50GB (all diffs).
        """
        # Normalize stack format (standard tuples or LoRAStack dicts)
        if not lora_stack or len(lora_stack) == 0:
            return (model, clip, "No LoRAs in stack.", None)

        normalized_stack = self._normalize_stack(lora_stack, normalize_keys=normalize_keys)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]

        if len(active_loras) == 0:
            return (model, clip, "No LoRAs in stack (all zero strength or malformed).", None)

        # Single LoRA: skip analysis, apply directly via ComfyUI's standard
        # additive LoRA application (faster than diff-based pipeline).
        # auto_strength is a no-op with a single LoRA (scale would be 1.0).
        # Skip fast path for Z-Image: normalized keys (to_q/to_k/to_v) won't
        # match the model's fused qkv keys — need full pipeline + re-fusion.
        if len(active_loras) == 1 and getattr(self, '_detected_arch', None) != 'zimage':
            item = active_loras[0]
            lora_dict = item["lora"]
            strength = item["strength"]

            if item["clip_strength"] is not None:
                clip_str = item["clip_strength"]
            else:
                clip_str = strength * clip_strength_multiplier
            new_model, new_clip = comfy.sd.load_lora_for_models(
                model, clip, lora_dict, output_strength * strength, output_strength * clip_str
            )

            report = (
                "=" * 50 + "\n"
                "LORA OPTIMIZER - ANALYSIS REPORT\n"
                "=" * 50 + "\n\n"
                "Single LoRA detected — bypassing analysis.\n"
                f"  Name: {item['name']}\n"
                f"  Strength: {strength}\n"
                f"  Applied directly with output_strength={output_strength}\n"
                "\n" + "=" * 50
            )
            return (new_model, new_clip, report, None)

        # Check instance-level patch cache (survives ComfyUI re-execution
        # triggered by downstream seed changes or similar non-LoRA changes)
        cache_key = self._compute_cache_key(lora_stack, output_strength,
                                            clip_strength_multiplier, auto_strength,
                                            optimization_mode, compress_patches,
                                            svd_device, normalize_keys,
                                            sparsification, sparsification_density,
                                            merge_strategy_override)
        if cache_patches == "enabled" and cache_key in self._merge_cache:
            model_patches, clip_patches, report, clip_strength_out, lora_data = self._merge_cache[cache_key]
            new_model = model
            new_clip = clip
            if model is not None and len(model_patches) > 0:
                new_model = model.clone()
                new_model.add_patches(model_patches, output_strength)
            if clip is not None and len(clip_patches) > 0:
                new_clip = clip.clone()
                new_clip.add_patches(clip_patches, clip_strength_out)
            logging.info(f"[LoRA Optimizer] Using cached merge result ({len(model_patches)} model + {len(clip_patches)} CLIP patches)")
            return (new_model, new_clip, report, lora_data)

        logging.info(f"[LoRA Optimizer] Starting analysis of {len(active_loras)} LoRAs")
        t_start = time.time()

        # Get key maps
        model_keys = {}
        if model is not None:
            model_keys = comfy.lora.model_lora_keys_unet(model.model, {})

        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})

        # Collect all LoRA key prefixes
        all_lora_prefixes = set()
        for item in active_loras:
            for key in item["lora"].keys():
                for suffix in [".lora_up.weight", ".lora_down.weight", "_lora.up.weight",
                              "_lora.down.weight", ".lora_B.weight", ".lora_A.weight",
                              ".lora.up.weight", ".lora.down.weight", ".alpha"]:
                    if key.endswith(suffix):
                        prefix = key[:-len(suffix)]
                        all_lora_prefixes.add(prefix)
                        break

        compute_device = self._get_compute_device()
        use_gpu = compute_device.type != "cpu"

        # =====================================================================
        # Pass 1 — Analysis (streaming: diffs computed, sampled, and discarded)
        # =====================================================================
        logging.info("[LoRA Optimizer] Pass 1: Analyzing weight diffs (streaming)...")
        logging.info(f"[LoRA Optimizer]   {len(all_lora_prefixes)} key prefixes across {len(active_loras)} LoRAs")
        logging.info(f"[LoRA Optimizer]   Compute device: {compute_device}"
                     f" ({'sequential' if use_gpu else 'threaded'})")
        t_pass1 = time.time()

        # Lightweight accumulators (no full-rank diff tensors stored)
        all_key_targets = {}          # prefix -> (target_key, is_clip)
        skipped_keys = 0
        per_lora_stats = [{
            "name": item["name"],
            "strength": item["strength"],
            "ranks": [],
            "key_count": 0,
            "l2_norms": [],
        } for item in active_loras]

        pairs = [(i, j) for i in range(len(active_loras))
                         for j in range(i + 1, len(active_loras))]
        pair_accum = {(i, j): [0, 0, 0.0, 0.0, 0.0] for i, j in pairs}  # [overlap, conflict, dot, norm_a_sq, norm_b_sq]
        all_magnitude_samples = []    # list of small CPU tensors
        prefix_count = 0              # number of prefixes with valid diffs
        prefix_stats = {}             # prefix -> {conflict_ratio, n_loras, magnitude_samples, magnitude_ratio}

        def _collect_analysis_result(result):
            nonlocal skipped_keys, prefix_count
            if result is None:
                return
            prefix, partial_stats, pair_conflicts, mag_samples, target_info, skips = result
            skipped_keys += skips
            if len(partial_stats) > 0:
                all_key_targets[prefix] = target_info
                prefix_count += 1
            for (idx, rank, l2) in partial_stats:
                per_lora_stats[idx]["ranks"].append(rank)
                per_lora_stats[idx]["key_count"] += 1
                per_lora_stats[idx]["l2_norms"].append(l2)
            for (i, j), (ov, conf, dot, na_sq, nb_sq) in pair_conflicts.items():
                pair_accum[(i, j)][0] += ov
                pair_accum[(i, j)][1] += conf
                pair_accum[(i, j)][2] += dot
                pair_accum[(i, j)][3] += na_sq
                pair_accum[(i, j)][4] += nb_sq
            all_magnitude_samples.extend(mag_samples)

            # Store per-prefix stats for per_prefix optimization mode
            if len(partial_stats) > 0:
                # Number of LoRAs contributing to this prefix
                n_contributing = len(partial_stats)

                # Per-prefix conflict ratio
                pf_overlap = sum(ov for ov, conf, dot, na_sq, nb_sq in pair_conflicts.values())
                pf_conflict = sum(conf for ov, conf, dot, na_sq, nb_sq in pair_conflicts.values())
                pf_conflict_ratio = pf_conflict / pf_overlap if pf_overlap > 0 else 0.0

                # Per-prefix magnitude ratio (max/min L2 among contributing LoRAs)
                pf_l2s = [l2 for _, _, l2 in partial_stats if l2 > 0]
                if len(pf_l2s) >= 2:
                    pf_mag_ratio = max(pf_l2s) / min(pf_l2s)
                else:
                    pf_mag_ratio = 1.0

                prefix_stats[prefix] = {
                    "n_loras": n_contributing,
                    "conflict_ratio": pf_conflict_ratio,
                    "magnitude_ratio": pf_mag_ratio,
                    "magnitude_samples": list(mag_samples),  # copy, not reference
                }

        if use_gpu:
            for lora_prefix in all_lora_prefixes:
                result = self._analyze_prefix(lora_prefix, active_loras,
                                              model_keys, clip_keys, model, clip, compute_device)
                _collect_analysis_result(result)
        else:
            max_workers = min(4, max(1, len(all_lora_prefixes)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._analyze_prefix, lora_prefix, active_loras,
                                    model_keys, clip_keys, model, clip, compute_device): lora_prefix
                    for lora_prefix in all_lora_prefixes
                }
                for future in concurrent.futures.as_completed(futures):
                    _collect_analysis_result(future.result())

        if prefix_count == 0:
            return (model, clip, "No compatible LoRA keys found. "
                    "LoRAs may be incompatible with this model architecture.", None)

        # Log per-LoRA summaries
        for i, stat in enumerate(per_lora_stats):
            avg_r = sum(stat["ranks"]) / len(stat["ranks"]) if stat["ranks"] else 0
            logging.info(f"[LoRA Optimizer]   {stat['name']} ({i+1}/{len(active_loras)}): "
                         f"{stat['key_count']} keys, avg rank {avg_r:.0f}")
        logging.info(f"[LoRA Optimizer]   Total: {prefix_count} prefixes ({time.time() - t_pass1:.1f}s)")

        # =====================================================================
        # Decision — finalize stats, auto-select params (scalars only)
        # =====================================================================

        # Finalize per-LoRA stats
        lora_stats = []
        l2_means = []
        for i, stat in enumerate(per_lora_stats):
            avg_rank = sum(stat["ranks"]) / len(stat["ranks"]) if stat["ranks"] else 0
            l2_mean = sum(stat["l2_norms"]) / len(stat["l2_norms"]) if stat["l2_norms"] else 0
            l2_means.append(l2_mean)
            lora_stats.append({
                "name": stat["name"],
                "strength": stat["strength"],
                "key_count": stat["key_count"],
                "avg_rank": avg_rank,
                "l2_mean": l2_mean,
                "conflict_mode": active_loras[i].get("conflict_mode", "all"),
            })

        # Pairwise conflict stats and cosine similarity from accumulated counts
        total_overlap = 0
        total_conflict = 0
        pairwise_conflicts = []
        pairwise_similarities = {}
        for i, j in pairs:
            pair_overlap, pair_conflict, pair_dot, pair_na_sq, pair_nb_sq = pair_accum[(i, j)]
            total_overlap += pair_overlap
            total_conflict += pair_conflict
            ratio = pair_conflict / pair_overlap if pair_overlap > 0 else 0
            denom = math.sqrt(pair_na_sq) * math.sqrt(pair_nb_sq)
            cos_sim = pair_dot / denom if denom > 0 else 0.0
            pairwise_similarities[(i, j)] = cos_sim
            name_i = active_loras[i]['name']
            name_j = active_loras[j]['name']
            if name_i == name_j:
                pair_label = f"{name_i} [#{i+1}, str={active_loras[i]['strength']}] vs {name_j} [#{j+1}, str={active_loras[j]['strength']}]"
            else:
                pair_label = f"{name_i} vs {name_j}"
            pairwise_conflicts.append({
                "pair": pair_label,
                "overlap": pair_overlap,
                "conflicts": pair_conflict,
                "ratio": ratio,
                "cosine_sim": cos_sim,
            })
            logging.info(f"[LoRA Optimizer]   {pair_label} -> {ratio:.1%} conflict, cos_sim={cos_sim:.3f}")

        avg_conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0
        logging.info(f"[LoRA Optimizer]   Average conflict ratio: {avg_conflict_ratio:.1%}")

        # Magnitude ratio
        valid_l2 = [m for m in l2_means if m > 0]
        if len(valid_l2) >= 2:
            magnitude_ratio = max(valid_l2) / min(valid_l2)
        else:
            magnitude_ratio = 1.0

        collection_stats = {
            "n_loras": len(active_loras),
            "total_keys": prefix_count,
            "avg_conflict": avg_conflict_ratio,
            "magnitude_ratio": magnitude_ratio,
        }

        # Auto-select parameters (density estimated from pre-sampled magnitudes)
        mode, density, sign_method, reasoning = self._auto_select_params(
            avg_conflict_ratio, magnitude_ratio, magnitude_samples=all_magnitude_samples
        )
        del all_magnitude_samples

        # Apply merge strategy override from Conflict Editor
        # Skip when user explicitly chose weighted_sum_only (protects DPO/edit LoRAs)
        if merge_strategy_override and optimization_mode != "weighted_sum_only":
            if merge_strategy_override in ("ties", "weighted_average", "weighted_sum"):
                mode = merge_strategy_override
                reasoning.append(f"Merge mode overridden to '{mode}' by Conflict Editor")
            else:
                logging.warning(f"[LoRA Optimizer] Invalid merge_strategy_override '{merge_strategy_override}' — ignoring")

        logging.info(f"[LoRA Optimizer] Decision: {mode} (conflict {avg_conflict_ratio:.1%} "
                     f"{'>' if avg_conflict_ratio > 0.25 else '<='} 25% threshold)")
        if mode == "ties":
            logging.info(f"[LoRA Optimizer]   density={density:.2f}, sign_method={sign_method}")

        # Auto-strength adjustment
        auto_strength_info = None
        scale_ratios = {}
        if auto_strength == "enabled":
            new_strengths, strength_reasoning = self._compute_auto_strengths(
                active_loras, lora_stats, pairwise_similarities=pairwise_similarities)

            for i in range(len(active_loras)):
                orig = active_loras[i]["strength"]
                if abs(orig) > 1e-9:
                    scale_ratios[i] = new_strengths[i] / orig
                else:
                    scale_ratios[i] = 1.0

            for i, stat in enumerate(lora_stats):
                stat["original_strength"] = stat["strength"]
                stat["strength"] = new_strengths[i]

            auto_strength_info = {
                "reasoning": strength_reasoning,
                "original_strengths": [item["strength"] for item in active_loras],
                "new_strengths": new_strengths,
                "names": [item["name"] for item in active_loras],
            }

            logging.info(f"[LoRA Optimizer] Auto-strength: {strength_reasoning[0]}")
            for i in range(len(active_loras)):
                logging.info(f"[LoRA Optimizer]   {active_loras[i]['name']}: "
                             f"{active_loras[i]['strength']} -> {new_strengths[i]:.4f}")

        # Free GPU cache between passes if requested
        if free_vram_between_passes == "enabled" and use_gpu:
            torch.cuda.empty_cache()

        # Resolve compress_patches rank (sum of input LoRA ranks)
        compress_rank = 0  # 0 = disabled
        if compress_patches in ("non_ties", "all"):
            sum_rank = sum(int(stat["avg_rank"]) for stat in lora_stats if stat["avg_rank"] > 0)
            compress_rank = max(sum_rank, 64)  # floor at 64
            logging.info(f"[LoRA Optimizer] Patch compression: {compress_patches} (rank {compress_rank} from sum of input LoRA ranks)")

        # Resolve SVD device for compression
        resolved_svd_device = None
        if compress_rank > 0 and svd_device == "gpu" and torch.cuda.is_available():
            resolved_svd_device = torch.device("cuda")
        elif compress_rank > 0 and svd_device == "cpu":
            resolved_svd_device = None  # CPU is the default in _compress_to_lowrank

        # =====================================================================
        # Pass 2 — Merge (recompute diffs per-prefix, merge, discard)
        # =====================================================================
        logging.info(f"[LoRA Optimizer] Pass 2: Merging {len(all_key_targets)} keys "
                     f"({optimization_mode} strategy, "
                     f"{'sequential' if use_gpu else 'threaded'})...")
        t_pass2 = time.time()
        model_patches = {}
        clip_patches = {}
        processed_keys = 0
        compressed_count = 0
        strategy_counts = {"weighted_sum": 0, "weighted_average": 0, "slerp": 0, "ties": 0}
        prefix_decisions = []  # list of (prefix, mode, conflict_ratio, n_loras) for block map

        def _merge_one_prefix(lora_prefix, target_key, is_clip_key):
            """Recompute diffs for one prefix, merge, return patch or None."""
            offset = None
            if isinstance(target_key, tuple):
                actual_key = target_key[0]
                if len(target_key) > 1:
                    offset = target_key[1]
            else:
                actual_key = target_key

            try:
                if is_clip_key:
                    target_weight = comfy.utils.get_attr(clip.cond_stage_model, actual_key)
                else:
                    target_weight = comfy.utils.get_attr(model.model, actual_key)
                target_shape = list(target_weight.shape)
                if offset is not None:
                    target_shape[offset[0]] = offset[2]
                target_shape = torch.Size(target_shape)
            except (AttributeError, RuntimeError, IndexError):
                return None

            # Determine strategy BEFORE computing diffs (use Pass 1 stats)
            pf_conflict = 0.0
            pf_n_loras = 0
            pf_mode = mode
            pf_density = density
            pf_sign = sign_method
            if optimization_mode == "weighted_sum_only":
                pf_mode = "weighted_sum"
                pf_n_loras = prefix_stats.get(lora_prefix, {}).get("n_loras", 0)
                pf_conflict = prefix_stats.get(lora_prefix, {}).get("conflict_ratio", 0.0)
            elif optimization_mode == "per_prefix" and lora_prefix in prefix_stats:
                pf = prefix_stats[lora_prefix]
                pf_conflict = pf["conflict_ratio"]
                pf_n_loras = pf["n_loras"]
                if pf["n_loras"] <= 1:
                    pf_mode = "weighted_sum"
                    pf_density = 0.5
                    pf_sign = "frequency"
                else:
                    pf_mode, pf_density, pf_sign, _ = self._auto_select_params(
                        pf["conflict_ratio"], pf["magnitude_ratio"],
                        magnitude_samples=pf.get("magnitude_samples")
                    )
                    # Upgrade weighted_average → slerp for exactly 2 LoRAs
                    # SLERP preserves magnitude better (no cancellation from opposing vectors)
                    if pf_mode == "weighted_average" and pf["n_loras"] == 2:
                        pf_mode = "slerp"

            # Apply merge strategy override from Conflict Editor (takes priority over auto-selection)
            # Skip when user explicitly chose weighted_sum_only (protects DPO/edit LoRAs)
            if (merge_strategy_override and optimization_mode != "weighted_sum_only"
                    and merge_strategy_override in ("ties", "weighted_average", "weighted_sum")):
                pf_mode = merge_strategy_override

            # LOW-RANK PATH: single-LoRA weighted_sum — keep low-rank matrices
            # instead of expanding to full-rank diff. Saves ~128x memory per key.
            # ComfyUI applies "lora" patches as: up @ down * (alpha/rank) * strength
            if pf_mode == "weighted_sum" and pf_n_loras <= 1:
                for i, item in enumerate(active_loras):
                    lora_info = self._get_lora_key_info(item["lora"], lora_prefix)
                    if lora_info is None:
                        continue
                    mat_up, mat_down, alpha, mid = lora_info
                    if is_clip_key and item["clip_strength"] is not None:
                        eff_strength = item["clip_strength"]
                    else:
                        eff_strength = item["strength"]
                        if scale_ratios:
                            eff_strength *= scale_ratios.get(i, 1.0)
                    # Bake eff_strength into alpha so ComfyUI applies it correctly
                    alpha_scaled = alpha * eff_strength
                    patch = LoRAAdapter(set(), (mat_up, mat_down, alpha_scaled, mid, None, None))
                    return (target_key, is_clip_key, patch, pf_mode, lora_prefix, pf_conflict, max(pf_n_loras, 1), False)
                return None

            # FULL-RANK PATH: compute diffs on GPU, merge
            diffs_list = []
            diff_to_lora = []  # maps diffs_list index -> active_loras index
            for i, item in enumerate(active_loras):
                lora_info = self._get_lora_key_info(item["lora"], lora_prefix)
                if lora_info is None:
                    continue

                mat_up, mat_down, alpha, mid = lora_info
                rank = mat_down.shape[0]

                if use_gpu:
                    mat_up = mat_up.to(compute_device)
                    mat_down = mat_down.to(compute_device)
                    if mid is not None:
                        mid = mid.to(compute_device)

                if mid is not None:
                    final_shape = [mat_down.shape[1], mat_down.shape[0],
                                   mid.shape[2], mid.shape[3]]
                    mat_down = (
                        torch.mm(
                            mat_down.transpose(0, 1).flatten(start_dim=1).float(),
                            mid.transpose(0, 1).flatten(start_dim=1).float(),
                        )
                        .reshape(final_shape)
                        .transpose(0, 1)
                    )

                diff = torch.mm(
                    mat_up.flatten(start_dim=1).float(),
                    mat_down.flatten(start_dim=1).float()
                )
                del mat_up, mat_down  # Free LoRA matrices from GPU
                try:
                    diff = diff.reshape(target_shape)
                except RuntimeError:
                    continue

                diff = diff * (alpha / rank)

                if is_clip_key and item["clip_strength"] is not None:
                    eff_strength = item["clip_strength"]
                else:
                    eff_strength = item["strength"]
                    if scale_ratios:
                        eff_strength *= scale_ratios.get(i, 1.0)

                diffs_list.append((diff, eff_strength))
                diff_to_lora.append(i)

            if len(diffs_list) == 0:
                return None

            # Conflict-mode masking (zero overhead when all use "all")
            if len(diffs_list) > 1:
                has_conflict_modes = any(
                    active_loras[diff_to_lora[idx]].get("conflict_mode", "all") != "all"
                    for idx in range(len(diffs_list))
                )
                if has_conflict_modes:
                    # Unweighted sign voting (frequency method, consistent with TIES):
                    # each LoRA gets one vote per element regardless of strength.
                    sign_sum = torch.zeros_like(diffs_list[0][0])
                    for diff, _ in diffs_list:
                        sign_sum += diff.sign()
                    majority_sign = torch.where(sign_sum >= 0, 1.0, -1.0)

                    masked_diffs = []
                    for idx, (diff, weight) in enumerate(diffs_list):
                        cm = active_loras[diff_to_lora[idx]].get("conflict_mode", "all")
                        if cm == "low_conflict":
                            diff = diff * ((diff * majority_sign) > 0).float()
                        elif cm == "high_conflict":
                            diff = diff * ((diff * majority_sign) < 0).float()
                        masked_diffs.append((diff, weight))
                    diffs_list = masked_diffs
                    del sign_sum, majority_sign

            # Re-check single-LoRA case (diff computation may have failed for some)
            if pf_mode == "weighted_sum" and len(diffs_list) <= 1:
                pass  # weighted_sum with 1 diff is fine
            elif len(diffs_list) <= 1 and pf_mode != "weighted_sum":
                pf_mode = "weighted_sum"

            # Create deterministic per-prefix RNG for reproducible sparsification
            # Generator lives on compute_device so mask generation stays on GPU
            sp_gen = None
            if sparsification != "disabled":
                seed = int(hashlib.sha256(lora_prefix.encode()).hexdigest(), 16) % (2**63)
                gen_device = compute_device if compute_device is not None else torch.device('cpu')
                sp_gen = torch.Generator(device=gen_device)
                sp_gen.manual_seed(seed)

            merged_diff = self._merge_diffs(
                diffs_list, pf_mode,
                density=pf_density, majority_sign_method=pf_sign,
                compute_device=compute_device,
                sparsification=sparsification,
                sparsification_density=sparsification_density,
                sparsification_generator=sp_gen
            )
            diffs_list.clear()  # Free input diffs from GPU
            if merged_diff is None:
                return None
            # Compress full-rank diff to low-rank via SVD if requested
            # non_ties: skip compression on TIES prefixes (lossy); all: compress everything
            should_compress = (compress_rank > 0 and
                               (compress_patches == "all" or pf_mode != "ties"))
            if should_compress:
                patch = self._compress_to_lowrank(merged_diff, compress_rank, svd_device=resolved_svd_device)
                del merged_diff
                is_compressed = True
            else:
                patch = ("diff", (merged_diff,))
                is_compressed = False
            return (target_key, is_clip_key, patch, pf_mode, lora_prefix, pf_conflict, max(pf_n_loras, 1), is_compressed)

        lowrank_count = 0

        def _collect_merge_result(result):
            nonlocal processed_keys, lowrank_count, compressed_count
            if result is None:
                return
            target_key, is_clip_key, patch, used_mode, prefix, conflict, n_loras, is_compressed = result
            if is_clip_key:
                clip_patches[target_key] = patch
            else:
                model_patches[target_key] = patch
            processed_keys += 1
            if isinstance(patch, LoRAAdapter):
                lowrank_count += 1
            if is_compressed:
                compressed_count += 1
            strategy_counts[used_mode] = strategy_counts.get(used_mode, 0) + 1
            prefix_decisions.append((prefix, used_mode, conflict, n_loras))

        if use_gpu:
            for lora_prefix, (target_key, is_clip_key) in all_key_targets.items():
                _collect_merge_result(_merge_one_prefix(lora_prefix, target_key, is_clip_key))
        else:
            max_workers = min(4, max(1, len(all_key_targets)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_merge_one_prefix, lora_prefix, target_key, is_clip_key): lora_prefix
                    for lora_prefix, (target_key, is_clip_key) in all_key_targets.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    _collect_merge_result(future.result())

        fullrank_count = processed_keys - lowrank_count
        logging.info(f"[LoRA Optimizer]   Model patches: {len(model_patches)}, "
                     f"CLIP patches: {len(clip_patches)} ({time.time() - t_pass2:.1f}s)")
        if lowrank_count > 0:
            logging.info(f"[LoRA Optimizer]   Low-rank patches: {lowrank_count} "
                         f"(full-rank: {fullrank_count}) — "
                         f"~{lowrank_count}/{processed_keys} keys use minimal RAM")
        if optimization_mode == "per_prefix":
            logging.info(f"[LoRA Optimizer]   Per-prefix strategies: "
                         f"{strategy_counts.get('weighted_sum', 0)} sum, "
                         f"{strategy_counts.get('slerp', 0)} slerp, "
                         f"{strategy_counts.get('weighted_average', 0)} avg, "
                         f"{strategy_counts.get('ties', 0)} ties")
        if compressed_count > 0:
            passthrough_count = lowrank_count - compressed_count
            logging.info(f"[LoRA Optimizer]   SVD-compressed: {compressed_count} patches "
                         f"(rank {compress_rank}), passthrough: {passthrough_count}, "
                         f"full-rank: {fullrank_count}")

        # Free analysis data no longer needed
        prefix_stats.clear()
        self.loaded_loras.clear()
        if use_gpu:
            torch.cuda.empty_cache()

        # Re-fuse Z-Image QKV patches if architecture normalization was used
        if getattr(self, '_detected_arch', None) == 'zimage':
            if len(model_patches) > 0:
                model_patches = self._refuse_zimage_patches(model_patches)
                logging.info(f"[LoRA Optimizer] Re-fused Z-Image QKV patches ({len(model_patches)} model patches)")

        # Build reverse key map: target_key → lora_prefix
        # (used by SaveMergedLoRA to reconstruct standard LoRA key names)
        reverse_key_map = {}
        for lora_prefix, (target_key, is_clip) in all_key_targets.items():
            tkey = target_key[0] if isinstance(target_key, tuple) else target_key
            reverse_key_map[tkey] = lora_prefix

        # Apply patches
        new_model = model
        new_clip = clip

        # If ALL LoRAs have explicit clip_strength (standard tuple format),
        # clip strengths are already baked into the diffs — skip global multiplier.
        # If ANY lack it (dict format), apply the multiplier for those LoRAs.
        all_explicit_clip = all(item["clip_strength"] is not None for item in active_loras)
        if all_explicit_clip:
            clip_strength_out = output_strength
        else:
            clip_strength_out = output_strength * clip_strength_multiplier

        if model is not None and len(model_patches) > 0:
            new_model = model.clone()
            new_model.add_patches(model_patches, output_strength)

        if clip is not None and len(clip_patches) > 0:
            new_clip = clip.clone()
            new_clip.add_patches(clip_patches, clip_strength_out)

        merge_summary = {
            "keys_processed": processed_keys,
            "model_patches": len(model_patches),
            "clip_patches": len(clip_patches),
            "skipped_keys": skipped_keys,
            "output_strength": output_strength,
            "clip_strength": clip_strength_out,
        }

        report = self._build_report(
            lora_stats, pairwise_conflicts, collection_stats,
            mode, density, sign_method, reasoning, merge_summary,
            auto_strength_info=auto_strength_info,
            strategy_counts=strategy_counts if optimization_mode == "per_prefix" else None,
            optimization_mode=optimization_mode,
            prefix_decisions=prefix_decisions if optimization_mode == "per_prefix" else None,
            detected_arch=getattr(self, '_detected_arch', None),
            normalize_keys=normalize_keys,
            sparsification=sparsification,
            sparsification_density=sparsification_density,
        )

        # Bundle LORA_DATA for optional downstream saving
        lora_data = {
            "model_patches": model_patches,
            "clip_patches": clip_patches,
            "key_map": reverse_key_map,
            "output_strength": output_strength,
            "clip_strength": clip_strength_out,
        }

        # Cache patches for re-use (single entry to limit memory)
        if cache_patches == "enabled":
            self._merge_cache = {cache_key: (model_patches, clip_patches, report, clip_strength_out, lora_data)}
        else:
            self._merge_cache = {}
            logging.info("[LoRA Optimizer] Patch cache disabled — RAM freed after merge")

        # Save report to disk for later reference
        lora_combo = [[item["name"], item["strength"]] for item in active_loras]
        selected_params = {"mode": mode, "density": density, "sign_method": sign_method, "optimization_mode": optimization_mode}
        if optimization_mode == "per_prefix":
            selected_params["strategy_counts"] = dict(strategy_counts)
        report_path = self._save_report_to_disk(cache_key, lora_combo, auto_strength, report, selected_params)
        if report_path:
            logging.info(f"[LoRA Optimizer] Report saved to: {report_path}")

        logging.info(f"[LoRA Optimizer] Done! {processed_keys} keys processed ({time.time() - t_start:.1f}s total)")

        return (new_model, new_clip, report, lora_data)


class SaveMergedLoRA:
    """
    Saves merged LoRA patches from LoRA Optimizer as a standalone .safetensors
    file that can be loaded by any standard LoRA loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_data": ("LORA_DATA", {"tooltip": "Connect the lora_data output from LoRA Optimizer here."}),
                "filename": ("STRING", {"default": "merged_lora", "tooltip": "Name or path for the saved file. Plain name (e.g. 'merged_lora') saves to your ComfyUI loras folder. Absolute path (e.g. '/path/to/my_lora') saves to that location. Extension .safetensors is added automatically."}),
                "save_rank": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 4,
                    "tooltip": "0 = auto (uses each layer's existing rank from the merge — recommended). Non-zero = force this rank for any layers that need compression. Higher values = more accurate but larger file."
                }),
                "bake_strength": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, the saved LoRA reproduces your exact merge when loaded at strength 1.0. When disabled, strengths are not baked in — you'll need to set the strength manually when loading."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_lora"
    CATEGORY = "loaders/lora"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves merged LoRA data as a standalone .safetensors file that can be loaded by any standard LoRA loader."

    def save_lora(self, lora_data, filename, save_rank=0, bake_strength=True):
        # Determine save path
        if os.path.isabs(filename) or os.sep in filename or '/' in filename:
            # Absolute or relative path provided — use as-is
            save_path = filename if filename.endswith('.safetensors') else f"{filename}.safetensors"
            save_dir = os.path.dirname(save_path)
        else:
            # Plain filename — save to primary ComfyUI loras folder
            save_dir = folder_paths.get_folder_paths("loras")[0]
            save_path = os.path.join(save_dir, f"{filename}.safetensors")
        os.makedirs(save_dir, exist_ok=True)

        model_patches = lora_data["model_patches"]
        clip_patches = lora_data["clip_patches"]
        key_map = lora_data["key_map"]
        output_strength = lora_data["output_strength"]
        clip_strength = lora_data["clip_strength"]

        auto_rank = save_rank == 0

        # Auto mode: collect ranks from existing LoRAAdapter patches to use
        # as the fallback rank for any full-rank diffs that need compression
        if auto_rank:
            existing_ranks = []
            for patch in list(model_patches.values()) + list(clip_patches.values()):
                if isinstance(patch, LoRAAdapter):
                    existing_ranks.append(patch.weights[1].shape[0])  # mat_down rows = rank
            # Use the most common rank, or 128 as a safe default
            if existing_ranks:
                fallback_rank = max(set(existing_ranks), key=existing_ranks.count)
            else:
                fallback_rank = 128
            logging.info(f"[Save Merged LoRA] Auto rank: using rank {fallback_rank} for full-rank patches "
                         f"(from {len(existing_ranks)} low-rank patches)")

        state_dict = {}

        for target_key, patch in list(model_patches.items()) + list(clip_patches.items()):
            is_clip = target_key in clip_patches
            tkey = target_key[0] if isinstance(target_key, tuple) else target_key
            lora_prefix = key_map.get(tkey, tkey)

            if isinstance(patch, LoRAAdapter):
                mat_up, mat_down, alpha, mid, _, _ = patch.weights
                alpha = float(alpha) if alpha is not None else float(mat_down.shape[0])
            elif isinstance(patch, tuple) and len(patch) == 2 and patch[0] == "diff":
                diff_tensor = patch[1][0]
                rank = fallback_rank if auto_rank else save_rank
                compressed = LoRAOptimizer._compress_to_lowrank(diff_tensor, rank)
                mat_up, mat_down, alpha, mid, _, _ = compressed.weights
                alpha = float(alpha)
            else:
                logging.warning(f"[Save Merged LoRA] Skipping unknown patch type for {lora_prefix}: {type(patch)}")
                continue

            if bake_strength:
                strength = clip_strength if is_clip else output_strength
                alpha *= strength

            state_dict[f"{lora_prefix}.lora_up.weight"] = mat_up.contiguous()
            state_dict[f"{lora_prefix}.lora_down.weight"] = mat_down.contiguous()
            state_dict[f"{lora_prefix}.alpha"] = torch.tensor(alpha)

        save_file(state_dict, save_path)
        logging.info(f"[Save Merged LoRA] Saved {len(state_dict) // 3} LoRA keys to {save_path}")

        return (save_path,)


class LoRAConflictEditor(_LoRAMergeBase):
    """
    Analyzes pairwise sign conflicts between LoRAs in a stack and enriches
    each entry with an auto-suggested (or manually overridden) conflict_mode.

    Connect between a LoRA Stack and the LoRA Optimizer. The editor loads
    all LoRA weights, computes full-rank diffs, and measures how much each
    pair's weight updates disagree in sign. Based on the results it suggests
    per-LoRA conflict modes (all / low_conflict / high_conflict) and a
    merge strategy (ties / weighted_average / weighted_sum).

    The enriched stack and a human-readable analysis report are passed
    downstream so the optimizer can apply the right strategy per LoRA.
    """

    MAX_LORAS = 10

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "Connect a LoRA Stack node here. The editor will analyze conflicts between these LoRAs."
                }),
                "merge_strategy": (["auto", "ties", "weighted_average", "weighted_sum"], {
                    "default": "auto",
                    "tooltip": "Merge strategy to pass to the optimizer. "
                               "'auto': let the optimizer decide based on conflict analysis. "
                               "'ties': force TIES merging (good for high-conflict stacks). "
                               "'weighted_average': force simple averaging (good for compatible LoRAs). "
                               "'weighted_sum': force direct addition (preserves all weights exactly)."
                }),
            }
        }
        for i in range(1, cls.MAX_LORAS + 1):
            inputs["required"][f"conflict_mode_{i}"] = (["auto", "all", "low_conflict", "high_conflict"], {
                "default": "auto",
                "tooltip": f"LoRA #{i} conflict filter. "
                           f"'auto': node suggests based on analysis. "
                           f"'all': apply everywhere. "
                           f"'low_conflict': only where this LoRA agrees with the majority. "
                           f"'high_conflict': only where this LoRA disagrees."
            })
        return inputs

    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("lora_stack", "analysis_report", "merge_strategy")
    FUNCTION = "analyze_and_enrich"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Analyzes LoRA conflicts and lets you control per-LoRA conflict modes and merge strategy. Connect between a LoRA Stack and the LoRA Optimizer."

    def __init__(self):
        super().__init__()

    @classmethod
    def IS_CHANGED(cls, lora_stack, merge_strategy, **kwargs):
        """Cache key so ComfyUI skips re-execution when nothing changed."""
        h = hashlib.sha256()
        if lora_stack:
            first = lora_stack[0] if len(lora_stack) > 0 else None
            entries = []
            if isinstance(first, (tuple, list)):
                for entry in lora_stack:
                    cm = entry[3] if len(entry) > 3 else "all"
                    entries.append((str(entry[0]), float(entry[1]), float(entry[2]), cm))
            elif isinstance(first, dict):
                for item in lora_stack:
                    cm = item.get("conflict_mode", "all")
                    entries.append((str(item.get("name", "")), float(item.get("strength", 0)), cm))
            entries.sort()
            h.update(json.dumps(entries).encode())
        h.update(f"|ms={merge_strategy}".encode())
        # Include all conflict_mode widgets
        for i in range(1, cls.MAX_LORAS + 1):
            cm = kwargs.get(f"conflict_mode_{i}", "auto")
            h.update(f"|cm{i}={cm}".encode())
        return h.hexdigest()[:16]

    def analyze_and_enrich(self, lora_stack, merge_strategy, **kwargs):
        """
        Analyze pairwise conflicts in the LoRA stack, auto-suggest conflict
        modes, and return an enriched stack with the analysis report.
        """
        # --- 1. Normalize and filter ---
        normalized = self._normalize_stack(lora_stack)
        active_loras = [item for item in normalized if item["strength"] != 0]

        n_active = len(active_loras)
        if n_active == 0:
            return (lora_stack, "No active LoRAs in stack.", "")

        if n_active == 1:
            # Single LoRA — no pairwise conflicts to analyze
            cm = kwargs.get("conflict_mode_1", "auto")
            resolved = "all" if cm == "auto" else cm
            first = lora_stack[0]
            item = active_loras[0]
            if isinstance(first, dict):
                enriched = [dict(item, conflict_mode=resolved)]
            else:
                # clip_strength may be None (dict-format LoRAs use global multiplier);
                # preserve None so the optimizer applies clip_strength_multiplier correctly
                enriched = [(item["name"], item["strength"], item["clip_strength"], resolved)]
            report = (
                "=" * 46 + "\n"
                "LORA CONFLICT EDITOR - ANALYSIS REPORT\n"
                "=" * 46 + "\n\n"
                f"Single LoRA: {item['name']}\n"
                f"Conflict mode: {resolved}\n"
                f"No pairwise conflicts to analyze."
            )
            resolved_strategy = merge_strategy if merge_strategy != "auto" else "weighted_average"
            return (enriched, report, resolved_strategy)

        # Read per-LoRA conflict_mode widgets
        widget_modes = {}
        for i, item in enumerate(active_loras):
            widget_key = f"conflict_mode_{i + 1}"
            widget_modes[i] = kwargs.get(widget_key, "auto")

        # --- 2. Collect all unique LoRA prefixes ---
        # Must match the optimizer's suffix list (line ~2330) for consistency
        all_prefixes = set()
        suffixes_to_strip = [
            ".lora_up.weight", ".lora_down.weight",
            ".lora_A.weight", ".lora_B.weight",
            ".lora.up.weight", ".lora.down.weight",
            "_lora.up.weight", "_lora.down.weight",
        ]
        for item in active_loras:
            for key in item["lora"].keys():
                for suffix in suffixes_to_strip:
                    if key.endswith(suffix):
                        all_prefixes.add(key[:-len(suffix)])
                        break

        compute_device = self._get_compute_device()

        # --- 3. Pairwise conflict analysis ---
        # Accumulators: per-pair totals
        n_pairs = n_active * (n_active - 1) // 2
        pair_overlap = [[0] * n_active for _ in range(n_active)]
        pair_conflict = [[0] * n_active for _ in range(n_active)]
        pair_dot = [[0.0] * n_active for _ in range(n_active)]
        pair_norm_a_sq = [[0.0] * n_active for _ in range(n_active)]
        pair_norm_b_sq = [[0.0] * n_active for _ in range(n_active)]

        prefixes_analyzed = 0

        for prefix in sorted(all_prefixes):
            # Compute diffs for each LoRA that has this prefix
            diffs = {}
            for idx, item in enumerate(active_loras):
                info = self._get_lora_key_info(item["lora"], prefix)
                if info is None:
                    continue
                mat_up, mat_down, alpha, mid = info
                rank = mat_down.shape[0]
                scale = alpha / rank

                if compute_device.type != "cpu":
                    mat_up = mat_up.to(compute_device)
                    mat_down = mat_down.to(compute_device)
                    if mid is not None:
                        mid = mid.to(compute_device)

                if mid is not None:
                    final_shape = [mat_down.shape[1], mat_down.shape[0], mid.shape[2], mid.shape[3]]
                    mat_down = (
                        torch.mm(
                            mat_down.transpose(0, 1).flatten(start_dim=1).float(),
                            mid.transpose(0, 1).flatten(start_dim=1).float(),
                        )
                        .reshape(final_shape)
                        .transpose(0, 1)
                    )

                diff = torch.mm(
                    mat_up.flatten(start_dim=1).float(),
                    mat_down.flatten(start_dim=1).float()
                )
                del mat_up, mat_down
                diff = diff * scale * item["strength"]
                diffs[idx] = diff

            if len(diffs) < 2:
                continue

            prefixes_analyzed += 1

            # Pairwise conflict sampling
            indices = sorted(diffs.keys())
            for ai in range(len(indices)):
                for bi in range(ai + 1, len(indices)):
                    a_idx, b_idx = indices[ai], indices[bi]
                    result = self._sample_conflict(
                        diffs[a_idx], diffs[b_idx], device=compute_device
                    )
                    n_ov, n_cf, dot, na_sq, nb_sq = result
                    pair_overlap[a_idx][b_idx] += n_ov
                    pair_conflict[a_idx][b_idx] += n_cf
                    pair_dot[a_idx][b_idx] += dot
                    pair_norm_a_sq[a_idx][b_idx] += na_sq
                    pair_norm_b_sq[a_idx][b_idx] += nb_sq

            # Free diffs for this prefix immediately
            del diffs

        # --- 4. Compute per-LoRA average conflict ratio ---
        per_lora_conflict_ratios = [0.0] * n_active
        per_lora_pair_count = [0] * n_active

        # Also build pairwise summary for the report
        pair_summaries = []

        for i in range(n_active):
            for j in range(i + 1, n_active):
                total_ov = pair_overlap[i][j]
                total_cf = pair_conflict[i][j]
                total_dot = pair_dot[i][j]
                total_na = pair_norm_a_sq[i][j]
                total_nb = pair_norm_b_sq[i][j]

                if total_ov > 0:
                    conflict_ratio = total_cf / total_ov
                    denom = math.sqrt(total_na * total_nb) if (total_na > 0 and total_nb > 0) else 1.0
                    cosine_sim = total_dot / denom if denom > 0 else 0.0
                else:
                    conflict_ratio = 0.0
                    cosine_sim = 0.0

                per_lora_conflict_ratios[i] += conflict_ratio
                per_lora_conflict_ratios[j] += conflict_ratio
                per_lora_pair_count[i] += 1
                per_lora_pair_count[j] += 1

                pair_summaries.append({
                    "i": i, "j": j,
                    "overlap": total_ov,
                    "conflict_ratio": conflict_ratio,
                    "cosine_sim": cosine_sim,
                })

        # Average per-LoRA conflict ratio
        avg_conflict = [0.0] * n_active
        for i in range(n_active):
            if per_lora_pair_count[i] > 0:
                avg_conflict[i] = per_lora_conflict_ratios[i] / per_lora_pair_count[i]

        # --- 5. Auto-suggest conflict modes ---
        resolved_modes = []
        auto_reasons = []
        for i in range(n_active):
            widget_mode = widget_modes.get(i, "auto")
            if widget_mode != "auto":
                resolved_modes.append(widget_mode)
                auto_reasons.append("manual")
            else:
                ratio = avg_conflict[i]
                if ratio < 0.15:
                    mode = "all"
                    reason = f"auto \u2014 avg conflict {ratio:.1%} < 15%"
                elif ratio <= 0.40:
                    mode = "low_conflict"
                    reason = f"auto \u2014 avg conflict {ratio:.1%} (15\u201340%)"
                else:
                    mode = "high_conflict"
                    reason = f"auto \u2014 avg conflict {ratio:.1%} > 40%"
                resolved_modes.append(mode)
                auto_reasons.append(reason)

        # --- 6. Resolve merge strategy ---
        if merge_strategy == "auto":
            # Use the global average conflict across all pairs
            if n_pairs > 0 and len(pair_summaries) > 0:
                global_avg = sum(p["conflict_ratio"] for p in pair_summaries) / len(pair_summaries)
            else:
                global_avg = 0.0

            if global_avg > 0.25:
                resolved_strategy = "ties"
                strategy_reason = f"auto \u2014 avg conflict {global_avg:.1%} > 25%"
            else:
                resolved_strategy = "weighted_average"
                strategy_reason = f"auto \u2014 avg conflict {global_avg:.1%} \u2264 25%"
        else:
            resolved_strategy = merge_strategy
            strategy_reason = "manual"

        # --- 7. Build enriched output stack ---
        first = lora_stack[0] if lora_stack else None
        is_dict_format = isinstance(first, dict)

        enriched = []
        active_idx = 0
        for item in normalized:
            if item["strength"] == 0:
                # Pass through inactive LoRAs unchanged
                if is_dict_format:
                    enriched.append(item)
                else:
                    enriched.append((item["name"], item["strength"],
                                     item["clip_strength"], item.get("conflict_mode", "all")))
            else:
                mode = resolved_modes[active_idx]
                if is_dict_format:
                    enriched_item = dict(item)
                    enriched_item["conflict_mode"] = mode
                    enriched.append(enriched_item)
                else:
                    enriched.append((
                        item["name"],
                        item["strength"],
                        item["clip_strength"],
                        mode,
                    ))
                active_idx += 1

        # --- 8. Build report ---
        short_names = []
        for item in active_loras:
            short_names.append(os.path.splitext(os.path.basename(item["name"]))[0])

        report = self._build_conflict_report(
            active_loras, short_names, pair_summaries,
            resolved_modes, auto_reasons,
            resolved_strategy, strategy_reason,
            prefixes_analyzed,
        )

        return (enriched, report, resolved_strategy)

    @staticmethod
    def _build_conflict_report(active_loras, short_names, pair_summaries,
                               resolved_modes, auto_reasons,
                               resolved_strategy, strategy_reason,
                               prefixes_analyzed):
        """Build a human-readable conflict analysis report."""
        lines = []
        lines.append("=" * 60)
        lines.append("LoRA Conflict Analysis Report")
        lines.append("=" * 60)

        # Stack overview
        lines.append("")
        lines.append("Stack Overview")
        lines.append("-" * 40)
        for i, item in enumerate(active_loras):
            lines.append(f"  {i + 1}. {short_names[i]}  (strength={item['strength']:.2f})")
        lines.append(f"  Prefixes analyzed: {prefixes_analyzed}")

        # Pairwise conflicts
        if pair_summaries:
            lines.append("")
            lines.append("Pairwise Conflicts")
            lines.append("-" * 40)
            for p in pair_summaries:
                name_a = short_names[p["i"]]
                name_b = short_names[p["j"]]
                lines.append(
                    f"  {name_a} \u00d7 {name_b}: "
                    f"overlap={p['overlap']:,}  "
                    f"sign_conflict={p['conflict_ratio']:.1%}  "
                    f"cosine={p['cosine_sim']:.3f}"
                )

        # Applied conflict modes
        lines.append("")
        lines.append("Applied Conflict Modes")
        lines.append("-" * 40)
        for i in range(len(active_loras)):
            lines.append(f"  {i + 1}. {short_names[i]}: {resolved_modes[i]}  ({auto_reasons[i]})")

        # Merge strategy
        lines.append("")
        lines.append("Merge Strategy")
        lines.append("-" * 40)
        lines.append(f"  {resolved_strategy}  ({strategy_reason})")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoRAStack": LoRAStack,
    "LoRAStackDynamic": LoRAStackDynamic,
    "LoRAOptimizer": LoRAOptimizer,
    "SaveMergedLoRA": SaveMergedLoRA,
    "LoRAConflictEditor": LoRAConflictEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAStack": "LoRA Stack",
    "LoRAStackDynamic": "LoRA Stack (Dynamic)",
    "LoRAOptimizer": "LoRA Optimizer",
    "SaveMergedLoRA": "Save Merged LoRA",
    "LoRAConflictEditor": "LoRA Conflict Editor",
}
