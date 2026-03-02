"""
Z-Image LoRA Merger for ComfyUI
Custom nodes for combining multiple LoRAs without overexposure on distilled models.

Problem: Standard LoRA application adds effects additively, causing overexposure
on distilled models like Z-Image Turbo, SDXL-Turbo, LCM, etc.

Solution: Various blending strategies to normalize the combined LoRA effect.

Author: DanrisiUA (https://github.com/DanrisiUA)
"""

import torch
import logging
import math
import os
import json
import hashlib
import time
import concurrent.futures
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora


class ZImageLoRAMerger:
    """
    Node for combining multiple LoRAs with various blending strategies,
    optimized for Z-Image Turbo and other distilled models.
    """
    
    BLEND_MODES = [
        "normalize",      # Normalizes total strength to target_strength
        "average",        # Averages LoRA effects
        "sqrt_scale",     # Scales each LoRA by 1/sqrt(n)
        "linear_decay",   # Linear decay: 1, 0.5, 0.33, ...
        "geometric_decay",# Geometric decay: 1, 0.5, 0.25, ...
        "additive",       # Standard additive (for comparison)
    ]
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model"}),
                "blend_mode": (cls.BLEND_MODES, {"default": "normalize", "tooltip": "LoRA blending mode"}),
                "target_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, 
                                              "tooltip": "Target total strength (for normalize/average)"}),
                "lora_1": (["None"] + lora_list, {"tooltip": "First LoRA"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_2": (["None"] + lora_list, {"tooltip": "Second LoRA"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_3": (["None"] + lora_list, {"tooltip": "Third LoRA"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_4": (["None"] + lora_list, {"tooltip": "Fourth LoRA"}),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_5": (["None"] + lora_list, {"tooltip": "Fifth LoRA"}),
                "strength_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                                       "tooltip": "Strength multiplier for CLIP"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "merge_loras"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Combines multiple LoRAs with normalization for Z-Image Turbo and other distilled models"
    
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
    
    def _calculate_blend_factors(self, strengths, blend_mode, target_strength):
        """
        Calculates blending coefficients for each LoRA
        based on the selected mode.
        """
        n = len(strengths)
        if n == 0:
            return []
        
        if blend_mode == "additive":
            # Standard additive application
            return strengths
        
        elif blend_mode == "normalize":
            # Normalizes so that sum of squared strengths = target_strength^2
            # This preserves the "energy" of the effect
            sum_sq = sum(s*s for s in strengths)
            if sum_sq == 0:
                return [0.0] * n
            scale = target_strength / math.sqrt(sum_sq)
            return [s * scale for s in strengths]
        
        elif blend_mode == "average":
            # Simple averaging with target_strength
            scale = target_strength / n
            return [s * scale for s in strengths]
        
        elif blend_mode == "sqrt_scale":
            # Scales by 1/sqrt(n) - maintains balance when adding LoRAs
            scale = 1.0 / math.sqrt(n)
            return [s * scale for s in strengths]
        
        elif blend_mode == "linear_decay":
            # Each subsequent LoRA has less weight: 1, 0.5, 0.33, 0.25, ...
            factors = [1.0 / (i + 1) for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        
        elif blend_mode == "geometric_decay":
            # Geometric decay: 1, 0.5, 0.25, 0.125, ...
            factors = [0.5 ** i for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        
        else:
            return strengths
    
    def merge_loras(self, model, clip, blend_mode, target_strength,
                    lora_1, strength_1, lora_2, strength_2,
                    lora_3="None", strength_3=1.0,
                    lora_4="None", strength_4=1.0,
                    lora_5="None", strength_5=1.0,
                    clip_strength_multiplier=1.0):
        """
        Main function for combining LoRAs.
        """
        
        # Collect active LoRAs
        loras_data = []
        for lora_name, strength in [
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4),
            (lora_5, strength_5),
        ]:
            if lora_name != "None" and lora_name is not None and strength != 0:
                lora = self._load_lora(lora_name)
                if lora is not None:
                    loras_data.append((lora, strength))
        
        if len(loras_data) == 0:
            return (model, clip)
        
        # Calculate blend factors
        original_strengths = [s for _, s in loras_data]
        blended_strengths = self._calculate_blend_factors(
            original_strengths, blend_mode, target_strength
        )
        
        logging.info(f"Z-Image LoRA Merger: {blend_mode} mode, {len(loras_data)} LoRAs")
        logging.info(f"  Original strengths: {original_strengths}")
        logging.info(f"  Blended strengths:  {[round(s, 4) for s in blended_strengths]}")
        
        # Apply LoRAs sequentially with calculated coefficients
        new_model = model
        new_clip = clip
        
        for i, ((lora, _), strength) in enumerate(zip(loras_data, blended_strengths)):
            clip_strength = strength * clip_strength_multiplier
            new_model, new_clip = comfy.sd.load_lora_for_models(
                new_model, new_clip, lora, strength, clip_strength
            )
        
        return (new_model, new_clip)


class ZImageLoRAStack:
    """
    Node for creating a LoRA stack that can be used with ZImageLoRAStackApply.
    Allows more flexible LoRA management through connections.
    """
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (lora_list, {"tooltip": "LoRA file"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Previous LoRA stack"}),
            }
        }
    
    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "add_to_stack"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Adds LoRA to stack for later application with ZImageLoRAStackApply"
    
    def add_to_stack(self, lora_name, strength, lora_stack=None):
        lora_list = list(lora_stack) if lora_stack else []
        
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        lora_list.append({
            "name": lora_name,
            "lora": lora,
            "strength": strength
        })
        
        return (lora_list,)


class ZImageLoRAStackApply:
    """
    Applies a LoRA stack with various blending modes.
    """
    
    BLEND_MODES = ZImageLoRAMerger.BLEND_MODES
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model"}),
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRA stack"}),
                "blend_mode": (cls.BLEND_MODES, {"default": "normalize"}),
                "target_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_stack"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Applies LoRA stack with selected blending mode"
    
    def _calculate_blend_factors(self, strengths, blend_mode, target_strength):
        """Same logic as ZImageLoRAMerger"""
        n = len(strengths)
        if n == 0:
            return []
        
        if blend_mode == "additive":
            return strengths
        elif blend_mode == "normalize":
            sum_sq = sum(s*s for s in strengths)
            if sum_sq == 0:
                return [0.0] * n
            scale = target_strength / math.sqrt(sum_sq)
            return [s * scale for s in strengths]
        elif blend_mode == "average":
            scale = target_strength / n
            return [s * scale for s in strengths]
        elif blend_mode == "sqrt_scale":
            scale = 1.0 / math.sqrt(n)
            return [s * scale for s in strengths]
        elif blend_mode == "linear_decay":
            factors = [1.0 / (i + 1) for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        elif blend_mode == "geometric_decay":
            factors = [0.5 ** i for i in range(n)]
            total = sum(factors)
            return [strengths[i] * factors[i] * target_strength / total for i in range(n)]
        else:
            return strengths
    
    def apply_stack(self, model, clip, lora_stack, blend_mode, target_strength, clip_strength_multiplier):
        if not lora_stack or len(lora_stack) == 0:
            return (model, clip)
        
        original_strengths = [item["strength"] for item in lora_stack]
        blended_strengths = self._calculate_blend_factors(
            original_strengths, blend_mode, target_strength
        )
        
        logging.info(f"Z-Image LoRA Stack Apply: {blend_mode} mode, {len(lora_stack)} LoRAs")
        logging.info(f"  Original strengths: {original_strengths}")
        logging.info(f"  Blended strengths:  {[round(s, 4) for s in blended_strengths]}")
        
        new_model = model
        new_clip = clip
        
        for item, strength in zip(lora_stack, blended_strengths):
            clip_strength = strength * clip_strength_multiplier
            new_model, new_clip = comfy.sd.load_lora_for_models(
                new_model, new_clip, item["lora"], strength, clip_strength
            )
        
        return (new_model, new_clip)


class ZImageLoRAMergeToSingle:
    """
    Merges weights of multiple LoRAs into a single "virtual" LoRA
    by pre-merging weights before applying to the model.
    
    This can give better results than sequential application,
    especially for overlapping weights.
    """
    
    MERGE_METHODS = [
        "weighted_sum",    # Weighted sum
        "add_difference",  # A + (B - C) * weight
        "weighted_average",# Weighted average
    ]
    
    def __init__(self):
        self.loaded_loras = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "merge_method": (cls.MERGE_METHODS, {"default": "weighted_average"}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "Strength of the merged LoRA"}),
                "lora_1": (["None"] + lora_list,),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_2": (["None"] + lora_list,),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "lora_3": (["None"] + lora_list,),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "merge_to_single"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Merges multiple LoRAs into one before applying - optimal for Z-Image Turbo"
    
    def _load_lora(self, lora_name):
        if lora_name == "None" or lora_name is None:
            return None
        if lora_name in self.loaded_loras:
            return self.loaded_loras[lora_name]
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_loras[lora_name] = lora
        return lora
    
    def _merge_lora_weights(self, loras_with_weights, method):
        """
        Merges LoRA weights into a single dictionary.
        Handles LoRAs with different ranks - incompatible layers are taken
        proportionally from the LoRA that has them.
        """
        if len(loras_with_weights) == 0:
            return {}
        
        if len(loras_with_weights) == 1:
            lora, weight = loras_with_weights[0]
            return {k: v * weight for k, v in lora.items()}
        
        # Collect all keys
        all_keys = set()
        for lora, _ in loras_with_weights:
            all_keys.update(lora.keys())
        
        merged = {}
        skipped_keys = 0
        
        for key in all_keys:
            tensors_weights = []
            for lora, weight in loras_with_weights:
                if key in lora:
                    tensors_weights.append((lora[key], weight))
            
            if len(tensors_weights) == 0:
                continue
            
            if len(tensors_weights) == 1:
                merged[key] = tensors_weights[0][0] * tensors_weights[0][1]
                continue
            
            # Get reference shape and dtype
            ref_tensor = tensors_weights[0][0]
            ref_shape = ref_tensor.shape
            device = ref_tensor.device
            dtype = ref_tensor.dtype
            
            # Filter only tensors with compatible shapes
            compatible_tensors = [(t, w) for t, w in tensors_weights if t.shape == ref_shape]
            
            # If shapes don't match, take weighted sum of compatible ones only
            # or first tensor if none are compatible
            if len(compatible_tensors) < len(tensors_weights):
                skipped_keys += 1
                if len(compatible_tensors) == 0:
                    # No compatible - take first with its weight
                    merged[key] = ref_tensor * tensors_weights[0][1]
                    continue
                elif len(compatible_tensors) == 1:
                    merged[key] = compatible_tensors[0][0] * compatible_tensors[0][1]
                    continue
                tensors_weights = compatible_tensors
            
            if method == "weighted_sum":
                result = torch.zeros_like(ref_tensor)
                for tensor, weight in tensors_weights:
                    result += tensor.to(device=device, dtype=dtype) * weight
                merged[key] = result
                
            elif method == "weighted_average":
                result = torch.zeros_like(ref_tensor)
                total_weight = sum(w for _, w in tensors_weights)
                if total_weight > 0:
                    for tensor, weight in tensors_weights:
                        result += tensor.to(device=device, dtype=dtype) * (weight / total_weight)
                merged[key] = result
                
            elif method == "add_difference":
                # First LoRA + difference with others
                result = tensors_weights[0][0].clone() * tensors_weights[0][1]
                for tensor, weight in tensors_weights[1:]:
                    diff = tensor.to(device=device, dtype=dtype) - tensors_weights[0][0]
                    result += diff * weight
                merged[key] = result
        
        if skipped_keys > 0:
            logging.warning(f"Z-Image LoRA Merge: {skipped_keys} keys had incompatible shapes (different LoRA ranks) - used first compatible tensor")
        
        return merged
    
    def merge_to_single(self, model, clip, merge_method, output_strength,
                        lora_1, weight_1, lora_2, weight_2,
                        lora_3="None", weight_3=1.0,
                        clip_strength_multiplier=1.0):
        
        loras_with_weights = []
        for lora_name, weight in [(lora_1, weight_1), (lora_2, weight_2), (lora_3, weight_3)]:
            if lora_name != "None" and lora_name is not None and weight != 0:
                lora = self._load_lora(lora_name)
                if lora is not None:
                    loras_with_weights.append((lora, weight))
        
        if len(loras_with_weights) == 0:
            return (model, clip)
        
        logging.info(f"Z-Image LoRA Merge: {merge_method}, {len(loras_with_weights)} LoRAs -> 1")
        
        merged_lora = self._merge_lora_weights(loras_with_weights, merge_method)
        
        clip_strength = output_strength * clip_strength_multiplier
        new_model, new_clip = comfy.sd.load_lora_for_models(
            model, clip, merged_lora, output_strength, clip_strength
        )
        
        return (new_model, new_clip)


class ZImageLoRATrueMerge:
    """
    "True" LoRA merging by computing full weight diffs.

    Works for LoRAs of ANY rank!

    Instead of merging raw A/B matrices (impossible for different ranks),
    computes the full diff = A @ B × alpha for each LoRA,
    then merges these diffs and applies as a single patch.

    Supports TIES-Merging (Trim, Elect Sign, Disjoint Merge — NeurIPS 2023)
    for resolving sign conflicts and filtering redundant noise, recommended
    for distilled/turbo models when stacking multiple LoRAs.

    Warning: Requires more memory and time than standard application.
    """
    
    MERGE_MODES = [
        "weighted_average",  # Weighted average of diffs
        "weighted_sum",      # Weighted sum (can be brighter)
        "normalize",         # Energy normalization
        "ties",              # TIES-Merging: Trim, Elect Sign, Disjoint Merge
    ]
    
    def __init__(self):
        self.loaded_loras = {}

    @staticmethod
    def _get_compute_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model"}),
                "merge_mode": (cls.MERGE_MODES, {"default": "weighted_average"}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "Strength of the merged effect"}),
                "lora_1": (["None"] + lora_list,),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_2": (["None"] + lora_list,),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_3": (["None"] + lora_list,),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_4": (["None"] + lora_list,),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "density": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                                      "tooltip": "TIES: fraction of top-magnitude weights to keep (lower = more aggressive pruning)"}),
                "majority_sign_method": (["frequency", "total"], {"default": "frequency",
                                         "tooltip": "TIES: how to elect majority sign — 'frequency' counts votes, 'total' sums magnitudes"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "true_merge"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "True LoRA merging for any rank combination. Supports TIES-Merging for distilled/turbo models."
    
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
    def _ties_elect_sign(trimmed_diffs, method="frequency"):
        """
        TIES Step 2: Elect Sign — determine majority sign direction per weight position.

        Args:
            trimmed_diffs: list of trimmed diff tensors (same shape)
            method: "frequency" (count votes) or "total" (sum magnitudes)

        Returns:
            majority_sign: tensor of +1/-1 per position
        """
        stacked = torch.stack(trimmed_diffs)  # [N, *shape]
        if method == "total":
            # Sum of signed values — magnitude-weighted vote
            total = stacked.sum(dim=0)
        else:
            # Count of positive vs negative (frequency vote)
            total = (stacked > 0).float().sum(dim=0) - (stacked < 0).float().sum(dim=0)
        # +1 where majority is positive or tied, -1 where majority is negative
        majority_sign = torch.where(total >= 0,
                                    torch.ones_like(total),
                                    -torch.ones_like(total))
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
            # Mask: keep only positions where diff sign matches majority
            sign_match = (diff_f.sign() == majority_sign) & (diff_f != 0)
            result += torch.where(sign_match, diff_f * weight, torch.zeros_like(diff_f))
            contributor_count += sign_match.float()

        # Average by number of contributors (avoid div-by-zero)
        contributor_count = contributor_count.clamp(min=1.0)
        return result / contributor_count

    def _merge_diffs(self, diffs_with_weights, mode, density=0.5, majority_sign_method="frequency"):
        """
        Merges a list of diffs with their weights.
        """
        if len(diffs_with_weights) == 0:
            return None

        if len(diffs_with_weights) == 1:
            diff, weight = diffs_with_weights[0]
            return diff * weight

        # All diffs should have the same shape (verified during computation)
        ref_diff = diffs_with_weights[0][0]
        device = ref_diff.device
        dtype = ref_diff.dtype

        if mode == "weighted_average":
            result = torch.zeros_like(ref_diff, dtype=torch.float32)
            total_weight = sum(abs(w) for _, w in diffs_with_weights)
            if total_weight == 0:
                return result.to(dtype)
            for diff, weight in diffs_with_weights:
                result += diff.to(device=device, dtype=torch.float32) * (weight / total_weight)
            return result.to(dtype)

        elif mode == "weighted_sum":
            result = torch.zeros_like(ref_diff, dtype=torch.float32)
            for diff, weight in diffs_with_weights:
                result += diff.to(device=device, dtype=torch.float32) * weight
            return result.to(dtype)

        elif mode == "normalize":
            # Normalization by "energy" (sum of squared weights)
            weights = [w for _, w in diffs_with_weights]
            sum_sq = sum(w*w for w in weights)
            if sum_sq == 0:
                return torch.zeros_like(ref_diff)
            scale = 1.0 / math.sqrt(sum_sq)

            result = torch.zeros_like(ref_diff, dtype=torch.float32)
            for diff, weight in diffs_with_weights:
                result += diff.to(device=device, dtype=torch.float32) * weight * scale
            return result.to(dtype)

        elif mode == "ties":
            # TIES-Merging: Trim, Elect Sign, Disjoint Merge
            # Pre-multiply diffs by sign(weight) so negative strengths vote correctly,
            # then use abs(weight) for magnitude in disjoint merge.
            diffs = []
            abs_weights = []
            for d, w in diffs_with_weights:
                d_f = d.to(device=device, dtype=torch.float32)
                if w < 0:
                    d_f = -d_f
                diffs.append(d_f)
                abs_weights.append(abs(w))

            # Step 1: Trim each diff
            trimmed = [self._ties_trim(d, density) for d in diffs]

            # Step 2: Elect majority sign
            majority_sign = self._ties_elect_sign(trimmed, majority_sign_method)

            # Step 3: Disjoint merge
            result = self._ties_disjoint_merge(trimmed, abs_weights, majority_sign)
            return result.to(dtype)

        return None
    
    def true_merge(self, model, clip, merge_mode, output_strength,
                   lora_1, strength_1, lora_2, strength_2,
                   lora_3="None", strength_3=1.0,
                   lora_4="None", strength_4=1.0,
                   clip_strength_multiplier=1.0,
                   density=0.5, majority_sign_method="frequency"):
        """
        Main function for true LoRA merging.
        """

        # Collect active LoRAs
        loras_data = []
        for lora_name, strength in [
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4),
        ]:
            if lora_name != "None" and lora_name is not None and strength != 0:
                lora = self._load_lora(lora_name)
                if lora is not None:
                    loras_data.append((lora_name, lora, strength))

        if len(loras_data) == 0:
            return (model, clip)

        logging.info(f"Z-Image LoRA True Merge: {merge_mode} mode, {len(loras_data)} LoRAs")
        if merge_mode == "ties":
            logging.info(f"  TIES params: density={density}, majority_sign={majority_sign_method}")
        for name, _, strength in loras_data:
            logging.info(f"  - {name}: strength={strength}")
        
        # Get key_map for model
        model_keys = {}
        if model is not None:
            model_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        
        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
        
        # Collect all keys from all LoRAs
        all_lora_prefixes = set()
        for _, lora_dict, _ in loras_data:
            for key in lora_dict.keys():
                # Extract prefix (before .lora_up, .lora_down, etc.)
                for suffix in [".lora_up.weight", ".lora_down.weight", "_lora.up.weight", 
                              "_lora.down.weight", ".lora_B.weight", ".lora_A.weight",
                              ".lora.up.weight", ".lora.down.weight", ".alpha"]:
                    if key.endswith(suffix):
                        prefix = key[:-len(suffix)]
                        all_lora_prefixes.add(prefix)
                        break
        
        # For each key, compute merged diff
        model_patches = {}
        clip_patches = {}
        processed_keys = 0
        
        for lora_prefix in all_lora_prefixes:
            # Find target key in model
            target_key = None
            is_clip = False
            
            if lora_prefix in model_keys:
                target_key = model_keys[lora_prefix]
            elif lora_prefix in clip_keys:
                target_key = clip_keys[lora_prefix]
                is_clip = True
            
            if target_key is None:
                continue
            
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
                continue

            # Compute diff for each LoRA
            diffs_with_weights = []
            for _, lora_dict, strength in loras_data:
                lora_info = self._get_lora_key_info(lora_dict, lora_prefix)
                if lora_info is None:
                    continue
                
                mat_up, mat_down, alpha, mid = lora_info
                diff = self._compute_lora_diff(mat_up, mat_down, alpha, mid, target_shape)
                
                if diff is not None:
                    diffs_with_weights.append((diff, strength))
            
            if len(diffs_with_weights) == 0:
                continue
            
            # Merge diffs
            merged_diff = self._merge_diffs(diffs_with_weights, merge_mode,
                                            density=density, majority_sign_method=majority_sign_method)
            if merged_diff is not None:
                if is_clip:
                    clip_patches[target_key] = ("diff", (merged_diff,))
                else:
                    model_patches[target_key] = ("diff", (merged_diff,))
                processed_keys += 1
        
        logging.info(f"  Processed {processed_keys} weight keys")
        
        # Apply to model
        new_model = model
        new_clip = clip

        if model is not None and len(model_patches) > 0:
            new_model = model.clone()
            new_model.add_patches(model_patches, output_strength)

        if clip is not None and len(clip_patches) > 0:
            new_clip = clip.clone()
            clip_strength = output_strength * clip_strength_multiplier
            new_clip.add_patches(clip_patches, clip_strength)
        
        return (new_model, new_clip)


class ZImageLoRAOptimizer(ZImageLoRATrueMerge):
    """
    Auto-optimizer that analyzes a LoRA stack (sign conflicts, magnitude
    distributions, overlap) and automatically selects the best merge mode
    and parameters, then performs the merge.

    Outputs the merged model/clip plus an analysis report explaining
    what was chosen and why.

    Warning: Expands all LoRA diffs into full-rank tensors and holds them
    in memory for cross-key analysis. Memory usage scales with
    (n_loras * n_keys * weight_size). For SDXL (~800 keys) with 4 LoRAs
    this can exceed 10GB. Pairwise conflict analysis is O(n^2) in the
    number of LoRAs. For large stacks (4+) or SDXL models, consider using
    ZImageLoRATrueMerge with manually chosen parameters instead.
    """

    def __init__(self):
        self.loaded_loras = {}

    def _normalize_stack(self, lora_stack):
        """
        Normalize a LoRA stack into a consistent list of dicts.

        Accepts two formats:
        - Standard tuples: [(lora_name, model_strength, clip_strength), ...]
          Used by Efficiency Nodes, Comfyroll, and other popular node packs.
          LoRAs are loaded from disk (cached in self.loaded_loras).
        - ZImageLoRAStack dicts: [{"name": str, "lora": dict, "strength": float}, ...]
          Already loaded, clip_strength defaults to None (use global multiplier).

        Returns list of dicts with keys: name, lora, strength, clip_strength.
        clip_strength is None when the global multiplier should be used.
        """
        if not lora_stack:
            return []

        first = lora_stack[0]

        if isinstance(first, (tuple, list)):
            # Standard format: (lora_name, model_strength, clip_strength)
            normalized = []
            for entry in lora_stack:
                if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                    logging.warning("[Z-Image Optimizer] Skipping malformed tuple entry (expected 3 elements)")
                    continue
                lora_name, model_str, clip_str = entry[0], entry[1], entry[2]

                # Load LoRA with caching
                if lora_name in self.loaded_loras:
                    lora_dict = self.loaded_loras[lora_name]
                else:
                    try:
                        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                        lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        self.loaded_loras[lora_name] = lora_dict
                    except Exception as e:
                        logging.warning(f"[Z-Image Optimizer] Failed to load LoRA '{lora_name}': {e}")
                        continue

                normalized.append({
                    "name": lora_name,
                    "lora": lora_dict,
                    "strength": model_str,
                    "clip_strength": clip_str,
                })
            return normalized

        elif isinstance(first, dict):
            # ZImageLoRAStack format: already loaded dicts
            normalized = []
            for item in lora_stack:
                if not isinstance(item, dict) or "lora" not in item or "strength" not in item or "name" not in item:
                    logging.warning("[Z-Image Optimizer] Skipping malformed dict entry (expected keys: name, lora, strength)")
                    continue
                normalized.append({
                    "name": item["name"],
                    "lora": item["lora"],
                    "strength": item["strength"],
                    "clip_strength": None,  # use global multiplier
                })
            return normalized

        logging.warning("[Z-Image Optimizer] Unrecognized stack format")
        return []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model"}),
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRA stack - accepts standard (name, model_str, clip_str) tuples or ZImageLoRAStack dicts"}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "Strength of the merged effect"}),
            },
            "optional": {
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                                       "tooltip": "Strength multiplier for CLIP"}),
                "auto_strength": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Auto-reduce LoRA strengths to prevent overexposure from stacking"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "analysis_report")
    FUNCTION = "optimize_merge"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Auto-analyzes LoRA stack and selects optimal merge strategy. Outputs merged model + analysis report."

    @staticmethod
    def _compute_cache_key(lora_stack, output_strength, clip_strength_multiplier, auto_strength):
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
                    entries.append((str(entry[0]), float(entry[1]), float(entry[2])))
            elif isinstance(first, dict):
                for item in lora_stack:
                    entries.append((str(item.get("name", "")), float(item.get("strength", 0))))
            entries.sort()
            h.update(json.dumps(entries).encode())
        h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}".encode())
        return h.hexdigest()[:16]

    @classmethod
    def IS_CHANGED(cls, model, clip, lora_stack, output_strength,
                   clip_strength_multiplier=1.0, auto_strength="disabled"):
        return cls._compute_cache_key(lora_stack, output_strength,
                                      clip_strength_multiplier, auto_strength)

    def _save_report_to_disk(self, cache_key, lora_combo, auto_strength, report, selected_params):
        """
        Persist the analysis report as JSON for later reference.
        Saved to {user_dir}/zimage_lora_reports/{cache_key}.json.
        Failures are silently logged — never blocks the merge.
        """
        try:
            user_dir = folder_paths.get_user_directory()
            report_dir = os.path.join(user_dir, "zimage_lora_reports")
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
            logging.warning(f"[Z-Image Optimizer] Failed to save report: {e}")
            return None

    def _sample_conflict(self, diff_a, diff_b, device=None):
        """
        Compute sign conflict ratio between two diff tensors.
        Samples up to 100k positions for large tensors.
        Returns (n_overlap, n_conflict).
        """
        if device is not None:
            flat_a = diff_a.flatten().float().to(device)
            flat_b = diff_b.flatten().float().to(device)
        else:
            flat_a = diff_a.flatten().float()
            flat_b = diff_b.flatten().float()

        if flat_a.numel() != flat_b.numel():
            return (0, 0)

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
            return (0, 0)

        n_conflict = (flat_a[mask].sign() != flat_b[mask].sign()).sum().item()
        return (n_overlap, n_conflict)

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

    def _process_pair(self, i, j, active_loras, per_lora_diffs, device):
        """
        Compute pairwise conflict stats for LoRA indices i and j across all
        prefixes. Thread-safe — reads from shared immutable data.

        Returns (pair_overlap, pair_conflict, ratio, pair_label).
        """
        pair_overlap = 0
        pair_conflict = 0
        for lora_prefix, lora_diffs in per_lora_diffs.items():
            if i in lora_diffs and j in lora_diffs:
                ov, conf = self._sample_conflict(lora_diffs[i], lora_diffs[j], device=device)
                pair_overlap += ov
                pair_conflict += conf

        ratio = pair_conflict / pair_overlap if pair_overlap > 0 else 0
        name_i = active_loras[i]['name']
        name_j = active_loras[j]['name']
        if name_i == name_j:
            pair_label = f"{name_i} [#{i+1}, str={active_loras[i]['strength']}] vs {name_j} [#{j+1}, str={active_loras[j]['strength']}]"
        else:
            pair_label = f"{name_i} vs {name_j}"

        return (pair_overlap, pair_conflict, ratio, pair_label)

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

    def _compute_auto_strengths(self, active_loras, lora_stats):
        """
        Compute reduced per-LoRA strengths using L2-aware energy normalization.
        Scales strengths so the total combined energy matches the strongest
        single LoRA's contribution, preventing overexposure from stacking.

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

        # Current combined energy: sqrt(sum(effective[i]^2))
        current_energy = math.sqrt(sum(e * e for e in effective))

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
        reasoning.append("Method: L2-aware energy normalization")
        for i in range(n):
            if effective[i] > 0 and abs(scale - 1.0) > 1e-9:
                reasoning.append(f"  {active_loras[i]['name']}: {original_strengths[i]} -> {new_strengths[i]:.4f}")
            else:
                reasoning.append(f"  {active_loras[i]['name']}: {original_strengths[i]} (unchanged)")

        return (new_strengths, reasoning)

    def _auto_select_params(self, avg_conflict_ratio, magnitude_ratio, all_key_diffs):
        """
        Decision logic for auto-selecting merge parameters.
        Returns (mode, density, sign_method, reasoning_lines).
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
                      auto_strength_info=None):
        """Format analysis as a multi-line report string."""
        lines = []
        lines.append("=" * 50)
        lines.append("Z-IMAGE LORA OPTIMIZER - ANALYSIS REPORT")
        lines.append("=" * 50)

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
        lines.append(f"  Merge mode: {mode}")
        if mode == "ties":
            lines.append(f"  Density: {density:.2f}")
            lines.append(f"  Sign method: {sign_method}")

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

    def optimize_merge(self, model, clip, lora_stack, output_strength, clip_strength_multiplier=1.0, auto_strength="disabled"):
        """
        Main entry point. Three phases:
        1. Compute diffs + collect per-LoRA stats
        2. Analyze: sign conflicts, magnitude ratio -> auto-select params
        3. Merge with chosen params, build report
        """
        # Normalize stack format (standard tuples or ZImageLoRAStack dicts)
        if not lora_stack or len(lora_stack) == 0:
            return (model, clip, "No LoRAs in stack.")

        normalized_stack = self._normalize_stack(lora_stack)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]

        if len(active_loras) == 0:
            return (model, clip, "No LoRAs in stack (all zero strength or malformed).")

        # Single LoRA: skip analysis, apply directly via ComfyUI's standard
        # additive LoRA application (faster than diff-based pipeline).
        # auto_strength is a no-op with a single LoRA (scale would be 1.0).
        if len(active_loras) == 1:
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
                "Z-IMAGE LORA OPTIMIZER - ANALYSIS REPORT\n"
                "=" * 50 + "\n\n"
                "Single LoRA detected — bypassing analysis.\n"
                f"  Name: {item['name']}\n"
                f"  Strength: {strength}\n"
                f"  Applied directly with output_strength={output_strength}\n"
                "\n" + "=" * 50
            )
            return (new_model, new_clip, report)

        logging.info(f"[Z-Image Optimizer] Starting analysis of {len(active_loras)} LoRAs")
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

        # Phase 1: Compute diffs and collect per-LoRA stats
        # all_key_diffs[lora_prefix] = list of (diff, strength, lora_index) triples
        #   lora_index stripped to (diff, strength) pairs before Phase 3
        # per_lora_diffs[lora_prefix][lora_index] = diff  (for pairwise analysis)
        # all_key_targets[lora_prefix] = (target_key, is_clip)
        logging.info("[Z-Image Optimizer] Phase 1: Computing weight diffs...")
        logging.info(f"[Z-Image Optimizer]   {len(all_lora_prefixes)} key prefixes to analyze across {len(active_loras)} LoRAs")
        t_phase1 = time.time()
        all_key_diffs = {}
        all_key_targets = {}
        per_lora_diffs = {}
        skipped_keys = 0
        per_lora_stats = [{
            "name": item["name"],
            "strength": item["strength"],
            "ranks": [],
            "key_count": 0,
            "l2_norms": [],
        } for item in active_loras]

        estimated_diffs = len(active_loras) * len(all_lora_prefixes)
        if estimated_diffs > 2000:
            logging.warning(
                f"[Z-Image Optimizer] Analyzing up to {estimated_diffs} diff tensors. "
                "This may use significant memory. Consider ZImageLoRATrueMerge for large stacks."
            )

        compute_device = self._get_compute_device()
        logging.info(f"[Z-Image Optimizer]   Compute device: {compute_device}")

        max_workers = min(4, max(1, len(all_lora_prefixes)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_prefix, lora_prefix, active_loras,
                                model_keys, clip_keys, model, clip, compute_device): lora_prefix
                for lora_prefix in all_lora_prefixes
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                prefix, diffs, lora_diffs, partial_stats, target_info, skips = result
                if len(diffs) > 0:
                    all_key_diffs[prefix] = diffs
                    all_key_targets[prefix] = target_info
                    per_lora_diffs[prefix] = lora_diffs
                skipped_keys += skips
                for (idx, rank, l2) in partial_stats:
                    per_lora_stats[idx]["ranks"].append(rank)
                    per_lora_stats[idx]["key_count"] += 1
                    per_lora_stats[idx]["l2_norms"].append(l2)

        if len(all_key_diffs) == 0:
            return (model, clip, "No compatible LoRA keys found. "
                    "LoRAs may be incompatible with this model architecture.")

        # Log per-LoRA summaries
        total_diffs = sum(len(v) for v in all_key_diffs.values())
        for i, stat in enumerate(per_lora_stats):
            avg_r = sum(stat["ranks"]) / len(stat["ranks"]) if stat["ranks"] else 0
            logging.info(f"[Z-Image Optimizer]   {stat['name']} ({i+1}/{len(active_loras)}): "
                         f"{stat['key_count']} keys, avg rank {avg_r:.0f}")
        logging.info(f"[Z-Image Optimizer]   Total: {len(all_key_diffs)} prefixes, "
                     f"{total_diffs} diffs ({time.time() - t_phase1:.1f}s)")

        # Finalize per-LoRA stats
        lora_stats = []
        l2_means = []
        for stat in per_lora_stats:
            avg_rank = sum(stat["ranks"]) / len(stat["ranks"]) if stat["ranks"] else 0
            l2_mean = sum(stat["l2_norms"]) / len(stat["l2_norms"]) if stat["l2_norms"] else 0
            l2_means.append(l2_mean)
            lora_stats.append({
                "name": stat["name"],
                "strength": stat["strength"],
                "key_count": stat["key_count"],
                "avg_rank": avg_rank,
                "l2_mean": l2_mean,
            })

        # Phase 2: Pairwise sign conflict analysis
        logging.info("[Z-Image Optimizer] Phase 2: Analyzing conflicts...")
        t_phase2 = time.time()
        pairwise_conflicts = []
        total_overlap = 0
        total_conflict = 0

        pairs = [(i, j) for i in range(len(active_loras))
                         for j in range(i + 1, len(active_loras))]

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, max(1, len(pairs)))) as executor:
            futures = [
                executor.submit(self._process_pair, i, j, active_loras,
                                per_lora_diffs, compute_device)
                for i, j in pairs
            ]
            for future in futures:  # preserve submission order for deterministic reports
                pair_overlap, pair_conflict, ratio, pair_label = future.result()
                total_overlap += pair_overlap
                total_conflict += pair_conflict
                pairwise_conflicts.append({
                    "pair": pair_label,
                    "overlap": pair_overlap,
                    "conflicts": pair_conflict,
                    "ratio": ratio,
                })
                logging.info(f"[Z-Image Optimizer]   {pair_label} -> {ratio:.1%} conflict")

        avg_conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0
        logging.info(f"[Z-Image Optimizer]   Average conflict ratio: {avg_conflict_ratio:.1%} ({time.time() - t_phase2:.1f}s)")

        # Magnitude ratio
        valid_l2 = [m for m in l2_means if m > 0]
        if len(valid_l2) >= 2:
            magnitude_ratio = max(valid_l2) / min(valid_l2)
        else:
            magnitude_ratio = 1.0

        collection_stats = {
            "n_loras": len(active_loras),
            "total_keys": len(all_key_diffs),
            "avg_conflict": avg_conflict_ratio,
            "magnitude_ratio": magnitude_ratio,
        }

        # Auto-select parameters
        mode, density, sign_method, reasoning = self._auto_select_params(
            avg_conflict_ratio, magnitude_ratio, all_key_diffs
        )

        logging.info(f"[Z-Image Optimizer] Decision: {mode} (conflict {avg_conflict_ratio:.1%} "
                     f"{'>' if avg_conflict_ratio > 0.25 else '<='} 25% threshold)")
        if mode == "ties":
            logging.info(f"[Z-Image Optimizer]   density={density:.2f}, sign_method={sign_method}")

        # Drop analysis index (tensors remain live in all_key_diffs until Phase 3)
        del per_lora_diffs

        # Auto-strength adjustment: reduce per-LoRA strengths to prevent overexposure
        auto_strength_info = None
        if auto_strength == "enabled":
            new_strengths, strength_reasoning = self._compute_auto_strengths(active_loras, lora_stats)

            # Compute per-LoRA scale ratio so we can apply it to both model
            # and clip strengths (new_strengths only contains model strengths)
            scale_ratios = {}
            for i in range(len(active_loras)):
                orig = active_loras[i]["strength"]
                if abs(orig) > 1e-9:
                    scale_ratios[i] = new_strengths[i] / orig
                else:
                    scale_ratios[i] = 1.0

            # Rebuild all_key_diffs with scaled strengths, stripping lora_index
            for lora_prefix in all_key_diffs:
                all_key_diffs[lora_prefix] = [
                    (diff, old_strength * scale_ratios[idx])
                    for diff, old_strength, idx in all_key_diffs[lora_prefix]
                ]

            # Update lora_stats with new strengths
            for i, stat in enumerate(lora_stats):
                stat["original_strength"] = stat["strength"]
                stat["strength"] = new_strengths[i]

            auto_strength_info = {
                "reasoning": strength_reasoning,
                "original_strengths": [item["strength"] for item in active_loras],
                "new_strengths": new_strengths,
                "names": [item["name"] for item in active_loras],
            }

            logging.info(f"[Z-Image Optimizer] Auto-strength: {strength_reasoning[0]}")
            for i in range(len(active_loras)):
                logging.info(f"[Z-Image Optimizer]   {active_loras[i]['name']}: "
                             f"{active_loras[i]['strength']} -> {new_strengths[i]:.4f}")
        else:
            # Strip lora_index from triples -> (diff, strength) pairs for _merge_diffs
            for lora_prefix in all_key_diffs:
                all_key_diffs[lora_prefix] = [
                    (diff, strength)
                    for diff, strength, _idx in all_key_diffs[lora_prefix]
                ]

        # Phase 3: Merge using stored diffs
        logging.info(f"[Z-Image Optimizer] Phase 3: Merging {len(all_key_diffs)} keys...")
        t_phase3 = time.time()
        model_patches = {}
        clip_patches = {}
        processed_keys = 0

        for lora_prefix, diffs_list in all_key_diffs.items():
            target_key, is_clip_key = all_key_targets[lora_prefix]

            merged_diff = self._merge_diffs(
                diffs_list, mode,
                density=density, majority_sign_method=sign_method
            )

            if merged_diff is not None:
                if is_clip_key:
                    clip_patches[target_key] = ("diff", (merged_diff,))
                else:
                    model_patches[target_key] = ("diff", (merged_diff,))
                processed_keys += 1

        logging.info(f"[Z-Image Optimizer]   Model patches: {len(model_patches)}, "
                     f"CLIP patches: {len(clip_patches)} ({time.time() - t_phase3:.1f}s)")

        # Apply patches
        new_model = model
        new_clip = clip

        # If ALL LoRAs have explicit clip_strength (standard tuple format),
        # clip strengths are already baked into the diffs — skip global multiplier.
        # If ANY lack it (dict format), apply the multiplier for those LoRAs.
        # Mixed stacks: multiplier applies since dict-format LoRAs need it and
        # tuple-format LoRAs already baked their clip_strength into the diff weight.
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
            auto_strength_info=auto_strength_info
        )

        # Save report to disk for later reference
        cache_key = self._compute_cache_key(lora_stack, output_strength,
                                            clip_strength_multiplier, auto_strength)
        lora_combo = [[item["name"], item["strength"]] for item in active_loras]
        selected_params = {"mode": mode, "density": density, "sign_method": sign_method}
        report_path = self._save_report_to_disk(cache_key, lora_combo, auto_strength, report, selected_params)
        if report_path:
            logging.info(f"[Z-Image Optimizer] Report saved to: {report_path}")

        logging.info(f"[Z-Image Optimizer] Done! {processed_keys} keys processed ({time.time() - t_start:.1f}s total)")

        return (new_model, new_clip, report)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ZImageLoRAMerger": ZImageLoRAMerger,
    "ZImageLoRAStack": ZImageLoRAStack,
    "ZImageLoRAStackApply": ZImageLoRAStackApply,
    "ZImageLoRAMergeToSingle": ZImageLoRAMergeToSingle,
    "ZImageLoRATrueMerge": ZImageLoRATrueMerge,
    "ZImageLoRAOptimizer": ZImageLoRAOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoRAMerger": "Z-Image LoRA Merger",
    "ZImageLoRAStack": "Z-Image LoRA Stack",
    "ZImageLoRAStackApply": "Z-Image LoRA Stack Apply",
    "ZImageLoRAMergeToSingle": "Z-Image LoRA Merge to Single",
    "ZImageLoRATrueMerge": "Z-Image LoRA True Merge",
    "ZImageLoRAOptimizer": "Z-Image LoRA Optimizer",
}
