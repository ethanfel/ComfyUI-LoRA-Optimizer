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
import gc
import concurrent.futures
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.weight_adapter.lokr import LoKrAdapter
from comfy.weight_adapter.loha import LoHaAdapter
from safetensors.torch import save_file

TUNER_DATA_DIR = os.path.join(folder_paths.models_dir, "tuner_data")
os.makedirs(TUNER_DATA_DIR, exist_ok=True)
folder_paths.add_model_folder_path("tuner_data", TUNER_DATA_DIR)


# ---------------------------------------------------------------------------
# Architecture-aware threshold presets
# ---------------------------------------------------------------------------
# Each preset contains numeric thresholds used by density estimation, conflict
# detection, auto-strength, and scoring heuristics.  The preset is selected
# based on detected model architecture (or manual override).
_ARCH_PRESETS = {
    "sd_unet": {
        "density_noise_floor_ratio": 0.1,
        "density_clamp_min": 0.1,
        "density_clamp_max": 0.9,
        "dare_ideal_density": 0.7,
        "consensus_cos_sim_min": 0.5,
        "consensus_conflict_max": 0.15,
        "orthogonal_cos_sim_max": 0.25,
        "orthogonal_conflict_max": 0.60,
        "ties_conflict_threshold": 0.25,
        "magnitude_ratio_total_sign": 2.0,
        "alignment_threshold": 0.1,
        "suggested_max_strength_cap": 3.0,
        "auto_strength_orthogonal_floor": 0.85,
        "display_name": "SD/SDXL UNet",
        "full_rank": {
            "rank_threshold": 512,
            "disable_slerp_upgrade": True,
            "prefer_sum_orthogonal": True,
            "auto_strength_floor": 1.0,
        },
    },
    "dit": {
        "density_noise_floor_ratio": 0.05,
        "density_clamp_min": 0.4,
        "density_clamp_max": 0.95,
        "dare_ideal_density": 0.8,
        "consensus_cos_sim_min": 0.5,
        "consensus_conflict_max": 0.15,
        "orthogonal_cos_sim_max": 0.25,
        "orthogonal_conflict_max": 0.60,
        "ties_conflict_threshold": 0.25,
        "magnitude_ratio_total_sign": 2.0,
        "alignment_threshold": 0.1,
        "suggested_max_strength_cap": 5.0,
        "auto_strength_orthogonal_floor": 0.85,
        "display_name": "DiT (Flux/WAN/Z-Image/LTX/HunyuanVideo)",
        "full_rank": {
            "rank_threshold": 512,
            "disable_slerp_upgrade": True,
            "prefer_sum_orthogonal": True,
            "auto_strength_floor": 1.0,
        },
    },
    "llm": {
        "density_noise_floor_ratio": 0.15,
        "density_clamp_min": 0.1,
        "density_clamp_max": 0.8,
        "dare_ideal_density": 0.5,
        "consensus_cos_sim_min": 0.5,
        "consensus_conflict_max": 0.15,
        "orthogonal_cos_sim_max": 0.25,
        "orthogonal_conflict_max": 0.60,
        "ties_conflict_threshold": 0.25,
        "magnitude_ratio_total_sign": 2.0,
        "alignment_threshold": 0.1,
        "suggested_max_strength_cap": 3.0,
        "auto_strength_orthogonal_floor": 0.9,
        "display_name": "LLM (Qwen/LLaMA)",
        "full_rank": {
            "rank_threshold": 512,
            "disable_slerp_upgrade": True,
            "prefer_sum_orthogonal": True,
            "auto_strength_floor": 1.0,
        },
    },
}

_ARCH_TO_PRESET = {
    "sdxl": "sd_unet", "unknown": "sd_unet",
    "flux": "dit", "wan": "dit", "zimage": "dit", "ltx": "dit",
    "acestep": "dit",
    "qwen_image": "llm",
}

# Video architectures where orthogonal LoRAs teach independent motions —
# their energy should add naturally without scaling down.
_VIDEO_ARCH_ORTHOGONAL_FLOOR = {"wan": 1.0, "ltx": 1.0}


def _resolve_arch_preset(arch_override, detected_arch):
    """Resolve architecture preset from override or detected architecture."""
    if arch_override and arch_override != "auto" and arch_override in _ARCH_PRESETS:
        key = arch_override
    else:
        key = _ARCH_TO_PRESET.get(detected_arch, "sd_unet")
    return key, _ARCH_PRESETS[key]


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
                "key_filter": (["all", "shared_only", "unique_only"], {
                    "default": "all",
                    "tooltip": "Filter which keys this LoRA contributes based on how many LoRAs share each key. "
                               "'all': contribute all keys (default). "
                               "'shared_only': only contribute to keys present in 2+ LoRAs. "
                               "'unique_only': only contribute to keys present in exactly 1 LoRA."
                }),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Connect another LoRA Stack node here to chain multiple LoRAs together."}),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "add_to_stack"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Adds a LoRA to the stack for use with LoRA Optimizer"

    def add_to_stack(self, lora_name, strength, conflict_mode="all", key_filter="all", lora_stack=None):
        lora_list = list(lora_stack) if lora_stack else []

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        lora_list.append({
            "name": lora_name,
            "lora": lora,
            "strength": strength,
            "conflict_mode": conflict_mode,
            "key_filter": key_filter,
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
                "settings_visibility": (["simple", "advanced"], {
                    "tooltip": "Simple: one strength slider per LoRA. "
                               "Advanced: separate model/clip strength, conflict mode, and key filter per LoRA."
                }),
                "input_mode": (["dropdown", "text"], {
                    "default": "dropdown",
                    "tooltip": "Dropdown: pick LoRAs from a list. "
                               "Text: type LoRA names or connect text nodes (short names are auto-resolved)."
                }),
                "lora_count": ("INT", {"default": 3, "min": 1, "max": cls.MAX_LORAS, "step": 1,
                                       "tooltip": "How many LoRA slots to show. Increase to add more LoRAs."}),
            }
        }
        for i in range(1, cls.MAX_LORAS + 1):
            inputs["required"][f"lora_name_{i}"] = (loras, {
                "tooltip": f"LoRA #{i} — pick a LoRA file or leave as 'None' to skip this slot."
            })
            inputs["required"][f"lora_name_text_{i}"] = ("STRING", {
                "default": "None",
                "tooltip": f"LoRA #{i} — type a LoRA filename (without extension or path), "
                           f"a full relative path, or 'None' to skip. Accepts text node connections."
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
            inputs["required"][f"key_filter_{i}"] = (["all", "shared_only", "unique_only"], {
                "default": "all",
                "tooltip": f"LoRA #{i} key filter. "
                           f"'all': contribute all keys (default). "
                           f"'shared_only': only contribute to keys present in 2+ LoRAs. "
                           f"'unique_only': only contribute to keys present in exactly 1 LoRA."
            })
        inputs["optional"] = {
            "lora_stack": ("LORA_STACK", {"tooltip": "Connect another LoRA Stack node here to add even more LoRAs to the list."}),
            "base_model_filter": (["All"], {
                "default": "All",
                "tooltip": "Filter LoRA dropdowns by base model type (dropdown mode only). Requires ComfyUI-Lora-Manager installed."
            }),
        }
        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Dynamic LoRA stacker with adjustable slot count and optional per-LoRA CLIP strength"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # base_model_filter options are populated dynamically by JS from Lora Manager
        return True

    @staticmethod
    def _resolve_lora_name(name):
        """Resolve a short LoRA name to its full relative path.

        Accepts full relative paths (returned as-is) or short names like
        'Milena_20260216180133' which are matched against installed LoRAs
        by filename (with or without extension).
        """
        lora_list = folder_paths.get_filename_list("loras")

        # Exact match (full relative path)
        if name in lora_list:
            return name

        # Match by filename stem (without extension)
        name_lower = name.lower()
        for lora_path in lora_list:
            filename = os.path.basename(lora_path)
            stem = os.path.splitext(filename)[0]
            if stem.lower() == name_lower:
                return lora_path

        # Match by filename with extension
        for lora_path in lora_list:
            filename = os.path.basename(lora_path)
            if filename.lower() == name_lower:
                return lora_path

        # Substring match on filename stem (if unique)
        matches = []
        for lora_path in lora_list:
            stem = os.path.splitext(os.path.basename(lora_path))[0]
            if name_lower in stem.lower():
                matches.append(lora_path)
        if len(matches) == 1:
            return matches[0]

        logger = logging.getLogger("LoRAOptimizer")
        if len(matches) > 1:
            logger.warning(f"LoRA name '{name}' matches multiple files: {matches[:5]}... using first match")
            return matches[0]

        logger.warning(f"LoRA name '{name}' not found in installed LoRAs, skipping")
        return None

    def build_stack(self, settings_visibility, input_mode, lora_count, lora_stack=None, base_model_filter=None, **kwargs):
        loras = []
        use_text = input_mode == "text"
        for i in range(1, lora_count + 1):
            if use_text:
                name = kwargs.get(f"lora_name_text_{i}", "None")
            else:
                name = kwargs.get(f"lora_name_{i}", "None")
            if name == "None" or not name.strip():
                continue
            resolved = self._resolve_lora_name(name.strip())
            if not resolved:
                continue
            conflict_mode = kwargs.get(f"conflict_mode_{i}", "all")
            kf = kwargs.get(f"key_filter_{i}", "all")
            if settings_visibility == "simple":
                wt = kwargs.get(f"strength_{i}", 1.0)
                loras.append((resolved, wt, wt, conflict_mode, kf))
            else:
                model_str = kwargs.get(f"model_strength_{i}", 1.0)
                clip_str = kwargs.get(f"clip_strength_{i}", 1.0)
                loras.append((resolved, model_str, clip_str, conflict_mode, kf))

        # Chained lora_stack entries
        if lora_stack is not None:
            for l in lora_stack:
                if isinstance(l, dict):
                    if l.get("name", "None") != "None":
                        s = l.get("strength", 1.0)
                        cm = l.get("conflict_mode", "all")
                        kf = l.get("key_filter", "all")
                        loras.append((l["name"], s, s, cm, kf))
                elif isinstance(l, (tuple, list)):
                    if l[0] != "None":
                        loras.append(tuple(l))
                else:
                    loras.append(l)
        return (loras,)


class _DiffCache:
    """Cache for raw LoRA diffs across AutoTuner candidates.

    Modes:
      - "ram": All entries in RAM (fastest, most memory).
      - "disk": All entries on disk via torch.save/mmap (slowest, least memory).
      - "auto": RAM up to ram_pct of free system memory, then spills to disk.
    """

    def __init__(self, mode="auto", ram_pct=0.5):
        self.mode = mode
        self._ram_store = {}
        self._disk_store = {}
        self._prefetch_buf = {}
        self._prefetch_thread = None
        self._ram_bytes = 0
        self._ram_limit = None
        self._cache_dir = None
        self._disk_failed = False
        if mode in ("disk", "auto"):
            import tempfile
            self._cache_dir = tempfile.mkdtemp(prefix="lora_diff_cache_",
                                                   dir=folder_paths.get_temp_directory())
        if mode == "auto":
            try:
                import psutil
                self._ram_limit = int(psutil.virtual_memory().available * ram_pct)
            except ImportError:
                # psutil not available — use 4GB as fallback
                self._ram_limit = 4 * 1024 * 1024 * 1024

    def _use_disk(self, tensor_bytes):
        if self.mode == "disk":
            return True
        if self.mode == "ram":
            return False
        # auto: spill to disk if RAM limit would be exceeded
        return (self._ram_bytes + tensor_bytes) > self._ram_limit

    def prefetch(self, keys):
        """Load disk entries into a prefetch buffer in a background thread."""
        self._wait_prefetch()
        self._prefetch_buf.clear()
        disk_keys = [k for k in keys if k in self._disk_store]
        if not disk_keys:
            return
        import threading
        def _load():
            for k in disk_keys:
                try:
                    import numpy as np
                    arr = np.load(self._disk_store[k], mmap_mode='r')
                    self._prefetch_buf[k] = torch.from_numpy(arr.copy())
                except Exception:
                    pass
        self._prefetch_thread = threading.Thread(target=_load, daemon=True)
        self._prefetch_thread.start()

    def _wait_prefetch(self):
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None

    def get(self, key, device=None):
        self._wait_prefetch()
        # Check prefetch buffer first
        if key in self._prefetch_buf:
            val = self._prefetch_buf.pop(key)
            return val.to(device) if device is not None else val
        if key in self._ram_store:
            val = self._ram_store[key]
            return val.to(device) if device is not None else val.clone()
        if key in self._disk_store:
            import numpy as np
            arr = np.load(self._disk_store[key], mmap_mode='r')
            t = torch.from_numpy(arr.copy())
            return t.to(device) if device is not None else t
        return None

    def put(self, key, tensor):
        if key in self._ram_store or key in self._disk_store:
            return
        cached = tensor.detach().half().cpu()
        tensor_bytes = cached.nelement() * cached.element_size()
        if self._use_disk(tensor_bytes) and not self._disk_failed:
            try:
                import hashlib, numpy as np
                name_hash = hashlib.sha256(f"{key[0]}_{key[1]}".encode()).hexdigest()[:16]
                path = os.path.join(self._cache_dir, f"{name_hash}.npy")
                np.save(path, cached.numpy())
                self._disk_store[key] = path
                return
            except Exception as e:
                self._disk_failed = True
                logging.warning(f"[DiffCache] Disk cache failed, falling back to RAM — {e}")
        # RAM storage (also used as fallback when disk fails)
        self._ram_store[key] = cached.clone()
        self._ram_bytes += tensor_bytes

    def size_mb(self):
        total = self._ram_bytes
        for path in self._disk_store.values():
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
        return total / (1024 * 1024)

    def clear(self):
        self._wait_prefetch()
        self._prefetch_buf.clear()
        self._ram_store.clear()
        self._disk_store.clear()
        self._ram_bytes = 0
        if self._cache_dir is not None:
            import shutil, tempfile
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = tempfile.mkdtemp(prefix="lora_diff_cache_",
                                                   dir=folder_paths.get_temp_directory())

    def __contains__(self, key):
        return key in self._ram_store or key in self._disk_store


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
        Returns: 'zimage', 'flux', 'wan', 'acestep', 'sdxl', 'ltx', 'qwen_image', or 'unknown'.
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

        # ACE-Step: layers.N with self_attn/cross_attn using q_proj/k_proj/v_proj
        if any('layers.' in k and ('self_attn' in k or 'cross_attn' in k)
               and any(x in k for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
               for k in keys):
            return 'acestep'

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

        # Note: RS-LoRA compensation removed. RS-LoRA files omit alpha and
        # rely on sqrt(rank) scaling, but we can't distinguish them from
        # standard PEFT LoRAs that also omit alpha. False positives cause
        # ~5.66x weight amplification (rank 32). When alpha is missing,
        # _get_lora_key_info defaults alpha=rank (scale=1.0), which is
        # correct for standard LoRAs and only slightly weak for RS-LoRA.

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
                if all(isinstance(p, tuple) and p[0] == "diff" for p in [q_patch, k_patch, v_patch]):
                    # Full-rank diff patch: ("diff", (tensor,))
                    q_diff = q_patch[1][0]
                    k_diff = k_patch[1][0]
                    v_diff = v_patch[1][0]
                    fused_diff = torch.cat([q_diff, k_diff, v_diff], dim=0)
                    fused[fused_key] = ("diff", (fused_diff,))
                elif all(isinstance(p, LoRAAdapter) for p in [q_patch, k_patch, v_patch]):
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
                        store_dtype = q_data[0].dtype
                        for comp_data in [q_data, k_data, v_data]:
                            alpha = comp_data[2] if comp_data[2] is not None else float(comp_data[1].shape[0])
                            rank = comp_data[1].shape[0]
                            diff = torch.mm(comp_data[0].float(), comp_data[1].float()) * (alpha / rank)
                            parts.append(diff)
                        fused_diff = torch.cat(parts, dim=0)
                        # Downcast back to original dtype to avoid doubling memory/file size
                        if store_dtype not in (torch.float32, torch.float64):
                            fused_diff = fused_diff.to(store_dtype)
                        fused[fused_key] = ("diff", (fused_diff,))
                elif any(hasattr(p, 'weights') for p in [q_patch, k_patch, v_patch]):
                    # LoKr/LoHa or mixed adapter types: expand to diff, then fuse
                    parts = []
                    # Try to detect the native dtype before float32 expansion
                    store_dtype = torch.float16
                    for p in [q_patch, k_patch, v_patch]:
                        if hasattr(p, 'weights') and p.weights[0] is not None:
                            dt = p.weights[0].dtype
                            if dt not in (torch.float32, torch.float64):
                                store_dtype = dt
                                break
                    for comp_patch in [q_patch, k_patch, v_patch]:
                        parts.append(self._expand_patch_to_diff(comp_patch))
                    fused_diff = torch.cat(parts, dim=0)
                    if store_dtype not in (torch.float32, torch.float64):
                        fused_diff = fused_diff.to(store_dtype)
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
    
    def _get_model_keys(self, model):
        """Get LoRA prefix → target key mapping for the model."""
        if model is None:
            return {}
        return comfy.lora.model_lora_keys_unet(model.model, {})

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
    def _dare_sparsify(tensor, density, generator=None, dampening=0.0):
        """
        DARE sparsification: randomly drop parameters and rescale survivors.
        Each element is kept with probability `density`, then rescaled by 1/q.
        With dampening=0: q=density (standard DARE). With dampening>0: q is
        interpolated toward 1.0, reducing noise amplification (DAREx, ICLR 2025).
        """
        if density >= 1.0:
            return tensor
        mask = torch.bernoulli(
            torch.full(tensor.shape, density, dtype=tensor.dtype, device=tensor.device),
            generator=generator
        )
        q = density + dampening * (1.0 - density)
        return tensor * mask * (1.0 / q)

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
    def _dare_sparsify_conflict(tensor, conflict_mask, density, generator=None, dampening=0.0):
        if density >= 1.0:
            return tensor
        rand_mask = torch.bernoulli(
            torch.full(tensor.shape, density, dtype=tensor.dtype, device=tensor.device),
            generator=generator
        )
        q = density + dampening * (1.0 - density)
        return torch.where(conflict_mask, tensor * rand_mask * (1.0 / q), tensor)

    @staticmethod
    def _della_sparsify_conflict(tensor, conflict_mask, density, epsilon=0.3, generator=None):
        if density >= 1.0:
            return tensor
        della_result = _LoRAMergeBase._della_sparsify(tensor, density, epsilon, generator)
        return torch.where(conflict_mask, della_result, tensor)

    @staticmethod
    def _estimate_patch_memory(patches_dict):
        """Estimate total bytes used by patch tensors in an add_patches-style dict."""
        total = 0
        for v in patches_dict.values():
            # LoRAAdapter: .weights = (mat_up, mat_down, alpha, ...)
            # diff patch:  ("diff", (tensor,))
            data = v.weights if hasattr(v, 'weights') else v
            if isinstance(data, (tuple, list)):
                for item in data:
                    if isinstance(item, torch.Tensor):
                        total += item.nelement() * item.element_size()
                    elif isinstance(item, (tuple, list)):
                        for sub in item:
                            if isinstance(sub, torch.Tensor):
                                total += sub.nelement() * sub.element_size()
        return total

    @staticmethod
    def _estimate_single_patch_bytes(patch):
        """Estimate byte size of a single patch entry (diff tuple or LoRAAdapter)."""
        total = 0
        data = patch.weights if hasattr(patch, 'weights') else patch
        if isinstance(data, (tuple, list)):
            for item in data:
                if isinstance(item, torch.Tensor):
                    total += item.nelement() * item.element_size()
                elif isinstance(item, (tuple, list)):
                    for sub in item:
                        if isinstance(sub, torch.Tensor):
                            total += sub.nelement() * sub.element_size()
        return total

    @staticmethod
    def _expand_patch_to_diff(patch):
        """Expand a patch (diff tuple, LoRAAdapter, LoKrAdapter, LoHaAdapter) to a float32 diff tensor."""
        if isinstance(patch, tuple) and patch[0] == "diff":
            return patch[1][0].float()
        elif isinstance(patch, LoKrAdapter):
            v = patch.weights
            # weights = (w1, w2, alpha, w1_a, w1_b, w2_a, w2_b, t2, dora_scale)
            w1, w2, alpha = v[0], v[1], v[2]
            w1_a, w1_b = v[3], v[4]
            w2_a, w2_b, t2 = v[5], v[6], v[7]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            dim = None
            if w1 is None:
                dim = w1_b.shape[0]
                w1 = torch.mm(w1_a.float(), w1_b.float())
            else:
                w1 = w1.float()
            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    w2 = torch.mm(w2_a.float(), w2_b.float())
                else:
                    w2 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        t2.float(), w2_b.float(), w2_a.float(),
                    )
            else:
                w2 = w2.float()
            if len(w2.shape) == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)
            scale = alpha / dim if (alpha is not None and dim is not None) else 1.0
            return torch.kron(w1, w2) * scale
        elif isinstance(patch, LoHaAdapter):
            v = patch.weights
            # weights = (w1a, w1b, alpha, w2a, w2b, t1, t2, dora_scale)
            w1a, w1b, alpha = v[0], v[1], v[2]
            w2a, w2b = v[3], v[4]
            t1, t2 = v[5], v[6]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            rank = w1b.shape[0]
            if t1 is not None:
                m1 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    t1.float(), w1b.float(), w1a.float(),
                )
                m2 = torch.einsum(
                    "i j k l, j r, i p -> p r k l",
                    t2.float(), w2b.float(), w2a.float(),
                )
            else:
                m1 = torch.mm(w1a.float(), w1b.float())
                m2 = torch.mm(w2a.float(), w2b.float())
            scale = alpha / rank if alpha is not None else 1.0
            return (m1 * m2) * scale
        elif hasattr(patch, 'weights'):
            # LoRAAdapter fallback
            w = patch.weights
            mat_up, mat_down, alpha = w[0], w[1], w[2]
            mid = w[3]
            rank = mat_down.shape[0]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            if mid is not None:
                final_shape = [mat_down.shape[1], mat_down.shape[0], mid.shape[2], mid.shape[3]]
                mat_down = torch.mm(
                    mat_down.transpose(0, 1).flatten(start_dim=1).float(),
                    mid.transpose(0, 1).flatten(start_dim=1).float(),
                ).reshape(final_shape).transpose(0, 1)
            diff = torch.mm(
                mat_up.flatten(start_dim=1).float(),
                mat_down.flatten(start_dim=1).float(),
            )
            return diff * (alpha / rank)
        raise ValueError(f"Unknown patch format: {type(patch)}")

    @staticmethod
    def _move_patch_to_device(patch, device):
        """Move all tensors in a patch to the given device. Returns new patch."""
        if hasattr(patch, 'weights'):
            inner = patch.weights
            moved = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t
                for t in inner
            )
            if isinstance(patch, LoKrAdapter):
                return LoKrAdapter(patch.loaded_keys, moved)
            elif isinstance(patch, LoHaAdapter):
                return LoHaAdapter(patch.loaded_keys, moved)
            return LoRAAdapter(patch.loaded_keys, moved)
        elif isinstance(patch, tuple) and len(patch) == 2 and isinstance(patch[0], str):
            # ("diff", (tensor,))
            moved_inner = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t
                for t in patch[1]
            )
            return (patch[0], moved_inner)
        return patch

    @staticmethod
    def _update_model_size(patcher, patches_dict):
        """Update a ModelPatcher's reported size to include patch memory.

        ComfyUI's model_size() only counts base model weights, making patches
        invisible to the memory manager. This causes large patched models to
        never be evicted when other models need RAM. By updating the cached
        size, ComfyUI's eviction logic sees the true memory footprint.
        """
        patch_bytes = _LoRAMergeBase._estimate_patch_memory(patches_dict)
        if patch_bytes > 0 and hasattr(patcher, 'model_size'):
            patcher.size = patcher.model_size() + patch_bytes
            logging.debug(f"[LoRA Optimizer] Updated model size: +{patch_bytes / (1024**2):.0f}MB patches")

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
    def _columnwise_elect_sign(trimmed_diffs, method="frequency"):
        """
        Column-wise sign election: each output neuron (row) votes as a unit
        instead of each element voting independently. For Conv2d tensors,
        [c_out, c_in, kH, kW] is reshaped to [c_out, -1] so each output
        channel votes together.

        Falls back to element-wise for 1D tensors (biases, norms).
        """
        ref = trimmed_diffs[0]
        if ref.dim() < 2:
            return _LoRAMergeBase._ties_elect_sign(trimmed_diffs, method)

        out_dim = ref.shape[0]
        total = torch.zeros(out_dim, device=ref.device, dtype=torch.float32)
        for d in trimmed_diffs:
            row_vals = d.reshape(out_dim, -1).to(dtype=torch.float32)
            if method == "total":
                total.add_(row_vals.sum(dim=1))
            else:
                total.add_(row_vals.sum(dim=1).sign())
        majority = torch.where(total >= 0, 1.0, -1.0)
        return majority.reshape(-1, *([1] * (ref.dim() - 1))).expand_as(ref)

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
    def _columnwise_disjoint_merge(trimmed_diffs, weights, majority_sign):
        """
        Column-wise disjoint merge: a row contributes entirely if its dominant
        sign matches the majority. Falls back to element-wise for 1D tensors.
        """
        ref = trimmed_diffs[0]
        if ref.dim() < 2:
            return _LoRAMergeBase._ties_disjoint_merge(trimmed_diffs, weights, majority_sign)

        out_dim = ref.shape[0]
        row_majority = majority_sign.reshape(out_dim, -1)[:, 0]

        result = torch.zeros_like(ref, dtype=torch.float32)
        contributor_count = torch.zeros(out_dim, device=ref.device, dtype=torch.float32)

        for diff, weight in zip(trimmed_diffs, weights):
            diff_f = diff.to(dtype=torch.float32)
            row_sign = diff_f.reshape(out_dim, -1).sum(dim=1).sign()
            sign_match = (row_sign * row_majority) > 0
            mask = sign_match.reshape(-1, *([1] * (ref.dim() - 1))).expand_as(ref)
            result.add_(torch.where(mask, diff_f * weight, torch.zeros_like(diff_f)))
            contributor_count.add_(sign_match.float())

        contributor_count.clamp_(min=1.0)
        return result.div_(contributor_count.reshape(-1, *([1] * (ref.dim() - 1))))

    @staticmethod
    @torch.no_grad()
    def _tall_masks(diffs_with_weights, lambda_threshold=1.0):
        """
        TALL-masks: identify per-parameter importance. "Selfish" weights
        (important to only one LoRA) are protected from merge averaging.
        Returns (consensus_diffs, selfish_additions) where selfish_additions
        is a tensor to add back after merging, or None if no selfish weights.
        """
        if len(diffs_with_weights) < 2:
            return diffs_with_weights, None

        ref = diffs_with_weights[0][0]

        # Tentative weighted sum
        d_merged = torch.zeros_like(ref, dtype=torch.float32)
        for d, w in diffs_with_weights:
            d_merged.add_(d.to(dtype=torch.float32) * w)

        # Per-LoRA importance masks
        masks = []
        for d, w in diffs_with_weights:
            contribution = d.to(dtype=torch.float32) * w
            others = d_merged - contribution
            mask = contribution.abs() >= others.abs() * lambda_threshold
            masks.append(mask)

        # Agreement count per position
        agreement = torch.zeros_like(ref, dtype=torch.float32)
        for m in masks:
            agreement.add_(m.float())

        # Separate selfish (agreement==1) from consensus
        selfish_additions = torch.zeros_like(ref, dtype=torch.float32)
        consensus_diffs = []
        has_selfish = False
        for i, (d, w) in enumerate(diffs_with_weights):
            selfish_mask = masks[i] & (agreement == 1)
            if selfish_mask.any():
                selfish_additions.add_(d.to(dtype=torch.float32) * w * selfish_mask.float())
                has_selfish = True
            consensus_d = d.clone()
            consensus_d[selfish_mask] = 0
            consensus_diffs.append((consensus_d, w))

        return consensus_diffs, selfish_additions if has_selfish else None

    @staticmethod
    @torch.no_grad()
    def _do_orthogonalize(diffs_with_weights):
        """
        DO-Merging: Decouple & Orthogonalize direction vectors while preserving magnitudes.
        Reduces directional interference between LoRA diffs by applying Modified Gram-Schmidt
        orthogonalization on unit direction vectors, then recombining with original magnitudes.

        Paper: arxiv 2505.15875 (May 2025)
        """
        if len(diffs_with_weights) < 2:
            return diffs_with_weights

        first = diffs_with_weights[0][0]
        if first.dim() < 2:
            return diffs_with_weights

        # Flatten, decompose into magnitude and unit direction
        dtype = first.dtype
        shapes = [d.shape for d, _ in diffs_with_weights]
        flat = [d.to(dtype=torch.float32).flatten() for d, _ in diffs_with_weights]
        weights = [w for _, w in diffs_with_weights]

        magnitudes = []
        directions = []
        for v in flat:
            mag = v.norm()
            magnitudes.append(mag)
            if mag > 1e-8:
                directions.append(v / mag)
            else:
                directions.append(torch.zeros_like(v))
        del flat

        # Modified Gram-Schmidt orthogonalization (numerically stable)
        ortho = []
        for i, d in enumerate(directions):
            q = d.clone()
            for j in range(len(ortho)):
                proj = torch.dot(q, ortho[j])
                q = q - proj * ortho[j]
            q_norm = q.norm()
            if q_norm > 1e-8:
                q = q / q_norm
            else:
                q = torch.zeros_like(q)
            ortho.append(q)

        # Recombine: orthogonalized direction * original magnitude
        result = []
        for i in range(len(diffs_with_weights)):
            recombined = (ortho[i] * magnitudes[i]).to(dtype=dtype).reshape(shapes[i])
            result.append((recombined, weights[i]))

        return result

    @staticmethod
    @torch.no_grad()
    def _knots_align(diffs_with_weights, compute_device=None, svd_device=None):
        """
        KnOTS SVD alignment: project LoRA diffs into a shared SVD basis
        for better comparability before merging. Concatenates all diffs
        column-wise, computes truncated SVD, then reconstructs each diff
        in the shared basis.

        For [4096, 4096] with 5 LoRAs → M is [4096, 20480] ≈ 320MB.
        SVD rank capped at 256. Falls back to CPU on OOM.
        """
        if len(diffs_with_weights) < 2:
            return diffs_with_weights

        ref = diffs_with_weights[0][0]
        if ref.dim() < 2 or min(ref.shape) < 2:
            return diffs_with_weights

        n = len(diffs_with_weights)
        out_dim = ref.shape[0]
        in_dim = ref.reshape(out_dim, -1).shape[1]
        original_shape = ref.shape
        dev = svd_device if svd_device is not None else (compute_device or ref.device)

        # Concatenate column-wise: [out, N*in]
        cols = [d.reshape(out_dim, -1).to(device=dev, dtype=torch.float32)
                for d, _ in diffs_with_weights]
        M = torch.cat(cols, dim=1)
        del cols

        rank = min(out_dim, n * in_dim, 256)
        try:
            U, S, V = torch.svd_lowrank(M, q=rank)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            if M.is_cuda:
                logging.warning("[LoRA Optimizer] KnOTS SVD OOM on GPU, falling back to CPU")
                torch.cuda.empty_cache()
                M = M.cpu()
                try:
                    U, S, V = torch.svd_lowrank(M, q=rank)
                except RuntimeError:
                    logging.warning("[LoRA Optimizer] KnOTS SVD also failed on CPU, skipping alignment")
                    del M
                    return diffs_with_weights
            else:
                return diffs_with_weights
        del M

        # After CPU fallback, SVD results are on CPU — ensure aligned diffs
        # return on the same device as the input diffs
        output_device = compute_device if compute_device is not None else ref.device

        # Reconstruct each diff in shared basis
        aligned = []
        US = U * S.unsqueeze(0)  # [out, rank]
        for i, (_, w) in enumerate(diffs_with_weights):
            Vi = V[i * in_dim:(i + 1) * in_dim, :]  # [in, rank]
            aligned_diff = (US @ Vi.T).reshape(original_shape)
            if aligned_diff.device != output_device:
                aligned_diff = aligned_diff.to(output_device)
            aligned.append((aligned_diff, w))
        del U, S, V, US

        return aligned

    @staticmethod
    @torch.no_grad()
    def _compress_to_lowrank(diff, rank, svd_device=None, output_dtype=None):
        """
        Re-compress a full-rank diff tensor to low-rank via truncated SVD.
        Returns ("lora", (mat_up, mat_down, alpha=rank, None)) so ComfyUI
        computes up @ down * (rank/rank) = up @ down (no extra scaling).

        svd_device: where to run SVD. GPU is ~10-50x faster. CPU if None.
        output_dtype: cast output to this dtype. None = same as input (fp16 stays fp16).
        For a [4096, 4096] diff at rank 128: 64MB → 2MB (~32x reduction).
        """
        original_shape = diff.shape
        if output_dtype is None:
            output_dtype = diff.dtype
        # Reshape to 2D for SVD: [out_features, in_features]
        mat = diff.reshape(original_shape[0], -1).float()
        rank = min(rank, min(mat.shape))

        # Move to requested device for SVD (GPU is much faster for matmul-heavy randomized SVD)
        if svd_device is not None and mat.device != svd_device:
            mat = mat.to(svd_device)
        # When rank > dim/2, full SVD + truncate is both faster and more accurate
        # than randomized SVD (which would oversample beyond dim anyway).
        if rank > min(mat.shape) // 2:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            del mat
            U = U[:, :rank]
            S = S[:rank]
            V = Vh[:rank, :].T  # Vh is [min(dim), n], transpose slice to [n, rank]
            del Vh
        else:
            # Oversample by 20% for better randomized SVD accuracy, then truncate.
            q_oversample = min(rank + max(20, rank // 5), min(mat.shape))
            U, S, V = torch.svd_lowrank(mat, q=q_oversample, niter=4)
            del mat
            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]
        # U: [out, rank], S: [rank], V: [in, rank]
        # Reconstruct as: mat_up = U * sqrt(S), mat_down = sqrt(S) * V^T
        # Return on CPU for storage (ComfyUI moves to device when applying)
        sqrt_S = S.sqrt()
        mat_up = (U * sqrt_S.unsqueeze(0)).to(output_dtype).cpu()    # [out, rank]
        mat_down = ((V * sqrt_S.unsqueeze(0)).T).to(output_dtype).cpu()  # [rank, in]
        del U, S, V, sqrt_S
        # alpha=rank so ComfyUI computes: up @ down * (rank/rank) = up @ down
        return LoRAAdapter(set(), (mat_up, mat_down, float(rank), None, None, None))

    @staticmethod
    @torch.no_grad()
    def _estimate_save_rank(initial_rank, model_patches, clip_patches,
                            max_error=0.05, n_samples=3):
        """
        Determine the exact SVD rank needed to reconstruct merged diffs within
        `max_error` relative Frobenius error.  Uses exact SVD on a few sample
        diffs to find the minimum rank where the truncation error drops below
        the threshold.  No arbitrary cap — rank is bounded only by min(dim).
        """
        # Collect sample full-rank diffs (prefer larger ones for conservative estimate)
        samples = []
        for patch in list(model_patches.values()) + list(clip_patches.values()):
            if isinstance(patch, tuple) and patch[0] == "diff":
                samples.append(patch[1][0])
                if len(samples) >= n_samples:
                    break
        if not samples:
            return max(initial_rank, 64)

        rank = max(initial_rank, 64)
        for sample in samples:
            mat = sample.reshape(sample.shape[0], -1).float()
            # Exact SVD gives precise singular values — no randomized noise
            S = torch.linalg.svdvals(mat)
            total_sq = (S ** 2).sum().item()
            if total_sq == 0:
                continue
            # Find minimum rank where residual/total < max_error
            # ||A - A_k||_F / ||A||_F < max_error  ⟺  residual_sq/total_sq < max_error^2
            threshold_sq = max_error * max_error * total_sq
            cumulative_sq = 0.0
            needed = len(S)
            for r in range(len(S)):
                cumulative_sq += S[r].item() ** 2
                if total_sq - cumulative_sq <= threshold_sq:
                    needed = r + 1
                    break
            if needed > rank:
                error_at_initial = math.sqrt(max(0, total_sq - sum(s.item()**2 for s in S[:rank])) / total_sq)
                logging.info(f"[Save Merged LoRA] Sample diff ({mat.shape[0]}x{mat.shape[1]}): "
                             f"need rank {needed} for <{max_error:.0%} error "
                             f"(rank {rank} has {error_at_initial:.1%} error)")
                rank = needed
            del S, mat
        return rank

    @torch.no_grad()
    def _merge_diffs(self, diffs_with_weights, mode, density=0.5, majority_sign_method="frequency",
                     compute_device=None, sparsification="disabled",
                     sparsification_density=0.7, sparsification_generator=None,
                     merge_quality="standard", dare_dampening=0.0,
                     keep_on_gpu=False):
        """
        Merges a list of diffs with their weights.
        When compute_device is given, tensors are moved there for faster ops,
        then the result is returned on CPU (unless keep_on_gpu=True).
        """
        if len(diffs_with_weights) == 0:
            return None

        if len(diffs_with_weights) == 1:
            diff, weight = diffs_with_weights[0]
            result = diff * weight
            if compute_device is not None and compute_device.type != "cpu" and result.is_cuda and not keep_on_gpu:
                return result.cpu()
            return result

        # All diffs should have the same shape (verified during computation)
        ref_diff = diffs_with_weights[0][0]
        dtype = ref_diff.dtype
        dev = compute_device if compute_device is not None else ref_diff.device
        to_cpu = (compute_device is not None and compute_device.type != "cpu"
                  and not keep_on_gpu)

        # DARE/DELLA preprocessing for non-TIES modes
        # (TIES replaces its trim step instead — handled in the ties branch)
        if sparsification != "disabled" and mode != "ties":
            is_conflict = sparsification in ("dare_conflict", "della_conflict")

            if is_conflict:
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    diffs_with_weights[idx] = (diff.to(device=dev, dtype=torch.float32), weight)
                conflict_mask = self._compute_conflict_mask(diffs_with_weights)

                is_dare = sparsification == "dare_conflict"
                sparsify_fn = (self._dare_sparsify_conflict if is_dare
                               else self._della_sparsify_conflict)
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    kwargs = dict(generator=sparsification_generator)
                    if is_dare:
                        kwargs["dampening"] = dare_dampening
                    diff = sparsify_fn(diff, conflict_mask, sparsification_density,
                                       **kwargs)
                    diffs_with_weights[idx] = (diff.to(dtype), weight)
                del conflict_mask
            else:
                is_dare = sparsification == "dare"
                sparsify_fn = (self._dare_sparsify if is_dare
                               else self._della_sparsify)
                for idx in range(len(diffs_with_weights)):
                    diff, weight = diffs_with_weights[idx]
                    diff = diff.to(device=dev, dtype=torch.float32)
                    kwargs = dict(generator=sparsification_generator)
                    if is_dare:
                        kwargs["dampening"] = dare_dampening
                    diff = sparsify_fn(diff, sparsification_density,
                                       **kwargs)
                    diffs_with_weights[idx] = (diff.to(dtype), weight)

        # Enhanced/maximum merge quality pipeline (non-TIES modes)
        # TIES has its own enhancement path below (after trim)
        # Order matters: TALL-masks must run BEFORE orthogonalization.
        # Orthogonalized diffs have uncorrelated element-wise distributions,
        # which causes TALL-masks to classify every position as "selfish"
        # (agreement=1 everywhere), zeroing out all consensus diffs.
        selfish_additions = None
        if merge_quality != "standard" and len(diffs_with_weights) >= 2 and mode != "ties":
            diffs_with_weights, selfish_additions = self._tall_masks(diffs_with_weights)
            first = diffs_with_weights[0][0]
            if first.dim() >= 2:
                diffs_with_weights = self._do_orthogonalize(diffs_with_weights)
            if merge_quality == "maximum":
                first = diffs_with_weights[0][0]
                if first.dim() >= 2 and min(first.shape) >= 2:
                    diffs_with_weights = self._knots_align(
                        diffs_with_weights, compute_device=dev, svd_device=dev)

        if mode == "weighted_average":
            result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            total_weight = sum(abs(w) for _, w in diffs_with_weights)
            if total_weight == 0:
                return result.to(dtype).cpu() if to_cpu else result.to(dtype)
            for idx in range(len(diffs_with_weights)):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                result.add_(diff.to(device=dev, dtype=torch.float32) * (weight / total_weight))
            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "weighted_sum":
            result = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            for idx in range(len(diffs_with_weights)):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                result.add_(diff.to(device=dev, dtype=torch.float32) * weight)
            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
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
            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "slerp":
            # Iterative pairwise SLERP — magnitude-preserving blend for N diffs.
            # For 2 diffs: equivalent to standard SLERP.
            # For 3+ diffs: iteratively SLERPs accumulated result with next diff,
            # sorted by descending |weight| so strongest LoRA anchors direction.
            # Final norm corrected to match weighted average of input norms.
            n_diffs = len(diffs_with_weights)

            # Handle negative weights by negating diff direction
            items = []
            for idx in range(n_diffs):
                diff, weight = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                v = diff.to(device=dev, dtype=torch.float32).flatten()
                del diff
                if weight < 0:
                    v = -v
                items.append((v, abs(weight)))

            # Sort by descending weight (strongest LoRA anchors direction)
            items.sort(key=lambda x: x[1], reverse=True)

            # All-zero weights: return zero (consistent with other modes)
            total_w = sum(w for _, w in items)
            if total_w == 0:
                z = torch.zeros(ref_diff.shape, device=dev, dtype=dtype)
                return z.cpu() if to_cpu else z

            # Pre-compute norms for later correction (before vectors are consumed)
            input_norms = [(v.norm().item(), w) for v, w in items]

            # Iterative pairwise SLERP
            acc_v, acc_w = items[0]
            items[0] = None  # Free tensor reference
            for k in range(1, n_diffs):
                next_v, next_w = items[k]
                items[k] = None  # Free tensor reference
                frac = next_w / (acc_w + next_w) if (acc_w + next_w) > 0 else 0.5

                # Compute angle between accumulated and next vector
                norm_acc = acc_v.norm()
                norm_next = next_v.norm()
                denom = norm_acc * norm_next
                if denom > 0:
                    cos_theta = (torch.dot(acc_v, next_v) / denom).clamp(-1.0, 1.0)
                else:
                    cos_theta = torch.tensor(1.0, device=dev)
                theta = torch.acos(cos_theta)

                # Nearly-parallel fallback to linear interpolation
                if theta.item() < 1e-6:
                    acc_v = (1.0 - frac) * acc_v + frac * next_v
                else:
                    sin_theta = torch.sin(theta)
                    a = torch.sin((1.0 - frac) * theta) / sin_theta
                    b = torch.sin(frac * theta) / sin_theta
                    acc_v = a * acc_v + b * next_v

                acc_w = acc_w + next_w
                del next_v
            del items

            # Norm correction: rescale to match weighted average of input norms
            target_norm = sum(n * w for n, w in input_norms) / total_w
            current_norm = acc_v.norm().item()
            if current_norm > 1e-8:
                acc_v = acc_v * (target_norm / current_norm)

            result = acc_v.reshape(ref_diff.shape)
            del acc_v
            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        elif mode == "consensus":
            # Consensus merge: Fisher-proxy + MAGIC calibration + MonoSoup spectral cleanup
            # Optimized for similar LoRAs (high cosine similarity, low conflict)

            # Step 1: Fisher-Proxy weighted merge
            # Weight each parameter by |diff|^2 as importance proxy
            numerator = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            denominator = torch.zeros(ref_diff.shape, dtype=torch.float32, device=dev)
            input_norms = []
            abs_weight_list = []

            for idx in range(len(diffs_with_weights)):
                d, w = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free early
                d_f = d.to(device=dev, dtype=torch.float32)
                del d
                importance = d_f.square()
                aw = abs(w)
                numerator.add_(d_f * w * importance)
                denominator.add_(aw * importance)
                input_norms.append(d_f.norm().item() * aw)
                abs_weight_list.append(aw)
                del d_f, importance

            # Safe division (zero importance → zero result)
            result = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
            del numerator, denominator

            # Step 2: MAGIC magnitude calibration
            # Rescale merged result so L2 norm matches weighted average of input norms
            merged_norm = result.norm().item()
            total_aw = sum(abs_weight_list)
            if total_aw > 0 and merged_norm > 1e-8:
                target_norm = sum(input_norms) / total_aw
                result.mul_(target_norm / merged_norm)
            del input_norms, abs_weight_list

            # Step 3: MonoSoup spectral cleanup (2D+ weights only)
            if result.dim() >= 2 and min(result.shape) >= 4:
                mat = result.reshape(result.shape[0], -1)
                try:
                    rank_budget = min(min(mat.shape), 128)
                    U, S, V = torch.svd_lowrank(mat, q=rank_budget)

                    # Entropy-based effective rank
                    s_sum = S.sum()
                    if s_sum < 1e-10:
                        del U, S, V, mat
                        raise RuntimeError("zero singular values")
                    p = S / s_sum
                    p = p.clamp(min=1e-10)  # avoid log(0)
                    entropy = -(p * p.log()).sum().item()
                    eff_rank = min(int(math.exp(entropy) + 0.5), rank_budget)
                    eff_rank = max(eff_rank, 1)

                    # Soft spectral gate: smooth transition instead of hard cutoff
                    gate = torch.sigmoid(4.0 * (torch.arange(rank_budget, device=dev, dtype=torch.float32) - eff_rank) * (-1.0 / max(eff_rank, 1)))
                    S_gated = S * gate

                    # Preserve original norm (spectral cleanup shouldn't change magnitude)
                    pre_norm = result.norm()
                    result = (U * S_gated.unsqueeze(0)) @ V.T
                    result = result.reshape(ref_diff.shape)
                    post_norm = result.norm()
                    if post_norm > 1e-8:
                        result.mul_(pre_norm / post_norm)
                    del U, S, V, S_gated, gate, mat
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    pass  # SVD failed, skip spectral cleanup

            if selfish_additions is not None:
                result = result + selfish_additions.to(device=result.device, dtype=torch.float32)
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

                is_dare = sparsification == "dare_conflict"
                sparsify_fn = (self._dare_sparsify_conflict if is_dare
                               else self._della_sparsify_conflict)
                for d_f in signed_diffs:
                    kwargs = dict(generator=sparsification_generator)
                    if is_dare:
                        kwargs["dampening"] = dare_dampening
                    trimmed.append(sparsify_fn(d_f, conflict_mask, sparsification_density,
                                               **kwargs))
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
                        trimmed.append(self._dare_sparsify(d_f, sparsification_density, generator=sparsification_generator, dampening=dare_dampening))
                    elif sparsification == "della":
                        trimmed.append(self._della_sparsify(d_f, sparsification_density, generator=sparsification_generator))
                    else:
                        trimmed.append(self._ties_trim(d_f, density))
                    abs_weights.append(abs(w))

            # Enhanced/maximum merge quality pipeline for TIES
            # TALL-masks before orthogonalization (see non-TIES comment above)
            ties_selfish = None
            if merge_quality != "standard" and len(trimmed) >= 2:
                # Re-pair trimmed diffs with abs_weights for enhancement pipeline
                trimmed_pairs = list(zip(trimmed, abs_weights))
                trimmed_pairs, ties_selfish = self._tall_masks(trimmed_pairs)
                first = trimmed_pairs[0][0]
                if first.dim() >= 2:
                    trimmed_pairs = self._do_orthogonalize(trimmed_pairs)
                if merge_quality == "maximum":
                    first = trimmed_pairs[0][0]
                    if first.dim() >= 2 and min(first.shape) >= 2:
                        trimmed_pairs = self._knots_align(
                            trimmed_pairs, compute_device=dev, svd_device=dev)
                trimmed = [d for d, _ in trimmed_pairs]
                abs_weights = [w for _, w in trimmed_pairs]
                del trimmed_pairs

            # Step 2: Elect majority sign
            if merge_quality != "standard":
                majority_sign = self._columnwise_elect_sign(trimmed, majority_sign_method)
            else:
                majority_sign = self._ties_elect_sign(trimmed, majority_sign_method)

            # Step 3: Disjoint merge
            if merge_quality != "standard":
                result = self._columnwise_disjoint_merge(trimmed, abs_weights, majority_sign)
            else:
                result = self._ties_disjoint_merge(trimmed, abs_weights, majority_sign)
            del trimmed, majority_sign
            if ties_selfish is not None:
                result = result.to(dtype=torch.float32) + ties_selfish.to(device=result.device, dtype=torch.float32)
            result = result.to(dtype)
            return result.cpu() if to_cpu else result

        return None

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
            # Standard format: (lora_name, model_strength, clip_strength[, conflict_mode[, key_filter]])
            for entry in lora_stack:
                if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                    logging.warning("[LoRA Optimizer] Skipping malformed tuple entry (expected 3 elements)")
                    continue
                lora_name, model_str, clip_str = entry[0], entry[1], entry[2]
                conflict_mode = entry[3] if len(entry) > 3 else "all"
                key_filter = entry[4] if len(entry) > 4 else "all"

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
                    "key_filter": key_filter,
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
                    "key_filter": item.get("key_filter", "all"),
                })

        else:
            logging.warning("[LoRA Optimizer] Unrecognized stack format")
            return []

        # Always detect architecture (used for preset selection even without key normalization)
        if len(normalized) > 0:
            arch = "unknown"
            for item in normalized:
                detected = self._detect_architecture(item["lora"])
                if detected != "unknown":
                    arch = detected
                    break
            self._detected_arch = arch if arch != "unknown" else None

            # Architecture-aware key normalization (only when enabled)
            if normalize_keys == "enabled":
                if arch != "unknown":
                    logging.info(f"[LoRA Optimizer] Architecture detected: {arch}")
                    logging.info(f"[LoRA Optimizer] Normalizing keys for {len(normalized)} LoRAs...")
                    for item in normalized:
                        item["lora"] = self._normalize_keys(item["lora"], arch)
                else:
                    logging.info("[LoRA Optimizer] Architecture: unknown (no key normalization applied)")
        else:
            self._detected_arch = None

        return normalized

    def _run_pass1_analysis(self, active_loras, all_lora_prefixes, model_keys, clip_keys, model, clip, compute_device, progress_callback=None):
        """
        Run Pass 1 streaming analysis over all LoRA prefixes.

        Args:
            progress_callback: Optional callable invoked after each prefix is analyzed.

        Returns a dict with:
          all_key_targets, per_lora_stats, pair_accum, all_magnitude_samples,
          prefix_count, prefix_stats, skipped_keys, pairs
        """
        use_gpu = compute_device.type != "cpu"

        all_key_targets = {}
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
        pair_accum = {(i, j): [0, 0, 0.0, 0.0, 0.0] for i, j in pairs}
        pair_shared_prefixes = {(i, j): 0 for i, j in pairs}
        all_magnitude_samples = []
        prefix_count = 0
        prefix_stats = {}

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
                pair_shared_prefixes[(i, j)] += 1
            all_magnitude_samples.extend(mag_samples)

            if len(partial_stats) > 0:
                n_contributing = len(partial_stats)
                pf_overlap = sum(ov for ov, conf, dot, na_sq, nb_sq in pair_conflicts.values())
                pf_conflict = sum(conf for ov, conf, dot, na_sq, nb_sq in pair_conflicts.values())
                pf_conflict_ratio = pf_conflict / pf_overlap if pf_overlap > 0 else 0.0
                pf_l2s = [l2 for _, _, l2 in partial_stats if l2 > 0]
                if len(pf_l2s) >= 2:
                    pf_mag_ratio = max(pf_l2s) / min(pf_l2s)
                else:
                    pf_mag_ratio = 1.0
                pf_cos_sims = []
                for (ov, conf, dot, na_sq, nb_sq) in pair_conflicts.values():
                    denom = (na_sq ** 0.5) * (nb_sq ** 0.5)
                    if denom > 0:
                        pf_cos_sims.append(dot / denom)
                avg_cos_sim = sum(pf_cos_sims) / len(pf_cos_sims) if pf_cos_sims else 0.0
                prefix_stats[prefix] = {
                    "n_loras": n_contributing,
                    "conflict_ratio": pf_conflict_ratio,
                    "magnitude_ratio": pf_mag_ratio,
                    "magnitude_samples": list(mag_samples),
                    "avg_cos_sim": avg_cos_sim,
                }

        if use_gpu:
            for lora_prefix in all_lora_prefixes:
                result = self._analyze_prefix(lora_prefix, active_loras,
                                              model_keys, clip_keys, model, clip, compute_device)
                _collect_analysis_result(result)
                if progress_callback:
                    progress_callback()
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
                    if progress_callback:
                        progress_callback()

        return {
            "all_key_targets": all_key_targets,
            "per_lora_stats": per_lora_stats,
            "pair_accum": pair_accum,
            "pair_shared_prefixes": pair_shared_prefixes,
            "all_magnitude_samples": all_magnitude_samples,
            "prefix_count": prefix_count,
            "prefix_stats": prefix_stats,
            "skipped_keys": skipped_keys,
            "pairs": pairs,
        }

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


def _generate_param_grid():
    """Generate all valid merge parameter combinations for AutoTuner sweep."""
    grid = []
    merge_modes = ["weighted_average", "slerp", "consensus", "ties"]
    sparsifications = ["disabled", "dare", "della", "dare_conflict", "della_conflict"]
    densities = [0.5, 0.7, 0.9]
    dampenings = [0.0, 0.3, 0.6]
    qualities = ["standard", "enhanced", "maximum"]
    auto_strengths = ["enabled", "disabled"]

    for spars in sparsifications:
        density_vals = densities if spars != "disabled" else [0.7]
        for density in density_vals:
            damp_vals = dampenings if spars in ("dare", "dare_conflict") else [0.0]
            for dampening in damp_vals:
                for quality in qualities:
                    for auto_str in auto_strengths:
                        # per_prefix: optimizer auto-selects mode per prefix,
                        # merge_mode is irrelevant — emit one entry only
                        grid.append({
                            "merge_mode": "per_prefix_auto",
                            "sparsification": spars,
                            "sparsification_density": density,
                            "dare_dampening": dampening,
                            "merge_quality": quality,
                            "auto_strength": auto_str,
                            "optimization_mode": "per_prefix",
                        })
                        # global: merge_mode matters — emit one per mode
                        for mode in merge_modes:
                            grid.append({
                                "merge_mode": mode,
                                "sparsification": spars,
                                "sparsification_density": density,
                                "dare_dampening": dampening,
                                "merge_quality": quality,
                                "auto_strength": auto_str,
                                "optimization_mode": "global",
                            })
    return grid


def _score_config_heuristic(config, avg_conflict_ratio, avg_cos_sim,
                            magnitude_ratio, prefix_stats, arch_preset=None):
    """
    Score a merge config against analysis metrics (no merge needed).
    Returns float score in [0, 1] where higher = better predicted quality.
    Thresholds come from arch_preset.
    """
    if arch_preset is None:
        arch_preset = _ARCH_PRESETS["sd_unet"]
    mode = config["merge_mode"]
    spars = config["sparsification"]
    density = config["sparsification_density"]
    quality = config["merge_quality"]
    auto_str = config["auto_strength"]
    opt_mode = config["optimization_mode"]

    score = 0.0

    consensus_cos = arch_preset["consensus_cos_sim_min"]
    consensus_conf = arch_preset["consensus_conflict_max"]
    ortho_cos = arch_preset["orthogonal_cos_sim_max"]
    ortho_conf = arch_preset["orthogonal_conflict_max"]
    ties_thresh = arch_preset["ties_conflict_threshold"]
    mag_thresh = arch_preset["magnitude_ratio_total_sign"]
    ideal_density = arch_preset["dare_ideal_density"]

    # --- Mode fit score (0-0.4) ---
    if opt_mode == "per_prefix":
        # Per-prefix auto-selects the best mode per layer.
        # Key advantage: detects orthogonal LoRAs (high conflict + low cos_sim)
        # where TIES would be wrong but per-prefix correctly uses weighted_avg/SLERP.
        if abs(avg_cos_sim) < ortho_cos and avg_conflict_ratio > ties_thresh:
            # Orthogonal LoRAs with high conflict — per-prefix is critical here.
            # Global would pick TIES (wrong), per-prefix detects orthogonality.
            score += 0.40
        elif prefix_stats:
            conflict_ratios = [ps["conflict_ratio"] for ps in prefix_stats.values()
                               if ps.get("n_loras", 0) > 1]
            if conflict_ratios and max(conflict_ratios) - min(conflict_ratios) > 0.15:
                score += 0.35  # high variance = per-prefix shines
            else:
                score += 0.30  # uniform conflict = per-prefix still good
        else:
            score += 0.30
    elif mode == "consensus":
        if avg_cos_sim > consensus_cos and avg_conflict_ratio < consensus_conf:
            score += 0.4
        elif avg_cos_sim > consensus_cos * 0.6 and avg_conflict_ratio < ties_thresh:
            score += 0.25
        else:
            score += 0.05
    elif mode == "slerp":
        if avg_conflict_ratio < ties_thresh * 1.2:
            score += 0.35
        elif avg_conflict_ratio < ortho_conf * 0.83:
            score += 0.20
        else:
            score += 0.10
    elif mode == "weighted_average":
        if abs(avg_cos_sim) < ortho_cos and avg_conflict_ratio < ortho_conf:
            score += 0.30
        elif avg_conflict_ratio < ortho_conf * 0.67:
            score += 0.20
        else:
            score += 0.10
    elif mode == "ties":
        if avg_conflict_ratio > ties_thresh:
            if abs(avg_cos_sim) < ortho_cos:
                # High conflict BUT orthogonal LoRAs — conflict is noise, not real.
                # TIES would aggressively prune, losing important information.
                score += 0.15
            else:
                score += 0.35
        elif avg_conflict_ratio > ties_thresh * 0.6:
            score += 0.20
        else:
            score += 0.10

    # --- Effective conflict (used by sparsification and quality scoring) ---
    # When LoRAs are orthogonal, high conflict ratio is base-rate noise, not real
    # conflict. Heavy processing fights noise and can amplify artifacts.
    is_orthogonal = abs(avg_cos_sim) < ortho_cos
    effective_conflict = avg_conflict_ratio * min(abs(avg_cos_sim) / ortho_cos, 1.0) if is_orthogonal else avg_conflict_ratio

    # --- Sparsification fit (0-0.15) ---
    if spars != "disabled":
        conflict_benefit = min(effective_conflict / 0.5, 1.0) * 0.10
        score += conflict_benefit
        density_penalty = abs(density - ideal_density) * 0.05
        score += 0.05 - density_penalty
        # Conflict-aware variants (dare_conflict, della_conflict) benefit from
        # high conflict variance across prefixes — they adapt masking per-prefix.
        if "_conflict" in spars and prefix_stats:
            conflict_ratios = [ps["conflict_ratio"] for ps in prefix_stats.values()
                               if ps.get("n_loras", 0) > 1]
            if conflict_ratios and len(conflict_ratios) > 1:
                variance = max(conflict_ratios) - min(conflict_ratios)
                score += variance * 0.05  # up to ~0.05 bonus for high variance
        # DELLA uses magnitude-aware masking — benefits when LoRAs have uneven norms
        if spars in ("della", "della_conflict"):
            if magnitude_ratio > mag_thresh:
                score += 0.03
    else:
        if effective_conflict < consensus_conf:
            score += 0.10

    # --- Quality fit (0-0.15) ---
    # Enhanced/maximum resolve conflicts via orthogonalization, column-wise
    # resolution, TALL masks, and SVD alignment.  When LoRAs are orthogonal
    # there are no real conflicts — the extra processing treats noise as
    # signal and can smooth away valid independent contributions.
    if quality == "maximum":
        if is_orthogonal:
            score += min(effective_conflict / 0.3, 1.0) * 0.10
        else:
            score += 0.05 + min(effective_conflict / 0.3, 1.0) * 0.10
    elif quality == "enhanced":
        if is_orthogonal:
            score += min(effective_conflict / 0.3, 1.0) * 0.10
        else:
            score += 0.08 + min(effective_conflict / 0.3, 1.0) * 0.07
    else:
        if effective_conflict < 0.10:
            score += 0.10
        else:
            score += 0.05

    # --- Auto-strength fit (0-0.15) ---
    if auto_str == "enabled":
        if magnitude_ratio > mag_thresh:
            score += 0.15
        elif magnitude_ratio > mag_thresh * 0.75:
            score += 0.10
        else:
            score += 0.05
    else:
        if magnitude_ratio < mag_thresh * 0.75:
            score += 0.10
        else:
            score += 0.03

    # --- Optimization mode fit (0-0.15) ---
    # Per-prefix benefit already scored in mode fit section above.
    if opt_mode == "per_prefix":
        score += 0.10
    else:
        score += 0.07

    return score


def _score_merge_result(model_patches, clip_patches, compute_svd=True, score_device=None):
    """
    Score an actual merge result by measuring output quality metrics.
    Returns dict with individual metrics and composite score in [0, 1].

    When compute_svd=False, skips the expensive SVD-based effective rank
    computation and scores using norm consistency + sparsity only.
    When score_device is set (e.g. "cuda"), tensors are moved there for
    faster norm/SVD/sparsity computation.
    """
    norms = []
    effective_ranks = []
    sparsities = []

    all_patches = list(model_patches.values()) + list(clip_patches.values())
    total = len(all_patches)
    device_label = f", device={score_device}" if score_device else ""
    logging.info(f"[LoRA AutoTuner]   Scoring merge quality ({total} patches"
                 f"{', +SVD' if compute_svd else ''}{device_label})")
    log_interval = max(1, total // 4)  # Log at ~25%, 50%, 75%, 100%
    for patch_idx, patch in enumerate(all_patches):
        if (patch_idx + 1) % log_interval == 0 or patch_idx + 1 == total:
            logging.info(f"[LoRA AutoTuner]     Scored {patch_idx + 1}/{total} patches")
        if patch is None:
            continue
        if isinstance(patch, tuple) and len(patch) >= 2:
            tensor = patch[1][0] if isinstance(patch[1], tuple) else patch[1]
        elif isinstance(patch, LoKrAdapter):
            # LoKr norm: ||kron(A,B)||_F = ||A||_F * ||B||_F — avoids materializing full diff
            v = patch.weights
            w1, w2, alpha = v[0], v[1], v[2]
            w1_a, w1_b, w2_a, w2_b, t2 = v[3], v[4], v[5], v[6], v[7]
            dim = None
            if w1 is None:
                dim = w1_b.shape[0]
                w1 = torch.mm(w1_a.float(), w1_b.float())
            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    w2 = torch.mm(w2_a.float(), w2_b.float())
                else:
                    w2 = torch.einsum("i j k l, j r, i p -> p r k l", t2.float(), w2_b.float(), w2_a.float())
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            scale = alpha / dim if (alpha is not None and dim is not None) else 1.0
            fro_norm = (w1.float().norm().item() * w2.float().norm().item() * abs(scale))
            norms.append(fro_norm)
            del w1, w2
            continue
        elif isinstance(patch, LoHaAdapter):
            # LoHa: must expand to diff — Hadamard product has no cheap norm identity
            diff = _LoRAMergeBase._expand_patch_to_diff(patch)
            if score_device is not None:
                diff = diff.to(score_device)
            norms.append(diff.norm().item())
            max_val = diff.abs().max().item()
            threshold = max_val * 0.01
            if threshold > 0:
                sparsity = (diff.abs() < threshold).float().mean().item()
                sparsities.append(sparsity)
            del diff
            continue
        elif isinstance(patch, LoRAAdapter):
            # Low-rank patch: compute norm from factors without materializing full diff
            mat_up, mat_down, alpha, mid, _, _ = patch.weights
            if mat_up is not None and mat_down is not None:
                rank = mat_down.shape[0] if mat_down.dim() >= 1 else 1
                scale = alpha / rank if rank > 0 else 1.0
                up_flat = mat_up.flatten(start_dim=1).float()
                down_flat = mat_down.flatten(start_dim=1).float()
                if score_device is not None:
                    up_flat = up_flat.to(score_device)
                    down_flat = down_flat.to(score_device)
                # ||AB||_F^2 = tr(A^T A B B^T) — avoids materializing full diff
                gram_up = torch.mm(up_flat.T, up_flat)
                gram_down = torch.mm(down_flat, down_flat.T)
                fro_norm = (torch.trace(gram_up @ gram_down).clamp(min=0) ** 0.5 * abs(scale)).item()
                norms.append(fro_norm)
                # Estimate element-wise sparsity by sampling columns of the product
                n_cols = down_flat.shape[1]
                sample_k = min(64, n_cols)
                col_idx = torch.randperm(n_cols, device=down_flat.device)[:sample_k]
                sampled = torch.mm(up_flat, down_flat[:, col_idx]) * scale
                max_val = sampled.abs().max().item()
                threshold = max_val * 0.01
                if threshold > 0:
                    sparsity = (sampled.abs() < threshold).float().mean().item()
                    sparsities.append(sparsity)
                del sampled
            continue
        else:
            continue

        if tensor is None:
            continue

        t = tensor.float()
        if score_device is not None:
            t = t.to(score_device)

        # Frobenius norm
        norms.append(t.norm().item())

        # Effective rank via spectral analysis (optional, expensive)
        if compute_svd and t.dim() == 2 and min(t.shape) > 1:
            try:
                s = torch.linalg.svdvals(t)[:min(min(t.shape), 64)]
                s_norm = s / (s.sum() + 1e-10)
                entropy = -(s_norm * (s_norm + 1e-10).log()).sum().item()
                eff_rank = min(math.exp(entropy), min(t.shape))
                effective_ranks.append(eff_rank)
            except Exception:
                pass

        # Sparsity
        threshold = t.abs().max().item() * 0.01
        if threshold > 0:
            sparsity = (t.abs() < threshold).float().mean().item()
            sparsities.append(sparsity)

    metrics = {}

    if norms:
        metrics["norm_mean"] = sum(norms) / len(norms)
        metrics["norm_std"] = (sum((n - metrics["norm_mean"])**2 for n in norms)
                               / len(norms)) ** 0.5
        metrics["norm_cv"] = metrics["norm_std"] / (metrics["norm_mean"] + 1e-10)
    else:
        metrics["norm_mean"] = 0.0
        metrics["norm_cv"] = 1.0

    if effective_ranks:
        metrics["effective_rank_mean"] = sum(effective_ranks) / len(effective_ranks)
    else:
        metrics["effective_rank_mean"] = 0.0

    if sparsities:
        avg_sparsity = sum(sparsities) / len(sparsities)
        metrics["sparsity_mean"] = avg_sparsity
        metrics["sparsity_fit"] = max(0.0, 1.0 - abs(avg_sparsity - 0.4) * 2.0)
    else:
        metrics["sparsity_mean"] = 0.0
        metrics["sparsity_fit"] = 0.5

    # Composite score — rebalance weights when SVD is skipped
    score = 0.0
    if effective_ranks:
        rank_score = min(metrics["effective_rank_mean"] / 40.0, 1.0)
        score += rank_score * 0.4
        cv_score = max(0.0, 1.0 - metrics["norm_cv"])
        score += cv_score * 0.3
        score += metrics["sparsity_fit"] * 0.3
    else:
        # No SVD data: norm consistency 50%, sparsity 50%
        cv_score = max(0.0, 1.0 - metrics["norm_cv"])
        score += cv_score * 0.5
        score += metrics["sparsity_fit"] * 0.5

    metrics["composite_score"] = score
    return metrics


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
        super().__init__()
        self._merge_cache = {}  # single-entry: {cache_key: (model_patches, clip_patches, report, clip_strength_out, lora_data)}
        self._detected_arch = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Your base model (e.g. SDXL, Flux). The merged LoRAs will be applied to it."}),
                "lora_stack": ("LORA_STACK", {"tooltip": "Connect a LoRA Stack node here. This is the list of LoRAs you want to merge together."}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 10.0, "step": 0.05,
                                              "tooltip": "Master volume for the merged result. 1.0 = full effect, 0.5 = half, 0 = disabled. Set to -1 for auto: uses the suggested max strength (compensates for energy lost during merge)."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "The text encoder. Connect this so LoRAs can also affect how your prompts are understood. Leave empty for video models that don't use CLIP."}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                                                       "tooltip": "How strongly LoRAs affect text understanding, relative to output_strength. At 1.0, CLIP uses the same strength as the model. Lower values reduce LoRA influence on prompt interpretation while keeping the visual effect."}),
                "auto_strength": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": "Automatically turns down individual LoRA strengths when combining many LoRAs to avoid oversaturated or distorted results. Useful when stacking 3+ LoRAs."
                }),
                "auto_strength_floor": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum scale factor for auto-strength when LoRAs are orthogonal (independent). "
                               "-1 = auto (1.0 for video models like Wan/LTX to preserve motion, 0.85 for image models). "
                               "Set manually to override — e.g. 0.9 for less reduction, 1.0 to disable reduction for orthogonal LoRAs entirely."
                }),
                "free_vram_between_passes": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Frees GPU memory between processing steps. Enable if you're running out of VRAM. Barely affects speed."
                }),
                "vram_budget": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of free VRAM to use for storing merged patches. 0 = all CPU (default), 1.0 = use all free VRAM. Reduces RAM usage on GPU systems."
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
                    "default": "enabled",
                    "tooltip": "Remaps LoRA keys to a canonical format so LoRAs from different training tools (Kohya, AI-Toolkit, PEFT, Musubi Tuner) can be merged correctly. Also splits fused QKV into separate Q/K/V for per-component conflict analysis. Recommended to keep enabled — disable only if it causes issues with unusual LoRA formats."
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
                "dare_dampening": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "DAREx dampening: reduces the aggressiveness of DARE's rescaling factor. "
                               "At 0.0 (default): standard DARE rescaling (1/density). "
                               "At higher values: dampened rescaling that reduces noise amplification at low density values. "
                               "Only affects DARE/DARE-conflict modes. Based on DAREx (ICLR 2025)."
                }),
                "merge_quality": (["standard", "enhanced", "maximum"], {
                    "default": "standard",
                    "tooltip": "Merge quality level. standard: element-wise conflict resolution. "
                               "enhanced: direction orthogonalization + column-wise conflict "
                               "resolution + selfish weight protection (minimal extra compute, no extra VRAM). "
                               "maximum: adds SVD alignment before merge (best quality, "
                               "uses more VRAM for SVD decomposition)."
                }),
                "behavior_profile": (["v1.2", "no_slerp", "classic"], {
                    "default": "v1.2",
                    "tooltip": "Controls auto-selection logic. "
                               "'v1.2': full v1.2 behavior (consensus, orthogonal detection, SLERP upgrade). "
                               "'no_slerp': v1.2 detection without SLERP upgrade (weighted_average stays as-is). "
                               "'classic': pre-1.2 behavior (TIES vs weighted_average only, no SLERP)."
                }),
                "architecture_preset": (["auto", "sd_unet", "dit", "llm"], {
                    "default": "auto",
                    "tooltip": "Architecture-aware threshold tuning. 'auto' detects from LoRA keys. "
                               "'sd_unet': SD/SDXL UNet defaults. 'dit': DiT models (Flux, WAN, Z-Image, LTX, HunyuanVideo) "
                               "with higher density floors and wider strength range. 'llm': LLM-based models (Qwen, LLaMA)."
                }),
                "merge_strategy_override": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Connect the merge_strategy output from a LoRA Conflict Editor to override the optimizer's auto-detected strategy."
                }),
                "tuner_data": ("TUNER_DATA", {
                    "tooltip": "Connect from the LoRA AutoTuner's tuner_data output. Used when settings_source is 'from_autotuner'."
                }),
                "settings_source": (["manual", "from_autotuner"], {
                    "default": "manual",
                    "tooltip": "manual: use widget settings. from_autotuner: use the AutoTuner's best config (connect tuner_data). Switch to manual to tweak from the AutoTuner's starting point."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "TUNER_DATA", "LORA_DATA")
    RETURN_NAMES = ("model", "clip", "analysis_report", "tuner_data", "lora_data")
    FUNCTION = "execute_node"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Auto-analyzes LoRA stack and selects optimal merge strategy per weight group. Outputs merged model + analysis report. Best for style/character LoRAs — apply edit, distillation (LCM/Turbo/Hyper), or DPO LoRAs via a standard Load LoRA node instead."

    @staticmethod
    def _compute_cache_key(lora_stack, output_strength, clip_strength_multiplier, auto_strength, optimization_mode="per_prefix", compress_patches="non_ties", svd_device="gpu", normalize_keys="disabled", sparsification="disabled", sparsification_density=0.7, dare_dampening=0.0, merge_strategy_override="", merge_quality="standard", behavior_profile="v1.2", architecture_preset="auto", auto_strength_floor=-1.0):
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
                    kf = entry[4] if len(entry) > 4 else "all"
                    cs = float(entry[2]) if entry[2] is not None else -1.0
                    entries.append((str(entry[0]), float(entry[1]), cs, cm, kf))
            elif isinstance(first, dict):
                for item in lora_stack:
                    cm = item.get("conflict_mode", "all")
                    kf = item.get("key_filter", "all")
                    entries.append((str(item.get("name", "")), float(item.get("strength", 0)), cm, kf))
            entries.sort()
            h.update(json.dumps(entries).encode())
        h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}|om={optimization_mode}|cp={compress_patches}|sd={svd_device}|nk={normalize_keys}|sp={sparsification}|spd={sparsification_density}|dd={dare_dampening}|mso={merge_strategy_override}|mq={merge_quality}|bp={behavior_profile}|ap={architecture_preset}|asf={auto_strength_floor}".encode())
        return h.hexdigest()[:16]

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, clip=None,
                   clip_strength_multiplier=1.0, auto_strength="disabled",
                   auto_strength_floor=-1.0,
                   free_vram_between_passes="disabled", vram_budget=0.0,
                   optimization_mode="per_prefix",
                   cache_patches="enabled", compress_patches="non_ties",
                   svd_device="gpu", normalize_keys="disabled",
                   sparsification="disabled", sparsification_density=0.7,
                   dare_dampening=0.0,
                   merge_strategy_override="", merge_quality="standard",
                   behavior_profile="v1.2", architecture_preset="auto",
                   tuner_data=None, settings_source="manual"):
        base_key = cls._compute_cache_key(lora_stack, output_strength,
                                          clip_strength_multiplier, auto_strength,
                                          optimization_mode, compress_patches,
                                          svd_device, normalize_keys,
                                          sparsification, sparsification_density,
                                          dare_dampening,
                                          merge_strategy_override, merge_quality,
                                          behavior_profile, architecture_preset,
                                          auto_strength_floor)
        cache_key = f"{base_key}|mid={id(model)}|ss={settings_source}"
        if settings_source == "from_autotuner" and tuner_data is not None:
            return f"{cache_key}|at={id(tuner_data)}"
        return cache_key

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

    def _estimate_density(self, all_key_diffs, arch_preset=None):
        """
        Estimate TIES density parameter from magnitude distribution.
        Uses fraction of values above noise floor of the max magnitude as a
        sparsity proxy. Thresholds come from arch_preset.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
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

        noise_floor = max_val * arch_preset["density_noise_floor_ratio"]
        above_noise = (all_samples > noise_floor).float().mean().item()

        return max(arch_preset["density_clamp_min"], min(arch_preset["density_clamp_max"], above_noise))

    def _estimate_density_from_samples(self, magnitude_samples, arch_preset=None):
        """
        Estimate TIES density from pre-sampled magnitude tensors.
        Takes a list of 1D CPU float tensors (from _analyze_prefix).
        Thresholds come from arch_preset.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
        if len(magnitude_samples) == 0:
            return 0.5

        all_samples = torch.cat(magnitude_samples)
        if all_samples.numel() == 0:
            return 0.5

        max_val = all_samples.max().item()
        if max_val <= 0:
            return 0.5

        noise_floor = max_val * arch_preset["density_noise_floor_ratio"]
        above_noise = (all_samples > noise_floor).float().mean().item()

        return max(arch_preset["density_clamp_min"], min(arch_preset["density_clamp_max"], above_noise))

    def _compute_auto_strengths(self, active_loras, lora_stats, pairwise_similarities=None, arch_preset=None, detected_arch=None, auto_strength_floor=-1.0, is_full_rank=False):
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
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
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

        # Scale factor — capped at 1.0 to prevent amplification.
        # For opposing LoRAs (negative cos_sim), energy_sq shrinks and scale
        # would exceed 1.0 (e.g. 1.58x at cos_sim=-0.8), boosting both LoRAs
        # instead of letting them naturally attenuate each other.
        if current_energy > 0:
            scale = min(reference_energy / current_energy, 1.0)
        else:
            scale = 1.0

        # Architecture-aware orthogonal floor: when LoRAs are orthogonal
        # (independent), video models should preserve full energy because
        # orthogonal video LoRAs teach independent motions.  Image models
        # get a modest floor to prevent over-reduction while still dampening.
        floor_applied = False
        if pairwise_similarities:
            avg_cos = sum(pairwise_similarities.values()) / len(pairwise_similarities)
            alignment_thresh = arch_preset["alignment_threshold"]
            if abs(avg_cos) <= alignment_thresh:
                if auto_strength_floor >= 0:
                    floor = auto_strength_floor
                elif is_full_rank:
                    # Full-rank patches already represent the complete weight delta;
                    # scaling them down loses information.
                    floor = arch_preset.get("full_rank", {}).get("auto_strength_floor", 1.0)
                else:
                    floor = _VIDEO_ARCH_ORTHOGONAL_FLOOR.get(
                        detected_arch,
                        arch_preset.get("auto_strength_orthogonal_floor", 0.85))
                if scale < floor:
                    scale = floor
                    floor_applied = True

        # Apply scale to all strengths
        new_strengths = []
        for i in range(n):
            if effective[i] > 0:
                new_strengths.append(original_strengths[i] * scale)
            else:
                new_strengths.append(original_strengths[i])

        # Build reasoning
        reasoning.append(f"Scale factor: {scale:.4f}")
        if floor_applied:
            arch_label = detected_arch or "unknown"
            if is_full_rank:
                reasoning.append(f"  (full-rank orthogonal floor {floor:.2f} applied — preserving complete weight deltas)")
            else:
                reasoning.append(f"  (orthogonal floor {floor:.2f} applied for {arch_label} — preserving independent contributions)")
        if pairwise_similarities:
            avg_cos = sum(pairwise_similarities.values()) / len(pairwise_similarities)
            orthogonal_energy = math.sqrt(orthogonal_energy_sq)
            alignment_thresh = arch_preset["alignment_threshold"]
            if avg_cos > alignment_thresh:
                alignment_desc = "mostly aligned (reinforcing)"
            elif avg_cos < -alignment_thresh:
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

    def _auto_select_params(self, avg_conflict_ratio, magnitude_ratio, all_key_diffs=None, magnitude_samples=None, avg_cos_sim=0.0, behavior_profile="v1.2", arch_preset=None, precomputed_density=None):
        """
        Decision logic for auto-selecting merge parameters.
        Returns (mode, density, sign_method, reasoning_lines).

        Density can be estimated from either all_key_diffs (legacy bulk path)
        or magnitude_samples (streaming path). Thresholds come from arch_preset.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
        reasoning = []

        # High similarity + low conflict → consensus mode (Fisher-proxy + magnitude calibration)
        if (behavior_profile == "v1.2"
                and avg_cos_sim > arch_preset["consensus_cos_sim_min"]
                and avg_conflict_ratio < arch_preset["consensus_conflict_max"]):
            mode = "consensus"
            reasoning.append(f"Cosine similarity {avg_cos_sim:.2f} > {arch_preset['consensus_cos_sim_min']} "
                             f"and conflict {avg_conflict_ratio:.1%} < {arch_preset['consensus_conflict_max']:.0%} -> consensus mode")
            reasoning.append("  Fisher-proxy importance weighting + magnitude calibration + spectral cleanup")
            density = 0.5  # unused
            sign_method = "frequency"  # unused
            return (mode, density, sign_method, reasoning)

        # Near-orthogonal LoRAs: ~50% sign conflict is the base rate for
        # independent vectors, not actual semantic conflict. TIES trimming
        # destroys both signals. Use weighted_average as the global mode,
        # upgraded to SLERP per-prefix to preserve magnitude (important for
        # video LoRAs where motion energy matters).
        if (behavior_profile in ("v1.2", "no_slerp")
                and abs(avg_cos_sim) < arch_preset["orthogonal_cos_sim_max"]
                and avg_conflict_ratio < arch_preset["orthogonal_conflict_max"]):
            mode = "weighted_average"
            reasoning.append(f"Cosine similarity {avg_cos_sim:.2f} near zero (orthogonal LoRAs) — "
                             f"sign conflict {avg_conflict_ratio:.1%} is base-rate noise, not real conflict")
            if behavior_profile == "v1.2":
                reasoning.append("  Using weighted_average (upgraded to SLERP per-prefix to preserve magnitude)")
            else:
                reasoning.append("  Using weighted_average to preserve both signals (SLERP upgrade disabled by profile)")
            density = 0.5
            sign_method = "frequency"
            return (mode, density, sign_method, reasoning)

        # Select mode based on sign conflict
        if avg_conflict_ratio > arch_preset["ties_conflict_threshold"]:
            mode = "ties"
            reasoning.append(f"Sign conflict ratio {avg_conflict_ratio:.1%} > {arch_preset['ties_conflict_threshold']:.0%} threshold -> TIES mode selected")
            reasoning.append("  TIES resolves sign conflicts via trim + elect sign + disjoint merge")
        else:
            mode = "weighted_average"
            reasoning.append(f"Sign conflict ratio {avg_conflict_ratio:.1%} <= {arch_preset['ties_conflict_threshold']:.0%} threshold -> weighted_average mode selected")
            reasoning.append("  Low conflict means LoRAs are mostly compatible, simple averaging works well")

        # Auto-density (TIES only)
        if mode == "ties":
            if magnitude_samples is not None:
                density = self._estimate_density_from_samples(magnitude_samples, arch_preset=arch_preset)
            elif all_key_diffs is not None:
                density = self._estimate_density(all_key_diffs, arch_preset=arch_preset)
            elif precomputed_density is not None:
                density = precomputed_density
            else:
                density = 0.5
            reasoning.append(f"Auto-density estimated at {density:.2f} from magnitude distribution")
        else:
            density = 0.5  # unused but set for completeness

        # Sign method (only relevant for TIES mode)
        if mode == "ties":
            if magnitude_ratio > arch_preset["magnitude_ratio_total_sign"]:
                sign_method = "total"
                reasoning.append(f"Magnitude ratio {magnitude_ratio:.2f}x > {arch_preset['magnitude_ratio_total_sign']:.0f}x -> 'total' sign method (magnitude-weighted voting)")
                reasoning.append("  Stronger LoRA gets more influence in sign election")
            else:
                sign_method = "frequency"
                reasoning.append(f"Magnitude ratio {magnitude_ratio:.2f}x <= {arch_preset['magnitude_ratio_total_sign']:.0f}x -> 'frequency' sign method (equal voting)")
                reasoning.append("  Similar-strength LoRAs get equal votes")
        else:
            sign_method = "frequency"  # unused, default for completeness

        return (mode, density, sign_method, reasoning)

    def _build_report(self, lora_stats, pairwise_conflicts, collection_stats,
                      mode, density, sign_method, reasoning, merge_summary,
                      auto_strength_info=None, strategy_counts=None, optimization_mode="global",
                      prefix_decisions=None, detected_arch=None, normalize_keys="disabled",
                      sparsification="disabled", sparsification_density=0.7,
                      dare_dampening=0.0,
                      merge_quality="standard",
                      compatibility_warnings=None,
                      behavior_profile="v1.2",
                      architecture_preset=None,
                      is_full_rank=False):
        """Format analysis as a multi-line report string."""
        lines = []
        lines.append("=" * 50)
        lines.append("LORA OPTIMIZER - ANALYSIS REPORT")
        lines.append("=" * 50)

        # Architecture preset info
        if architecture_preset and architecture_preset in _ARCH_PRESETS:
            preset_info = _ARCH_PRESETS[architecture_preset]
            lines.append(f"Architecture preset: {architecture_preset} ({preset_info['display_name']})")
            if detected_arch:
                arch_names = {
                    'zimage': 'Z-Image Turbo (Lumina2)',
                    'flux': 'FLUX',
                    'wan': 'Wan 2.1/2.2',
                    'acestep': 'ACE-Step',
                    'sdxl': 'SDXL',
                    'ltx': 'LTX Video',
                    'qwen_image': 'Qwen-Image',
                }
                lines.append(f"Detected architecture: {arch_names.get(detected_arch, detected_arch)}")
            if normalize_keys == "enabled":
                lines.append(f"Key normalization: enabled")
            lines.append("")
        elif normalize_keys == "enabled" and detected_arch:
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

        if is_full_rank:
            global_avg_rank = (sum(s["avg_rank"] for s in lora_stats) / len(lora_stats)) if lora_stats else 0
            lines.append("Full-rank LoRAs detected (avg rank {:.0f})".format(global_avg_rank))
            lines.append("  Using full-rank merge path: SLERP upgrade disabled,")
            lines.append("  orthogonal patches use weighted_sum, auto-strength floor raised")
            lines.append("")

        if compatibility_warnings:
            lines.append("")
            lines.append("!!! COMPATIBILITY WARNING !!!")
            for warn in compatibility_warnings:
                lines.append(f"  {warn['name_i']} vs {warn['name_j']}: cosine similarity = {warn['cosine_sim']:.3f}")
                lines.append(f"    These LoRAs work against each other and may cancel out.")
                lines.append(f"    Consider removing one or using conflict_mode='high_conflict'")
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
            if stat.get("key_filter", "all") != "all":
                lines.append(f"    Key filter: {stat['key_filter']}")

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
            if dare_dampening > 0 and sparsification in ("dare", "dare_conflict"):
                q = sparsification_density + dare_dampening * (1.0 - sparsification_density)
                lines.append(f"  DAREx dampening: {dare_dampening:.2f} (rescale factor: 1/{q:.2f} = {1.0/q:.2f}x vs standard 1/{sparsification_density:.2f} = {1.0/sparsification_density:.2f}x)")
            if optimization_mode == "per_prefix":
                lines.append(f"  For TIES prefixes: replaces trim step; others: preprocessing")
            elif optimization_mode == "weighted_sum_only":
                lines.append(f"  Applied as preprocessing before weighted_sum")
            elif mode == "ties":
                lines.append(f"  Note: {display_name} replaces TIES trim step")
            else:
                lines.append(f"  Applied as preprocessing before {mode}")

        if merge_quality != "standard":
            quality_desc = {
                "enhanced": "Enhanced (DO-orthogonalize + column-wise + TALL-masks)",
                "maximum": "Maximum (DO-orthogonalize + KnOTS SVD alignment + column-wise + TALL-masks)",
            }
            lines.append(f"  Merge quality: {quality_desc.get(merge_quality, merge_quality)}")

        if behavior_profile != "v1.2":
            profile_desc = {
                "no_slerp": "no_slerp (v1.2 detection, no SLERP upgrade)",
                "classic": "classic (pre-1.2 behavior, TIES vs weighted_average only)",
            }
            lines.append(f"  Behavior profile: {profile_desc.get(behavior_profile, behavior_profile)}")

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
                    lines.append(f"  slerp (low conflict):            {n:>4} prefixes ({n/total_pf:.0%})")
                if strategy_counts.get("weighted_average", 0) > 0:
                    n = strategy_counts["weighted_average"]
                    lines.append(f"  weighted_average (orthogonal):    {n:>4} prefixes ({n/total_pf:.0%})")
                if strategy_counts.get("consensus", 0) > 0:
                    n = strategy_counts["consensus"]
                    lines.append(f"  consensus (high similarity):     {n:>4} prefixes ({n/total_pf:.0%})")
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
            mode_priority = {"ties": 4, "consensus": 3, "slerp": 2, "weighted_average": 1, "weighted_sum": 0}
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
            symbols = {"weighted_sum": "====", "slerp": "~~~~", "weighted_average": "----", "ties": "####", "consensus": "++++"}
            labels = {"weighted_sum": "sum", "slerp": "slrp", "weighted_average": "avg", "ties": "TIES", "consensus": "cons"}
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
                    for m in ("weighted_sum", "weighted_average", "slerp", "consensus", "ties"):
                        if mode_counts.get(m, 0) > 0:
                            parts.append(f"{mode_counts[m]} {labels.get(m, m)}")
                    detail = f"{avg_conflict:.0%} conflict ({', '.join(parts)})"
                count_str = f"({n_prefixes}x)" if n_prefixes > 1 else ""
                lines.append(f"  {block_name:<{max_name}}  {sym}  {lbl:<5} {detail} {count_str}")
            lines.append(f"  Legend: ==== sum  ~~~~ slerp  ---- avg  ++++ cons  #### TIES")

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
        os_val = merge_summary['output_strength']
        auto_os = merge_summary.get('auto_output_strength', False)
        if auto_os:
            lines.append(f"  Output strength: {os_val:.2f} (auto — suggested max)")
        else:
            lines.append(f"  Output strength: {os_val}")
        lines.append(f"  CLIP strength: {merge_summary['clip_strength']}")
        if not auto_os and merge_summary.get('suggested_max_strength') is not None:
            sms = merge_summary['suggested_max_strength']
            lines.append(f"  Suggested max output_strength: {sms:.2f}")
            if sms >= 3.0:
                lines.append(f"    (capped at 3.0 — actual headroom may be higher)")
            elif sms == 1.0:
                lines.append(f"    (energy preserved — no compensation needed)")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)

    def execute_node(self, model, lora_stack, output_strength, clip=None,
                     clip_strength_multiplier=1.0, auto_strength="enabled",
                     auto_strength_floor=-1.0,
                     free_vram_between_passes="disabled", vram_budget=0.0,
                     optimization_mode="per_prefix", cache_patches="enabled",
                     compress_patches="non_ties", svd_device="gpu",
                     normalize_keys="enabled", sparsification="disabled",
                     sparsification_density=0.7, dare_dampening=0.0,
                     merge_strategy_override="", merge_quality="standard",
                     behavior_profile="v1.2", architecture_preset="auto",
                     tuner_data=None, settings_source="manual"):
        """
        ComfyUI entry point. Routes between passthrough and merge modes:
        - from_autotuner: passthrough (model already merged by AutoTuner upstream),
          sync widgets from tuner_data
        - manual: merge using widget settings (AutoTuner upstream in tuning_only
          mode passes the base model through)
        """
        # --- from_autotuner: passthrough mode ---
        if settings_source == "from_autotuner":
            if tuner_data is None or "top_n" not in tuner_data or len(tuner_data["top_n"]) == 0:
                logging.warning("[AutoTuner Bridge] No valid tuner_data — falling back to manual merge")
            else:
                config = tuner_data["top_n"][0]["config"]
                entry = tuner_data["top_n"][0]
                score = entry.get("score_measured", entry.get("score_heuristic", 0))
                mode_display = ("per-prefix (auto)" if config["optimization_mode"] == "per_prefix"
                                else config.get("merge_mode", "unknown"))

                report_lines = [
                    f"[AutoTuner Bridge] Passthrough — model merged by AutoTuner (rank #1, score: {score:.3f}):",
                    f"  optimization_mode: {config['optimization_mode']}",
                    f"  merge mode: {mode_display}",
                    f"  merge_quality: {config['merge_quality']}",
                ]
                if config["sparsification"] != "disabled":
                    report_lines.append(
                        f"  sparsification: {config['sparsification']} "
                        f"(density: {config['sparsification_density']}, "
                        f"dampening: {config['dare_dampening']})"
                    )
                report_lines.append(f"  auto_strength: {config['auto_strength']}")
                report_lines.append("")
                report_lines.append("Switch settings_source to 'manual' to tweak from these settings.")
                report = "\n".join(report_lines)

                logging.info(f"[AutoTuner Bridge] Passthrough mode — model already merged by AutoTuner")

                # Return model as-is + UI data for JS widget sync
                return {"result": (model, clip, report, tuner_data, None),
                        "ui": {"applied_settings": [json.dumps(config)]}}

        # --- manual: active merge mode ---
        result = self.optimize_merge(
            model, lora_stack, output_strength,
            clip=clip,
            clip_strength_multiplier=clip_strength_multiplier,
            auto_strength=auto_strength,
            auto_strength_floor=auto_strength_floor,
            free_vram_between_passes=free_vram_between_passes,
            vram_budget=vram_budget,
            optimization_mode=optimization_mode,
            cache_patches=cache_patches,
            compress_patches=compress_patches,
            svd_device=svd_device,
            normalize_keys=normalize_keys,
            sparsification=sparsification,
            sparsification_density=sparsification_density,
            dare_dampening=dare_dampening,
            merge_strategy_override=merge_strategy_override,
            merge_quality=merge_quality,
            behavior_profile=behavior_profile,
            architecture_preset=architecture_preset,
        )

        return result

    def optimize_merge(self, model, lora_stack, output_strength, clip=None, clip_strength_multiplier=1.0, auto_strength="disabled", auto_strength_floor=-1.0, free_vram_between_passes="disabled", vram_budget=0.0, optimization_mode="per_prefix", cache_patches="enabled", compress_patches="non_ties", svd_device="gpu", normalize_keys="disabled", sparsification="disabled", sparsification_density=0.7, dare_dampening=0.0, merge_strategy_override="", merge_quality="standard", behavior_profile="v1.2", architecture_preset="auto", _analysis_cache=None, _diff_cache=None, _skip_report=False):
        """
        Main entry point. Two-pass streaming architecture:
        Pass 1: Compute diffs per-prefix, sample conflicts + magnitudes, discard diffs
        Decision: Finalize stats, auto-select params from lightweight accumulators
        Pass 2: Recompute diffs per-prefix, merge immediately, discard
        Peak memory: ~260MB (one prefix's diffs at a time) vs ~50GB (all diffs).
        """
        # Normalize stack format (standard tuples or LoRAStack dicts)
        if not lora_stack or len(lora_stack) == 0:
            return (model, clip, "No LoRAs in stack.", None, None)

        normalized_stack = self._normalize_stack(lora_stack, normalize_keys=normalize_keys)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]

        if len(active_loras) == 0:
            return (model, clip, "No LoRAs in stack (all zero strength or malformed).", None, None)

        # Resolve architecture preset from override or auto-detection
        preset_key, arch_preset = _resolve_arch_preset(
            architecture_preset, getattr(self, '_detected_arch', None) or 'unknown')
        logging.info(f"[LoRA Optimizer] Architecture preset: {preset_key} ({arch_preset['display_name']})")

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
                                            dare_dampening,
                                            merge_strategy_override, merge_quality,
                                            behavior_profile, architecture_preset,
                                            auto_strength_floor)
        # Include model identity so switching base models invalidates the cache
        cache_key = f"{cache_key}|mid={id(model)}"
        if cache_patches == "enabled" and cache_key in self._merge_cache:
            model_patches, clip_patches, report, clip_strength_out, lora_data = self._merge_cache[cache_key]
            # Resolve auto output_strength from cached suggested_max_strength
            cached_os = output_strength
            if cached_os < 0 and lora_data and lora_data.get("suggested_max_strength") is not None:
                cached_os = lora_data["suggested_max_strength"]
                clip_strength_out = cached_os * clip_strength_multiplier
            elif cached_os < 0:
                cached_os = 1.0
            new_model = model
            new_clip = clip
            if model is not None and len(model_patches) > 0:
                new_model = model.clone()
                new_model.add_patches(model_patches, cached_os)
                self._update_model_size(new_model, model_patches)
            if clip is not None and len(clip_patches) > 0:
                new_clip = clip.clone()
                new_clip.add_patches(clip_patches, clip_strength_out)
                self._update_model_size(new_clip, clip_patches)
            logging.info(f"[LoRA Optimizer] Using cached merge result ({len(model_patches)} model + {len(clip_patches)} CLIP patches)")
            return (new_model, new_clip, report, None, lora_data)

        logging.info(f"[LoRA Optimizer] Starting analysis of {len(active_loras)} LoRAs")
        t_start = time.time()

        # Get key maps
        model_keys = self._get_model_keys(model)

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
        all_lora_prefixes = sorted(all_lora_prefixes)  # deterministic iteration order

        compute_device = self._get_compute_device()
        use_gpu = compute_device.type != "cpu"

        if _analysis_cache is not None:
            # Use pre-computed Pass 1 data (from AutoTuner)
            all_key_targets = _analysis_cache["all_key_targets"]
            prefix_stats = _analysis_cache["prefix_stats"]
            per_lora_stats = _analysis_cache["per_lora_stats"]
            pair_accum = _analysis_cache["pair_accum"]
            all_magnitude_samples = _analysis_cache["all_magnitude_samples"]
            prefix_count = _analysis_cache["prefix_count"]
            skipped_keys = _analysis_cache["skipped_keys"]
            pairs = [(i, j) for i in range(len(active_loras))
                             for j in range(i + 1, len(active_loras))]
            logging.info(f"[LoRA Optimizer] Using cached analysis ({prefix_count} prefixes, skipping Pass 1)")

        else:
            # =====================================================================
            # Pass 1 — Analysis (streaming: diffs computed, sampled, and discarded)
            # =====================================================================
            logging.info("[LoRA Optimizer] Pass 1: Analyzing weight diffs (streaming)...")
            logging.info(f"[LoRA Optimizer]   {len(all_lora_prefixes)} key prefixes across {len(active_loras)} LoRAs")
            logging.info(f"[LoRA Optimizer]   Compute device: {compute_device}"
                         f" ({'sequential' if use_gpu else 'threaded'})")
            t_pass1 = time.time()

            analysis = self._run_pass1_analysis(active_loras, all_lora_prefixes,
                                                model_keys, clip_keys, model, clip, compute_device)
            all_key_targets = analysis["all_key_targets"]
            per_lora_stats = analysis["per_lora_stats"]
            pair_accum = analysis["pair_accum"]
            all_magnitude_samples = analysis["all_magnitude_samples"]
            prefix_count = analysis["prefix_count"]
            prefix_stats = analysis["prefix_stats"]
            skipped_keys = analysis["skipped_keys"]
            pairs = analysis["pairs"]

            if prefix_count == 0:
                return (model, clip, "No compatible LoRA keys found. "
                        "LoRAs may be incompatible with this model architecture.", None, None)

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
                "key_filter": active_loras[i].get("key_filter", "all"),
            })

        # Full-rank detection: when average rank exceeds threshold, switch to
        # safer merge strategies that preserve information in high-dimensional patches.
        fr_preset = arch_preset.get("full_rank", {})
        fr_rank_threshold = fr_preset.get("rank_threshold", 256)
        global_avg_rank = (sum(s["avg_rank"] for s in lora_stats) / len(lora_stats)) if lora_stats else 0
        is_full_rank = global_avg_rank >= fr_rank_threshold
        if is_full_rank:
            logging.info(f"[LoRA Optimizer] Full-rank LoRAs detected (avg rank {global_avg_rank:.0f} >= {fr_rank_threshold})")
            if fr_preset.get("disable_slerp_upgrade", False):
                logging.info(f"[LoRA Optimizer]   SLERP upgrade disabled for full-rank patches")
            if fr_preset.get("prefer_sum_orthogonal", False):
                logging.info(f"[LoRA Optimizer]   Orthogonal full-rank patches will use weighted_sum (additive)")

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

        # Compatibility warnings for opposing LoRAs
        compatibility_warnings = []
        for (i, j), cos_sim in pairwise_similarities.items():
            if cos_sim < -0.1:
                compatibility_warnings.append({
                    "name_i": active_loras[i]['name'],
                    "name_j": active_loras[j]['name'],
                    "cosine_sim": cos_sim,
                })
                logging.warning(
                    f"[LoRA Optimizer] WARNING: {active_loras[i]['name']} vs {active_loras[j]['name']} "
                    f"have negative cosine similarity ({cos_sim:.3f}) — they work against each other"
                )

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
        global_avg_cos_sim = (sum(ps.get("cosine_sim", 0.0) for ps in pairwise_conflicts)
                              / len(pairwise_conflicts)) if pairwise_conflicts else 0.0
        mode, density, sign_method, reasoning = self._auto_select_params(
            avg_conflict_ratio, magnitude_ratio, magnitude_samples=all_magnitude_samples,
            avg_cos_sim=global_avg_cos_sim, behavior_profile=behavior_profile,
            arch_preset=arch_preset
        )
        del all_magnitude_samples

        # Apply merge strategy override from Conflict Editor
        # Skip when user explicitly chose weighted_sum_only (protects DPO/edit LoRAs)
        if merge_strategy_override and optimization_mode != "weighted_sum_only":
            if merge_strategy_override in ("ties", "weighted_average", "weighted_sum", "consensus", "slerp"):
                mode = merge_strategy_override
                reasoning.append(f"Merge mode overridden to '{mode}' by Conflict Editor")
            else:
                logging.warning(f"[LoRA Optimizer] Invalid merge_strategy_override '{merge_strategy_override}' — ignoring")

        logging.info(f"[LoRA Optimizer] Decision: {mode} ({reasoning[0] if reasoning else 'no reasoning'})")
        if mode == "ties":
            logging.info(f"[LoRA Optimizer]   density={density:.2f}, sign_method={sign_method}")

        # Auto-strength adjustment
        auto_strength_info = None
        scale_ratios = {}
        if auto_strength == "enabled":
            new_strengths, strength_reasoning = self._compute_auto_strengths(
                active_loras, lora_stats, pairwise_similarities=pairwise_similarities,
                arch_preset=arch_preset,
                detected_arch=getattr(self, '_detected_arch', None),
                auto_strength_floor=auto_strength_floor,
                is_full_rank=is_full_rank)

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
        sum_rank = sum(int(stat["avg_rank"]) for stat in lora_stats if stat["avg_rank"] > 0)
        compress_rank = 0  # 0 = disabled
        if compress_patches in ("non_ties", "all"):
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
        strategy_counts = {"weighted_sum": 0, "weighted_average": 0, "slerp": 0, "ties": 0, "consensus": 0}
        prefix_decisions = []  # list of (prefix, mode, conflict_ratio, n_loras) for block map

        # VRAM budget for patch storage
        vram_budget_bytes = 0
        gpu_patch_bytes = 0
        if vram_budget > 0 and use_gpu:
            free_vram = comfy.model_management.get_free_memory(compute_device)
            safety_margin = 256 * 1024 * 1024  # 256MB headroom
            usable = max(0, free_vram - safety_margin)
            vram_budget_bytes = int(usable * vram_budget)
            logging.info(f"[LoRA Optimizer] VRAM patch budget: {vram_budget_bytes // (1024**2)}MB "
                         f"({vram_budget*100:.0f}% of {usable // (1024**2)}MB free)")

        def _merge_one_prefix(lora_prefix, target_key, is_clip_key):
            """Recompute diffs for one prefix, merge, return patch or None."""
            nonlocal gpu_patch_bytes
            should_keep = vram_budget_bytes > 0 and gpu_patch_bytes < vram_budget_bytes
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
            pf_orthogonal = False
            pf_opposing = False
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
                        magnitude_samples=pf.get("magnitude_samples"),
                        avg_cos_sim=pf.get("avg_cos_sim", 0.0),
                        behavior_profile=behavior_profile,
                        arch_preset=arch_preset,
                        precomputed_density=pf.get("precomputed_density"),
                    )
                    # Upgrade weighted_average → slerp for 2+ non-opposing LoRAs.
                    # SLERP preserves magnitude better than weighted_average's /N reduction,
                    # which is critical for video LoRAs where motion energy matters.
                    # Skip for opposing LoRAs (cos < 0): SLERP interpolates between opposing
                    # directions while preserving magnitude, amplifying artifacts.
                    pf_raw_cos = pf.get("avg_cos_sim", 0.0)
                    pf_orthogonal = abs(pf_raw_cos) < arch_preset["orthogonal_cos_sim_max"]
                    pf_opposing = pf_raw_cos < 0
                    # Full-rank gate: skip SLERP upgrade — for full-rank patches the
                    # information is spread across all dimensions, and SLERP's
                    # hypersphere interpolation loses signal from both LoRAs.
                    if (pf_mode == "weighted_average" and pf["n_loras"] >= 2
                            and behavior_profile == "v1.2"
                            and not pf_opposing
                            and not (is_full_rank and fr_preset.get("disable_slerp_upgrade", False))):
                        pf_mode = "slerp"
                    # Full-rank gate: orthogonal full-rank patches → weighted_sum.
                    # Independent full-rank deltas should be added, not averaged.
                    if (is_full_rank and fr_preset.get("prefer_sum_orthogonal", False)
                            and pf_mode == "weighted_average" and pf_orthogonal):
                        pf_mode = "weighted_sum"

            # Apply merge strategy override from Conflict Editor (takes priority over auto-selection)
            # Skip when user explicitly chose weighted_sum_only (protects DPO/edit LoRAs)
            if (merge_strategy_override and optimization_mode != "weighted_sum_only"
                    and merge_strategy_override in ("ties", "weighted_average", "weighted_sum", "consensus", "slerp")):
                pf_mode = merge_strategy_override

            # LOW-RANK PATH: single-LoRA weighted_sum — keep low-rank matrices
            # instead of expanding to full-rank diff. Saves ~128x memory per key.
            # ComfyUI applies "lora" patches as: up @ down * (alpha/rank) * strength
            if pf_mode == "weighted_sum" and pf_n_loras <= 1:
                raw_n = prefix_stats.get(lora_prefix, {}).get("n_loras", 0)
                for i, item in enumerate(active_loras):
                    lora_info = self._get_lora_key_info(item["lora"], lora_prefix)
                    if lora_info is None:
                        continue
                    kf = item.get("key_filter", "all")
                    if kf == "shared_only" and raw_n < 2:
                        continue
                    if kf == "unique_only" and raw_n != 1:
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
                    if should_keep:
                        p_bytes = self._estimate_single_patch_bytes(patch)
                        if gpu_patch_bytes + p_bytes <= vram_budget_bytes:
                            patch = self._move_patch_to_device(patch, compute_device)
                            gpu_patch_bytes += p_bytes
                    return (target_key, is_clip_key, patch, pf_mode, lora_prefix, pf_conflict, max(pf_n_loras, 1), False, 0.0, 0.0)
                return None

            # FULL-RANK PATH: compute diffs on GPU, merge
            diffs_list = []
            diff_to_lora = []  # maps diffs_list index -> active_loras index
            storage_dtype = None  # native weight dtype before float32 upcast
            raw_n = prefix_stats.get(lora_prefix, {}).get("n_loras", 0)
            for i, item in enumerate(active_loras):
                lora_info = self._get_lora_key_info(item["lora"], lora_prefix)
                if lora_info is None:
                    continue
                kf = item.get("key_filter", "all")
                if kf == "shared_only" and raw_n < 2:
                    continue
                if kf == "unique_only" and raw_n != 1:
                    continue

                # Check diff cache
                cache_key = (lora_prefix, i)
                if _diff_cache is not None and cache_key in _diff_cache:
                    diff = _diff_cache.get(cache_key,
                                           device=compute_device if use_gpu else None).float()
                    if storage_dtype is None:
                        storage_dtype = lora_info[0].dtype

                    if is_clip_key and item["clip_strength"] is not None:
                        eff_strength = item["clip_strength"]
                    else:
                        eff_strength = item["strength"]
                        if scale_ratios:
                            eff_strength *= scale_ratios.get(i, 1.0)

                    diffs_list.append((diff, eff_strength))
                    diff_to_lora.append(i)
                    continue

                mat_up, mat_down, alpha, mid = lora_info
                rank = mat_down.shape[0]
                if storage_dtype is None:
                    storage_dtype = mat_up.dtype

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

                # Store in diff cache for subsequent candidates
                if _diff_cache is not None:
                    _diff_cache.put((lora_prefix, i), diff)

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
                    # Sign voting for conflict-mode masking.
                    # Negative eff_strength flips the effective direction, so we
                    # negate the sign vote accordingly.
                    if merge_quality != "standard" and diffs_list[0][0].dim() >= 2:
                        # Column-wise: each output neuron (row) votes as a unit
                        ref = diffs_list[0][0]
                        out_dim = ref.shape[0]
                        sign_sum = torch.zeros(out_dim, device=ref.device, dtype=torch.float32)
                        for diff, weight in diffs_list:
                            d = diff if weight >= 0 else -diff
                            sign_sum += d.reshape(out_dim, -1).to(dtype=torch.float32).sum(dim=1).sign()
                        majority_sign = torch.where(sign_sum >= 0, 1.0, -1.0)
                        majority_sign = majority_sign.reshape(-1, *([1] * (ref.dim() - 1))).expand_as(ref)
                        del sign_sum
                    else:
                        # Element-wise: each position votes independently
                        sign_sum = torch.zeros_like(diffs_list[0][0])
                        for diff, weight in diffs_list:
                            sign_sum += diff.sign() if weight >= 0 else -diff.sign()
                        majority_sign = torch.where(sign_sum >= 0, 1.0, -1.0)
                        del sign_sum

                    masked_diffs = []
                    for idx, (diff, weight) in enumerate(diffs_list):
                        cm = active_loras[diff_to_lora[idx]].get("conflict_mode", "all")
                        if cm == "low_conflict":
                            effective_diff = diff if weight >= 0 else -diff
                            diff = diff * ((effective_diff * majority_sign) > 0).float()
                        elif cm == "high_conflict":
                            effective_diff = diff if weight >= 0 else -diff
                            diff = diff * ((effective_diff * majority_sign) < 0).float()
                        masked_diffs.append((diff, weight))
                    diffs_list = masked_diffs
                    del majority_sign

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

            input_norms_mean = (sum(d.float().norm().item() * abs(w) for d, w in diffs_list)
                                / len(diffs_list)) if diffs_list else 0.0

            # For orthogonal/opposing weighted_average, force standard quality.
            # TALL-masks classifies most positions as "selfish" for orthogonal
            # LoRAs (each dominates independent positions) and adds them back at
            # full strength AFTER the merge, bypassing weighted_average's /N
            # normalization.  This converts weighted_average → weighted_sum (~Nx energy).
            # Opposing LoRAs have a similar problem: the magnitude calibration
            # in enhanced quality doesn't account for the directional cancellation
            # that weighted_average provides.
            pf_quality = merge_quality
            if (pf_mode == "weighted_average" and pf_opposing):
                pf_quality = "standard"

            merged_diff = self._merge_diffs(
                diffs_list, pf_mode,
                density=pf_density, majority_sign_method=pf_sign,
                compute_device=compute_device,
                sparsification=sparsification,
                sparsification_density=sparsification_density,
                sparsification_generator=sp_gen,
                merge_quality=pf_quality,
                dare_dampening=dare_dampening,
                keep_on_gpu=should_keep,
            )
            merged_norm = merged_diff.float().norm().item() if merged_diff is not None else 0.0
            diffs_list.clear()  # Free input diffs from GPU
            if merged_diff is None:
                return None
            # Compress full-rank diff to low-rank via SVD if requested
            # non_ties: skip compression on TIES prefixes (lossy); all: compress everything
            should_compress = (compress_rank > 0 and
                               (compress_patches == "all" or pf_mode != "ties"))
            # Downcast from float32 to native weight dtype (e.g. fp16/bf16)
            # to halve memory — ComfyUI handles dtype conversion when applying
            if storage_dtype is not None and merged_diff.dtype != storage_dtype:
                merged_diff = merged_diff.to(storage_dtype)
            if should_compress:
                patch = self._compress_to_lowrank(merged_diff, compress_rank, svd_device=resolved_svd_device)
                del merged_diff
                is_compressed = True
            else:
                patch = ("diff", (merged_diff,))
                is_compressed = False
            if should_keep:
                p_bytes = self._estimate_single_patch_bytes(patch)
                if gpu_patch_bytes + p_bytes <= vram_budget_bytes:
                    patch = self._move_patch_to_device(patch, compute_device)
                    gpu_patch_bytes += p_bytes
                else:
                    patch = self._move_patch_to_device(patch, torch.device("cpu"))
            return (target_key, is_clip_key, patch, pf_mode, lora_prefix, pf_conflict, max(pf_n_loras, 1), is_compressed, input_norms_mean, merged_norm)

        lowrank_count = 0
        total_input_energy = 0.0
        total_merged_energy = 0.0

        _overwrite_count = 0
        _overwrite_examples = []

        def _collect_merge_result(result):
            nonlocal processed_keys, lowrank_count, compressed_count
            nonlocal total_input_energy, total_merged_energy
            nonlocal _overwrite_count
            if result is None:
                return
            target_key, is_clip_key, patch, used_mode, prefix, conflict, n_loras, is_compressed, inp_norm, mrg_norm = result
            total_input_energy += inp_norm
            total_merged_energy += mrg_norm

            target_dict = clip_patches if is_clip_key else model_patches
            if target_key in target_dict:
                # Target-key collision: accumulate diffs instead of overwriting
                _overwrite_count += 1
                existing = target_dict[target_key]
                existing_diff = self._expand_patch_to_diff(existing)
                new_diff = self._expand_patch_to_diff(patch)
                accumulated = existing_diff + new_diff
                # Use the smaller dtype to save memory
                store_dt = existing_diff.dtype if existing_diff.dtype != torch.float32 else new_diff.dtype
                if store_dt != torch.float32:
                    accumulated = accumulated.to(store_dt)
                target_dict[target_key] = ("diff", (accumulated,))
                # Fix lowrank_count if existing was a low-rank adapter (now replaced by diff)
                if isinstance(existing, (LoRAAdapter, LoKrAdapter, LoHaAdapter)):
                    lowrank_count -= 1
                if len(_overwrite_examples) < 3:
                    _overwrite_examples.append(f"{'CLIP' if is_clip_key else 'MODEL'} {prefix} -> {target_key}")
            else:
                target_dict[target_key] = patch
                if isinstance(patch, (LoRAAdapter, LoKrAdapter, LoHaAdapter)):
                    lowrank_count += 1

            processed_keys += 1
            if is_compressed:
                compressed_count += 1
            strategy_counts[used_mode] = strategy_counts.get(used_mode, 0) + 1
            prefix_decisions.append((prefix, used_mode, conflict, n_loras))

        if use_gpu:
            prefix_list = list(all_key_targets.items())
            n_loras = len(active_loras)
            for idx, (lora_prefix, (target_key, is_clip_key)) in enumerate(prefix_list):
                # Prefetch next prefix's diffs from disk while merging current one
                if _diff_cache is not None and idx + 1 < len(prefix_list):
                    next_prefix = prefix_list[idx + 1][0]
                    _diff_cache.prefetch([(next_prefix, i) for i in range(n_loras)])
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
        if _overwrite_count > 0:
            logging.info(f"[LoRA Optimizer] {_overwrite_count} target-key collisions resolved "
                         f"(different LoRA key formats targeting the same model weight — diffs accumulated)")
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
                         f"{strategy_counts.get('consensus', 0)} cons, "
                         f"{strategy_counts.get('ties', 0)} ties")
        if compressed_count > 0:
            passthrough_count = lowrank_count - compressed_count
            logging.info(f"[LoRA Optimizer]   SVD-compressed: {compressed_count} patches "
                         f"(rank {compress_rank}), passthrough: {passthrough_count}, "
                         f"full-rank: {fullrank_count}")
        if vram_budget_bytes > 0:
            gpu_count = 0
            cpu_count = 0
            for p in list(model_patches.values()) + list(clip_patches.values()):
                data = p.weights if hasattr(p, 'weights') else p
                if isinstance(data, (tuple, list)):
                    for item in data:
                        if isinstance(item, torch.Tensor):
                            gpu_count += 1 if item.is_cuda else 0
                            cpu_count += 1 if not item.is_cuda else 0
                            break
                        elif isinstance(item, (tuple, list)):
                            for sub in item:
                                if isinstance(sub, torch.Tensor):
                                    gpu_count += 1 if sub.is_cuda else 0
                                    cpu_count += 1 if not sub.is_cuda else 0
                                    break
                            break
            logging.info(f"[LoRA Optimizer]   VRAM budget: {gpu_count} patches on GPU "
                         f"({gpu_patch_bytes // (1024**2)}MB), {cpu_count} on CPU")

        suggested_max_strength = None
        if total_merged_energy > 0 and total_input_energy > 0:
            norm_ratio = total_merged_energy / total_input_energy
            suggested_max_strength = max(1.0, min(1.0 / norm_ratio, arch_preset["suggested_max_strength_cap"]))

        # Free analysis data no longer needed (skip when AutoTuner owns the data)
        if _analysis_cache is None:
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
        # Use the FULL target_key (including offset tuple) as the dict key so
        # split Q/K/V patches targeting different slices of the same fused weight
        # don't overwrite each other (e.g., Z-Image fused QKV).
        reverse_key_map = {}
        for lora_prefix, (target_key, is_clip) in all_key_targets.items():
            reverse_key_map[target_key] = lora_prefix

        # Auto output strength: -1 = use suggested max strength
        auto_output_strength = False
        if output_strength < 0 and suggested_max_strength is not None:
            output_strength = suggested_max_strength
            auto_output_strength = True
            logging.info(f"[LoRA Optimizer] Auto output_strength: {output_strength:.2f} (suggested max)")
        elif output_strength < 0:
            output_strength = 1.0
            logging.info("[LoRA Optimizer] Auto output_strength: no suggestion available, using 1.0")

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
            self._update_model_size(new_model, model_patches)

        if clip is not None and len(clip_patches) > 0:
            new_clip = clip.clone()
            new_clip.add_patches(clip_patches, clip_strength_out)
            self._update_model_size(new_clip, clip_patches)

        merge_summary = {
            "keys_processed": processed_keys,
            "model_patches": len(model_patches),
            "clip_patches": len(clip_patches),
            "skipped_keys": skipped_keys,
            "output_strength": output_strength,
            "clip_strength": clip_strength_out,
            "suggested_max_strength": suggested_max_strength,
            "auto_output_strength": auto_output_strength,
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
            dare_dampening=dare_dampening,
            merge_quality=merge_quality,
            compatibility_warnings=compatibility_warnings,
            behavior_profile=behavior_profile,
            architecture_preset=preset_key,
            is_full_rank=is_full_rank,
        )

        # Bundle LORA_DATA for optional downstream saving
        lora_data = {
            "model_patches": model_patches,
            "clip_patches": clip_patches,
            "key_map": reverse_key_map,
            "output_strength": output_strength,
            "clip_strength": clip_strength_out,
            "suggested_max_strength": suggested_max_strength,
            "sum_rank": max(sum_rank, 64),
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
        if not _skip_report:
            report_path = self._save_report_to_disk(cache_key, lora_combo, auto_strength, report, selected_params)
            if report_path:
                logging.info(f"[LoRA Optimizer] Report saved to: {report_path}")

        logging.info(f"[LoRA Optimizer] Done! {processed_keys} keys processed ({time.time() - t_start:.1f}s total)")

        return (new_model, new_clip, report, None, lora_data)


class LoRAOptimizerSimple(LoRAOptimizer):
    """
    Simplified LoRA Optimizer with sensible defaults.  Exposes only model,
    LoRA stack, output strength, and optional CLIP inputs.  For sparsification,
    merge-quality, SVD-device, and other knobs use the Advanced variant.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Base model to merge LoRAs into."
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA stack from a LoRA Stack node."
                }),
                "output_strength": ("FLOAT", {
                    "default": 1.0, "min": -1.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global multiplier applied after merging. Set to -1 for auto (uses suggested max strength)."
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional CLIP model for text-encoder LoRA keys."
                }),
                "clip_strength_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier for CLIP LoRA strengths (stacks with per-LoRA clip_strength when provided)."
                }),
                "tuner_data": ("TUNER_DATA", {
                    "tooltip": "Connect from a Load Tuner Data or LoRA AutoTuner node. When connected, uses the best AutoTuner config instead of defaults."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "analysis_report")
    FUNCTION = "optimize_merge"
    DESCRIPTION = (
        "Simplified LoRA Optimizer — merges a LoRA stack with good defaults. "
        "Accepts optional tuner_data to use AutoTuner results. "
        "Use LoRA Optimizer (Advanced) for sparsification, merge quality, "
        "SVD device, and other fine-tuning options."
    )

    _SIMPLE_DEFAULTS = dict(
        auto_strength="enabled",
        auto_strength_floor=-1.0,
        free_vram_between_passes="disabled",
        vram_budget=0.0,
        optimization_mode="per_prefix",
        cache_patches="enabled",
        compress_patches="non_ties",
        svd_device="gpu",
        normalize_keys="enabled",
        sparsification="disabled",
        sparsification_density=0.7,
        dare_dampening=0.0,
        merge_strategy_override="",
        merge_quality="standard",
        behavior_profile="v1.2",
        architecture_preset="auto",
    )

    def optimize_merge(self, model, lora_stack, output_strength,
                       clip=None, clip_strength_multiplier=1.0,
                       tuner_data=None, **_kwargs):
        settings_source = "from_autotuner" if tuner_data is not None else "manual"
        result = super().optimize_merge(
            model, lora_stack, output_strength,
            clip=clip, clip_strength_multiplier=clip_strength_multiplier,
            tuner_data=tuner_data, settings_source=settings_source,
            **self._SIMPLE_DEFAULTS,
        )
        return result[:3]

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength,
                   clip=None, clip_strength_multiplier=1.0,
                   tuner_data=None, **_kwargs):
        return LoRAOptimizer.IS_CHANGED(
            model, lora_stack, output_strength,
            clip=clip, clip_strength_multiplier=clip_strength_multiplier,
            tuner_data=tuner_data,
            settings_source="from_autotuner" if tuner_data is not None else "manual",
            **cls._SIMPLE_DEFAULTS,
        )


class LoRAAutoTuner(LoRAOptimizer):
    """
    Automatic parameter sweep that finds the best merge configuration for
    a given LoRA stack. Runs Pass 1 analysis once, scores all parameter
    combinations via heuristic, then merges top-N candidates and measures
    output quality. Outputs the #1 result as MODEL/CLIP, plus a ranked
    report and TUNER_DATA for optional override via a Merge Selector node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Base model to merge LoRAs into."
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA stack from a LoRA Stack node."
                }),
                "output_strength": ("FLOAT", {
                    "default": 1.0, "min": -1.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Master volume for the merged result. Set to -1 for auto (uses suggested max strength)."
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional CLIP model for text-encoder LoRA keys."
                }),
                "clip_strength_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier for CLIP LoRA strengths."
                }),
                "top_n": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of top configurations to evaluate via actual merge. Higher = slower but explores more options."
                }),
                "normalize_keys": (["disabled", "enabled"], {
                    "default": "enabled",
                    "tooltip": "Makes LoRAs from different training tools compatible."
                }),
                "scoring_svd": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Enable SVD-based effective rank scoring for candidates. More thorough but much slower on large models."
                }),
                "scoring_device": (["cpu", "gpu"], {
                    "default": "gpu",
                    "tooltip": "Device for scoring computations. GPU is much faster when SVD scoring is enabled."
                }),
                "architecture_preset": (["auto", "sd_unet", "dit", "llm"], {
                    "default": "auto",
                    "tooltip": "Architecture-aware threshold tuning. 'auto' detects from LoRA keys."
                }),
                "auto_strength_floor": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum scale factor for auto-strength when LoRAs are orthogonal (independent). "
                               "-1 = auto (1.0 for video models like Wan/LTX to preserve motion, 0.85 for image models). "
                               "Set manually to override — e.g. 0.9 for less reduction, 1.0 to disable reduction for orthogonal LoRAs entirely."
                }),
                "record_dataset": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Record analysis metrics and scored configs to a JSONL dataset file for threshold tuning research. Saved to lora_optimizer_reports/autotuner_dataset.jsonl."
                }),
                "cache_patches": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": "Cache the AutoTuner result in RAM so re-execution with the same inputs skips the full sweep. Disable to free RAM after merge (recommended for video models)."
                }),
                "diff_cache_mode": (["disabled", "auto", "ram", "disk"], {
                    "default": "auto",
                    "tooltip": "Cache LoRA diffs across candidates to skip redundant computation. 'disabled' recomputes each time (slowest, no extra memory). 'auto' uses RAM up to diff_cache_ram_pct of free memory then spills to disk (recommended). 'ram' caches entirely in memory (~1.5GB SDXL, ~6GB Flux — fastest). 'disk' caches entirely to temp files (~1-10ms per diff vs 5-50ms to recompute). WARNING: ram/disk can use significant memory/storage on large models."
                }),
                "diff_cache_ram_pct": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Fraction of free system RAM to use for diff cache in 'auto' mode. 0.5 = use up to 50% of available RAM before spilling to disk."
                }),
                "vram_budget": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of free VRAM to use for storing merged patches. 0 = all CPU (default), 1.0 = use all free VRAM. Reduces RAM usage on GPU systems."
                }),
                "scoring_speed": (["full", "fast", "turbo", "turbo+"], {
                    "default": "turbo",
                    "tooltip": "Controls how many prefixes Phase 2 scores per candidate. "
                               "All candidates are scored on the same subset so ranking stays fair.\n\n"
                               "• full — Score every prefix. Best accuracy, slowest. Use when merging very different LoRAs (e.g. style + character + concept) where block behavior varies a lot.\n"
                               "• fast — Every 2nd prefix (~50%% faster). Higher accuracy than turbo, still much faster than full.\n"
                               "• turbo — Every 3rd prefix (~67%% faster). Works well when your LoRAs have similar conflict across blocks (e.g. multiple characters from the same trainer).\n"
                               "• turbo+ — Every 4th prefix (~75%% faster). Best for large models (DiT/Flux/WAN) or when iterating quickly. May miss subtle block-level differences on SD/SDXL."
                }),
                "output_mode": (["merge", "tuning_only"], {
                    "default": "merge",
                    "tooltip": "merge: output the best merged model (default). "
                               "tuning_only: skip the final merge — output the base model unchanged and pass tuner_data to a downstream LoRA Optimizer for merging. "
                               "Use tuning_only when chaining AutoTuner → Optimizer in a single model line."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "TUNER_DATA", "LORA_DATA")
    RETURN_NAMES = ("model", "clip", "report", "analysis_report", "tuner_data", "lora_data")
    FUNCTION = "auto_tune"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Automatically sweeps all merge parameters and finds the best "
        "configuration for your LoRA stack. Outputs the best merge directly. "
        "Connect TUNER_DATA to a Merge Selector node to try alternatives."
    )

    def auto_tune(self, model, lora_stack, output_strength, clip=None,
                  clip_strength_multiplier=1.0, top_n=3, normalize_keys="disabled",
                  scoring_svd="disabled", scoring_device="gpu",
                  architecture_preset="auto", auto_strength_floor=-1.0,
                  record_dataset="disabled",
                  cache_patches="enabled",
                  diff_cache_mode="disabled", diff_cache_ram_pct=0.5, vram_budget=0.0,
                  scoring_speed="full", output_mode="merge"):
        import hashlib, json

        # --- Normalize & validate stack ---
        normalized_stack = self._normalize_stack(lora_stack, normalize_keys=normalize_keys)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]
        if not active_loras:
            return (model, clip, "No active LoRAs in stack.", "", None, None)

        if len(active_loras) == 1:
            # Single LoRA: nothing to tune, delegate directly
            if output_mode == "tuning_only":
                return (model, clip, "Single LoRA detected -- tuning_only passthrough.", "", None, None)
            merged_model, merged_clip, report, _, lora_data = super().optimize_merge(
                model, lora_stack, output_strength,
                clip=clip, clip_strength_multiplier=clip_strength_multiplier,
                normalize_keys=normalize_keys, behavior_profile="v1.2",
                architecture_preset=architecture_preset, vram_budget=vram_budget,
            )
            return (merged_model, merged_clip,
                    "Single LoRA detected -- no parameters to tune.\n\n" + report, report, None, lora_data)

        # Compute lora_hash for cache validation
        hash_input = json.dumps([(l["name"], l["strength"]) for l in active_loras],
                                sort_keys=True)
        lora_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Check AutoTuner cache (output_mode excluded so mode switches reuse cached sweep)
        at_cache_key = hashlib.sha256(
            f"{lora_hash}|os={output_strength}|csm={clip_strength_multiplier}"
            f"|top_n={top_n}|nk={normalize_keys}|ss={scoring_svd}"
            f"|ap={architecture_preset}|vb={vram_budget}"
            f"|spd={scoring_speed}|mid={id(model)}".encode()
        ).hexdigest()[:16]
        if cache_patches == "enabled" and hasattr(self, '_autotuner_cache') and at_cache_key in self._autotuner_cache:
            cached_result, cached_mode = self._autotuner_cache[at_cache_key]
            if output_mode == "tuning_only":
                # Return base model + cached tuner_data (no merge needed)
                logging.info("[LoRA AutoTuner] Using cached result (tuning_only passthrough)")
                return (model, clip, cached_result[2], "", cached_result[4], None)
            elif cached_mode == "merge":
                # Cached has merged model, return as-is
                logging.info("[LoRA AutoTuner] Using cached result")
                return cached_result
            # else: cached was tuning_only but now need merge — fall through to re-run

        # --- Pass 1: Analysis (run once, reuse for all configs) ---
        model_keys = self._get_model_keys(model)
        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})

        # Collect all prefixes (match parent's suffix set)
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
        all_lora_prefixes = sorted(all_lora_prefixes)

        if not all_lora_prefixes:
            return (model, clip, "No LoRA prefixes found.", "", None, None)

        # Determine compute device
        compute_device = self._get_compute_device()
        use_gpu = compute_device.type != "cpu"

        # Run Pass 1 analysis
        logging.info(f"[LoRA AutoTuner] Pass 1: Analyzing {len(all_lora_prefixes)} prefixes...")
        t_start = time.time()
        # Progress bar: analysis prefixes + top_n merges (+ 1 final merge when subsampling)
        n_pbar_merges = top_n + (1 if scoring_speed != "full" and top_n > 1 and output_mode != "tuning_only" else 0)
        pbar = comfy.utils.ProgressBar(len(all_lora_prefixes) + n_pbar_merges)

        analysis = self._run_pass1_analysis(active_loras, all_lora_prefixes,
                                            model_keys, clip_keys, model, clip, compute_device,
                                            progress_callback=lambda: pbar.update(1))
        all_key_targets = analysis["all_key_targets"]
        per_lora_stats = analysis["per_lora_stats"]
        pair_accum = analysis["pair_accum"]
        all_magnitude_samples = analysis["all_magnitude_samples"]
        prefix_count = analysis["prefix_count"]
        prefix_stats = analysis["prefix_stats"]
        skipped_keys = analysis["skipped_keys"]
        pairs = analysis["pairs"]

        if prefix_count == 0:
            return (model, clip, "No compatible LoRA keys found.", "", None, None)

        t_analysis = time.time() - t_start
        logging.info(f"[LoRA AutoTuner] Analysis complete: {prefix_count} prefixes ({t_analysis:.1f}s)")

        # Finalize global stats
        l2_means = []
        for i, stat in enumerate(per_lora_stats):
            l2_mean = sum(stat["l2_norms"]) / len(stat["l2_norms"]) if stat["l2_norms"] else 0
            l2_means.append(l2_mean)

        total_overlap = sum(pair_accum[p][0] for p in pairs)
        total_conflict = sum(pair_accum[p][1] for p in pairs)
        avg_conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0

        pairwise_similarities = {}
        for i, j in pairs:
            ov, conf, dot, na_sq, nb_sq = pair_accum[(i, j)]
            denom = math.sqrt(na_sq) * math.sqrt(nb_sq)
            pairwise_similarities[(i, j)] = dot / denom if denom > 0 else 0.0

        valid_l2 = [m for m in l2_means if m > 0]
        magnitude_ratio = max(valid_l2) / min(valid_l2) if len(valid_l2) >= 2 else 1.0

        avg_cos_sim = (sum(pairwise_similarities.values())
                       / len(pairwise_similarities)) if pairwise_similarities else 0.0

        _analysis_cache = {
            "all_key_targets": all_key_targets,
            "prefix_stats": prefix_stats,
            "per_lora_stats": per_lora_stats,
            "pair_accum": pair_accum,
            "all_magnitude_samples": all_magnitude_samples,
            "prefix_count": prefix_count,
            "skipped_keys": skipped_keys,
        }

        # Resolve architecture preset for scoring
        tuner_preset_key, tuner_arch_preset = _resolve_arch_preset(
            architecture_preset, getattr(self, '_detected_arch', None) or 'unknown')
        logging.info(f"[LoRA AutoTuner] Architecture preset: {tuner_preset_key} ({tuner_arch_preset['display_name']})")

        # --- Phase 1: Score all parameter combos (heuristic, fast) ---
        logging.info("[LoRA AutoTuner] Phase 1: Scoring parameter grid...")
        grid = _generate_param_grid()
        scored = []
        for config in grid:
            h_score = _score_config_heuristic(
                config, avg_conflict_ratio, avg_cos_sim,
                magnitude_ratio, prefix_stats, arch_preset=tuner_arch_preset)
            scored.append((h_score, config))
        scored.sort(key=lambda x: x[0], reverse=True)
        logging.info(f"[LoRA AutoTuner] Scored {len(grid)} combos in {time.time() - t_start:.1f}s")
        for i in range(min(5, len(scored))):
            c = scored[i][1]
            logging.info(f"[LoRA AutoTuner]   #{i+1} heuristic={scored[i][0]:.3f}: "
                         f"{c['merge_mode']}/{c['merge_quality']}"
                         f"{' +' + c['sparsification'] if c['sparsification'] != 'disabled' else ''}"
                         f" auto_str={c['auto_strength']} {c['optimization_mode']}")

        # Pre-compute per-prefix density from magnitude samples before freeing them.
        # Phase 2's per-prefix auto_select_params needs density but magnitude_samples
        # are large — store the scalar result and free the raw samples.
        for pf in prefix_stats.values():
            samples = pf.get("magnitude_samples")
            if samples and pf.get("n_loras", 0) > 1:
                pf["precomputed_density"] = self._estimate_density_from_samples(
                    samples, arch_preset=tuner_arch_preset)
            pf.pop("magnitude_samples", None)
        _analysis_cache["all_magnitude_samples"] = []
        gc.collect()

        # --- Prefix subsampling for Phase 2 (scoring_speed) ---
        use_subsampling = scoring_speed != "full" and top_n > 1
        scoring_cache = _analysis_cache  # default: use full cache
        if use_subsampling:
            step = {"fast": 2, "turbo": 3, "turbo+": 4}[scoring_speed]
            # Separate single-LoRA prefixes (always included, identical across candidates)
            # from multi-LoRA prefixes (where merge strategy matters)
            single_lora_targets = {}
            multi_lora_targets = {}
            for pfx, tgt in all_key_targets.items():
                if prefix_stats.get(pfx, {}).get("n_loras", 0) <= 1:
                    single_lora_targets[pfx] = tgt
                else:
                    multi_lora_targets[pfx] = tgt
            # Sort multi-LoRA prefixes by conflict_ratio descending (highest conflict first)
            sorted_multi = sorted(
                multi_lora_targets.keys(),
                key=lambda p: prefix_stats.get(p, {}).get("conflict_ratio", 0.0),
                reverse=True,
            )
            # Keep every Nth prefix from the sorted list
            subsampled_multi = {p: multi_lora_targets[p] for p in sorted_multi[::step]}
            scoring_targets = {**single_lora_targets, **subsampled_multi}
            scoring_cache = {**_analysis_cache, "all_key_targets": scoring_targets}
            logging.info(f"[LoRA AutoTuner] Scoring speed '{scoring_speed}': "
                         f"scoring {len(scoring_targets)}/{len(all_key_targets)} prefixes "
                         f"({len(single_lora_targets)} single-LoRA + "
                         f"{len(subsampled_multi)}/{len(multi_lora_targets)} multi-LoRA)")

        # --- Phase 2: Merge top-N and measure ---
        # Initialize diff cache for Phase 2
        _diff_cache = None
        if diff_cache_mode != "disabled":
            _diff_cache = _DiffCache(mode=diff_cache_mode, ram_pct=diff_cache_ram_pct)
            logging.info(f"[LoRA AutoTuner] Diff cache enabled (mode={diff_cache_mode})")

        # Only keep the current best in memory to avoid accumulating ~40GB per candidate.
        top_candidates = scored[:top_n]
        results = []
        best_model = None
        best_clip = None
        best_lora_data = None
        best_analysis_report = ""
        best_score = -1.0
        best_config = None
        logging.info(f"[LoRA AutoTuner] Phase 2: Merging and measuring top {len(top_candidates)} candidates...")

        for rank_idx, (h_score, config) in enumerate(top_candidates):
            logging.info(f"[LoRA AutoTuner]   Candidate {rank_idx + 1}/{len(top_candidates)}: "
                         f"{config['merge_mode']}, {config['merge_quality']}"
                         f"{', ' + config['sparsification'] if config['sparsification'] != 'disabled' else ''}"
                         f" (heuristic={h_score:.3f})...")
            t_merge = time.time()

            # In per_prefix mode, let the optimizer auto-select strategy per prefix.
            # merge_strategy_override would force one mode everywhere, defeating per-prefix logic.
            strategy_override = config["merge_mode"] if config["optimization_mode"] == "global" else ""

            merged_model, merged_clip, _report, _, lora_data = super().optimize_merge(
                model, lora_stack, output_strength,
                clip=clip,
                clip_strength_multiplier=clip_strength_multiplier,
                auto_strength=config["auto_strength"],
                auto_strength_floor=auto_strength_floor,
                optimization_mode=config["optimization_mode"],
                sparsification=config["sparsification"],
                sparsification_density=config["sparsification_density"],
                dare_dampening=config["dare_dampening"],
                merge_quality=config["merge_quality"],
                merge_strategy_override=strategy_override,
                free_vram_between_passes="disabled",
                vram_budget=vram_budget,
                cache_patches="disabled",
                compress_patches="disabled",
                svd_device="gpu",
                normalize_keys=normalize_keys,
                behavior_profile="v1.2",
                architecture_preset=architecture_preset,
                _analysis_cache=scoring_cache,
                _diff_cache=_diff_cache,
                _skip_report=True,
            )

            # Measure output quality (single-LoRA prefixes may still produce
            # LoRAAdapter patches; _score_merge_result handles both formats)
            m_patches = lora_data["model_patches"] if lora_data else {}
            c_patches = lora_data["clip_patches"] if lora_data else {}
            compute_svd = scoring_svd == "enabled"
            score_dev = torch.device("cuda") if scoring_device == "gpu" and torch.cuda.is_available() else None
            t_score = time.time()
            measured = _score_merge_result(m_patches, c_patches, compute_svd=compute_svd, score_device=score_dev)
            t_score_elapsed = time.time() - t_score

            t_elapsed = time.time() - t_merge
            logging.info(f"[LoRA AutoTuner]   Candidate #{rank_idx + 1}: "
                         f"measured={measured['composite_score']:.3f} "
                         f"(merge {t_elapsed - t_score_elapsed:.1f}s + score {t_score_elapsed:.1f}s)")
            pbar.update(1)

            if use_subsampling or output_mode == "tuning_only":
                # When subsampling or tuning_only, discard all candidates
                del merged_model, merged_clip, lora_data
            elif measured["composite_score"] > best_score:
                # Keep only the best model in memory, free the rest immediately
                # Free previous best before replacing
                del best_model, best_clip, best_lora_data
                best_model = merged_model
                best_clip = merged_clip
                best_lora_data = lora_data
                best_analysis_report = _report
            else:
                # Discard this candidate's heavy objects immediately
                del merged_model, merged_clip, lora_data
            if measured["composite_score"] > best_score:
                best_score = measured["composite_score"]
                best_config = config
            del m_patches, c_patches  # Drop patch-dict references so tensors can free
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()

            results.append({
                "rank": rank_idx + 1,
                "score_heuristic": h_score,
                "score_measured": measured["composite_score"],
                "config": config,
                "metrics": {
                    "norm_preservation": measured.get("norm_mean", 0.0),
                    "effective_rank_mean": measured.get("effective_rank_mean", 0.0),
                    "sparsity_mean": measured.get("sparsity_mean", 0.0),
                    "norm_cv": measured.get("norm_cv", 0.0),
                },
            })

        # Final full merge when subsampling was used (skip if tuning_only)
        if use_subsampling and best_config is not None and output_mode != "tuning_only":
            logging.info(f"[LoRA AutoTuner] Final merge with winning config "
                         f"({best_config['merge_mode']}, {best_config['merge_quality']})...")
            t_final = time.time()
            strategy_override = best_config["merge_mode"] if best_config["optimization_mode"] == "global" else ""
            best_model, best_clip, best_analysis_report, _, best_lora_data = super().optimize_merge(
                model, lora_stack, output_strength,
                clip=clip,
                clip_strength_multiplier=clip_strength_multiplier,
                auto_strength=best_config["auto_strength"],
                auto_strength_floor=auto_strength_floor,
                optimization_mode=best_config["optimization_mode"],
                sparsification=best_config["sparsification"],
                sparsification_density=best_config["sparsification_density"],
                dare_dampening=best_config["dare_dampening"],
                merge_quality=best_config["merge_quality"],
                merge_strategy_override=strategy_override,
                free_vram_between_passes="disabled",
                vram_budget=vram_budget,
                cache_patches="disabled",
                compress_patches="disabled",
                svd_device="gpu",
                normalize_keys=normalize_keys,
                behavior_profile="v1.2",
                architecture_preset=architecture_preset,
                _analysis_cache=_analysis_cache,
                _diff_cache=_diff_cache,
                _skip_report=True,
            )
            logging.info(f"[LoRA AutoTuner] Final merge complete ({time.time() - t_final:.1f}s)")
            pbar.update(1)

        if _diff_cache is not None:
            n_ram = len(_diff_cache._ram_store)
            n_disk = len(_diff_cache._disk_store)
            logging.info(f"[LoRA AutoTuner] Diff cache: {n_ram + n_disk} entries, "
                         f"{_diff_cache.size_mb():.1f} MB "
                         f"({n_ram} ram, {n_disk} disk)")
            _diff_cache.clear()
            del _diff_cache
        del all_magnitude_samples
        del _analysis_cache
        self.loaded_loras.clear()
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        # Sort by measured score
        results.sort(key=lambda x: x["score_measured"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        best = results[0]
        best["merged_model"] = best_model
        best["merged_clip"] = best_clip
        best["lora_data"] = best_lora_data

        # Build TUNER_DATA (exclude model/clip objects)
        tuner_data = {
            "version": 1,
            "lora_hash": lora_hash,
            "normalize_keys": normalize_keys,
            "architecture_preset": architecture_preset,
            "analysis_summary": {
                "n_loras": len(active_loras),
                "prefix_count": prefix_count,
                "avg_conflict_ratio": avg_conflict_ratio,
                "avg_cosine_sim": avg_cos_sim,
                "magnitude_ratio": magnitude_ratio,
            },
            "top_n": [{
                "rank": r["rank"],
                "score_heuristic": r["score_heuristic"],
                "score_measured": r["score_measured"],
                "config": r["config"],
                "metrics": r["metrics"],
            } for r in results],
        }

        # Save dataset entry for threshold tuning (opt-in)
        if record_dataset == "enabled":
            self._save_tuner_dataset_entry(
                tuner_data, active_loras, prefix_stats,
                getattr(self, '_detected_arch', None))
        prefix_stats.clear()

        # Build report
        suggested_max = best_lora_data.get("suggested_max_strength") if best_lora_data else None
        report = self._build_autotuner_report(
            results, tuner_data["analysis_summary"], output_strength,
            suggested_max_strength=suggested_max, scoring_speed=scoring_speed)

        if output_mode == "tuning_only":
            # Passthrough: return base model unchanged, no merge result
            for r in results:
                r.pop("merged_model", None)
                r.pop("merged_clip", None)
                r.pop("lora_data", None)
            del results, best, best_model, best_clip, best_lora_data, best_analysis_report, active_loras
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()
            logging.info("[LoRA AutoTuner] tuning_only mode — returning base model (no merge)")
            result = (model, clip, report, "", tuner_data, None)
            if cache_patches == "enabled":
                self._autotuner_cache = {at_cache_key: (result, "tuning_only")}
            else:
                self._autotuner_cache = {}
            return result

        # Extract return values, then free heavy intermediates
        ret_model = best["merged_model"]
        ret_clip = best["merged_clip"]
        ret_lora_data = best.get("lora_data")
        ret_analysis_report = best_analysis_report
        for r in results:
            r.pop("merged_model", None)
            r.pop("merged_clip", None)
            r.pop("lora_data", None)
        del results, best, best_model, best_clip, best_lora_data, best_analysis_report, active_loras
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        result = (ret_model, ret_clip, report, ret_analysis_report, tuner_data, ret_lora_data)
        if cache_patches == "enabled":
            self._autotuner_cache = {at_cache_key: (result, "merge")}
        else:
            self._autotuner_cache = {}
            logging.info("[LoRA AutoTuner] Patch cache disabled — RAM freed after merge")

        return result

    def _save_tuner_dataset_entry(self, tuner_data, active_loras, prefix_stats,
                                  detected_arch):
        """
        Append one JSONL entry to the AutoTuner dataset for threshold tuning.
        Each entry records: analysis metrics, per-prefix stats distribution,
        detected architecture, all scored configs with measured quality.
        Failures are silently logged — never blocks the merge.
        """
        try:
            user_dir = folder_paths.get_user_directory()
            dataset_dir = os.path.join(user_dir, "lora_optimizer_reports")
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_path = os.path.join(dataset_dir, "autotuner_dataset.jsonl")

            # Summarize per-prefix conflict/cosine distributions
            conflict_ratios = []
            cos_sims = []
            mag_ratios = []
            for pf in prefix_stats.values():
                if pf.get("n_loras", 0) > 1:
                    conflict_ratios.append(pf["conflict_ratio"])
                    cos_sims.append(pf.get("avg_cos_sim", 0.0))
                    mag_ratios.append(pf.get("magnitude_ratio", 1.0))

            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "detected_arch": detected_arch,
                "architecture_preset": tuner_data.get("architecture_preset", "auto"),
                "lora_hash": tuner_data.get("lora_hash", ""),
                "lora_names": [l["name"] for l in active_loras],
                "analysis": tuner_data["analysis_summary"],
                "prefix_distributions": {
                    "conflict_ratios": {
                        "min": min(conflict_ratios) if conflict_ratios else 0,
                        "max": max(conflict_ratios) if conflict_ratios else 0,
                        "mean": sum(conflict_ratios) / len(conflict_ratios) if conflict_ratios else 0,
                        "std": (sum((x - sum(conflict_ratios) / len(conflict_ratios)) ** 2
                                    for x in conflict_ratios) / len(conflict_ratios)) ** 0.5
                               if len(conflict_ratios) > 1 else 0,
                        "n": len(conflict_ratios),
                    },
                    "cos_sims": {
                        "min": min(cos_sims) if cos_sims else 0,
                        "max": max(cos_sims) if cos_sims else 0,
                        "mean": sum(cos_sims) / len(cos_sims) if cos_sims else 0,
                        "n": len(cos_sims),
                    },
                    "mag_ratios": {
                        "min": min(mag_ratios) if mag_ratios else 1,
                        "max": max(mag_ratios) if mag_ratios else 1,
                        "mean": sum(mag_ratios) / len(mag_ratios) if mag_ratios else 1,
                        "n": len(mag_ratios),
                    },
                },
                "top_n": tuner_data["top_n"],
            }

            with open(dataset_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            logging.info(f"[LoRA AutoTuner] Dataset entry saved to: {dataset_path}")
        except Exception as e:
            logging.warning(f"[LoRA AutoTuner] Failed to save dataset entry: {e}")

    def _build_autotuner_report(self, results, analysis_summary, output_strength,
                               suggested_max_strength=None, scoring_speed="full"):
        """Build the ranked report for AutoTuner results."""
        lines = []
        lines.append("=" * 54)
        lines.append("  LoRA AutoTuner Results")
        lines.append("=" * 54)
        lines.append("")
        lines.append("  Analysis Summary:")
        s = analysis_summary
        lines.append(f"    LoRAs: {s['n_loras']} | Prefixes: {s['prefix_count']} "
                     f"| Avg conflict: {s['avg_conflict_ratio']:.1%}")
        lines.append(f"    Avg cosine similarity: {s['avg_cosine_sim']:.2f} "
                     f"| Magnitude ratio: {s['magnitude_ratio']:.1f}x")
        lines.append(f"    Output strength: {output_strength}")
        if suggested_max_strength is not None:
            lines.append(f"    Suggested max output_strength: {suggested_max_strength:.2f}")
        if scoring_speed != "full":
            lines.append(f"    Scoring speed: {scoring_speed} (subsampled prefix scoring)")
        lines.append("")
        lines.append("  " + "-" * 38)
        lines.append(f"  Top {len(results)} Configurations")
        lines.append("  " + "-" * 38)

        for r in results:
            lines.append("")
            c = r["config"]
            m = r["metrics"]
            marker = " (applied to output)" if r["rank"] == 1 else ""
            star = " \u2605" if r["rank"] == 1 else ""
            lines.append(f"  #{r['rank']}{star}{marker}"
                         f"          Score: {r['score_measured']:.2f}")
            mode_display = "per-prefix (auto)" if c["optimization_mode"] == "per_prefix" else c["merge_mode"]
            lines.append(f"    Mode: {mode_display} | Quality: {c['merge_quality']}")
            if c["sparsification"] != "disabled":
                spars_info = f"{c['sparsification']} (density={c['sparsification_density']}"
                if c["dare_dampening"] > 0:
                    spars_info += f", dampening={c['dare_dampening']}"
                spars_info += ")"
                lines.append(f"    Sparsification: {spars_info}")
            else:
                lines.append(f"    Sparsification: disabled")
            lines.append(f"    Auto-strength: {c['auto_strength']} "
                         f"| Optimization: {c['optimization_mode']}")
            if m.get("effective_rank_mean", 0) > 0:
                lines.append(f"    Effective rank: {m['effective_rank_mean']:.1f} "
                             f"| Sparsity: {m.get('sparsity_mean', 0):.1%}")

        lines.append("")
        lines.append("  To use a different config: connect TUNER_DATA")
        lines.append("  to a Merge Selector node and set selection=N")
        lines.append("=" * 54)
        return "\n".join(lines)

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, clip=None,
                   clip_strength_multiplier=1.0, top_n=3, normalize_keys="disabled",
                   scoring_svd="disabled", scoring_device="gpu",
                   architecture_preset="auto", record_dataset="disabled",
                   cache_patches="enabled",
                   diff_cache_mode="disabled", diff_cache_ram_pct=0.5,
                   vram_budget=0.0, scoring_speed="full", output_mode="merge"):
        return (id(model), id(lora_stack), output_strength, clip_strength_multiplier, top_n,
                normalize_keys, scoring_svd, scoring_device, architecture_preset,
                vram_budget, record_dataset, scoring_speed, output_mode)


class LoRAMergeSelector(LoRAOptimizer):
    """
    Applies a specific merge configuration from AutoTuner results.
    Connect TUNER_DATA from a LoRA AutoTuner node and set the selection
    index to choose which ranked configuration to apply.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Base model (same one connected to the AutoTuner)."
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA stack (same one connected to the AutoTuner)."
                }),
                "tuner_data": ("TUNER_DATA", {
                    "tooltip": "Connect from the LoRA AutoTuner's tuner_data output."
                }),
                "selection": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Which ranked configuration to apply (1 = best, 2 = second best, etc.)."
                }),
                "output_strength": ("FLOAT", {
                    "default": 1.0, "min": -1.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Master volume for the merged result. Set to -1 for auto (uses suggested max strength)."
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional CLIP model for text-encoder LoRA keys."
                }),
                "clip_strength_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier for CLIP LoRA strengths."
                }),
                "vram_budget": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of free VRAM to use for storing merged patches. 0 = all CPU (default), 1.0 = use all free VRAM. Reduces RAM usage on GPU systems."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "LORA_DATA")
    RETURN_NAMES = ("model", "clip", "report", "lora_data")
    FUNCTION = "select_merge"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Applies a specific merge configuration from LoRA AutoTuner results. "
        "Set selection to choose which ranked configuration to use."
    )

    def select_merge(self, model, lora_stack, tuner_data, selection,
                     output_strength, clip=None, clip_strength_multiplier=1.0,
                     vram_budget=0.0):
        import hashlib, json

        if tuner_data is None or "top_n" not in tuner_data:
            return (model, clip, "Error: No valid TUNER_DATA provided.", None)

        # Validate lora_hash (filter zero-strength to match AutoTuner)
        nk = tuner_data.get("normalize_keys", "disabled")
        all_loras = self._normalize_stack(lora_stack, normalize_keys=nk)
        active_loras = [item for item in all_loras if item["strength"] != 0]
        hash_input = json.dumps([(l["name"], l["strength"]) for l in active_loras],
                                sort_keys=True)
        current_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        if current_hash != tuner_data.get("lora_hash", ""):
            logging.warning("[Merge Selector] LoRA stack has changed since AutoTuner ran. "
                            "Results may not match. Re-run AutoTuner for accurate results.")

        # Get selected config
        top_n = tuner_data["top_n"]
        if selection < 1 or selection > len(top_n):
            return (model, clip,
                    f"Error: selection={selection} out of range (1-{len(top_n)}).", None)

        entry = top_n[selection - 1]
        config = entry["config"]

        logging.info(f"[Merge Selector] Applying config #{selection}: "
                     f"{config['merge_mode']}, {config['merge_quality']}")

        # Run merge with selected config
        # In per_prefix mode, let the optimizer auto-select strategy per prefix.
        strategy_override = config["merge_mode"] if config["optimization_mode"] == "global" else ""

        merged_model, merged_clip, _report, _, _lora_data = super().optimize_merge(
            model, lora_stack, output_strength,
            clip=clip,
            clip_strength_multiplier=clip_strength_multiplier,
            auto_strength=config["auto_strength"],
            optimization_mode=config["optimization_mode"],
            sparsification=config["sparsification"],
            sparsification_density=config["sparsification_density"],
            dare_dampening=config["dare_dampening"],
            merge_quality=config["merge_quality"],
            merge_strategy_override=strategy_override,
            free_vram_between_passes="disabled",
            vram_budget=vram_budget,
            cache_patches="enabled",
            compress_patches="non_ties",
            svd_device="gpu",
            normalize_keys=tuner_data.get("normalize_keys", "disabled"),
            behavior_profile="v1.2",
            architecture_preset=tuner_data.get("architecture_preset", "auto"),
        )

        # Build report for this selection
        lines = []
        lines.append(f"Merge Selector \u2014 Applied config #{selection}")
        mode_display = "per-prefix (auto)" if config["optimization_mode"] == "per_prefix" else config["merge_mode"]
        lines.append(f"  Mode: {mode_display} | Quality: {config['merge_quality']}")
        if config["sparsification"] != "disabled":
            lines.append(f"  Sparsification: {config['sparsification']} "
                         f"(density={config['sparsification_density']})")
        lines.append(f"  Auto-strength: {config['auto_strength']} "
                     f"| Optimization: {config['optimization_mode']}")
        lines.append(f"  Heuristic score: {entry['score_heuristic']:.3f} "
                     f"| Measured score: {entry['score_measured']:.3f}")
        report = "\n".join(lines)

        return (merged_model, merged_clip, report, _lora_data)

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, tuner_data, selection,
                   output_strength, clip=None, clip_strength_multiplier=1.0,
                   vram_budget=0.0):
        return (id(model), id(tuner_data), selection, output_strength, clip_strength_multiplier)


class WanVideoLoRAOptimizer(LoRAOptimizer):
    """
    WanVideo variant of the LoRA Optimizer. Accepts WANVIDEOMODEL instead of
    MODEL, skips CLIP, and applies merged LoRA patches in-memory.

    All merging algorithms (TIES, DARE/DELLA, SVD compression, auto-strength,
    conflict analysis) are inherited from LoRAOptimizer. Wan LoRA key
    normalization (LyCORIS, diffusers, Fun LoRA, finetrainer) is
    already handled by the parent's _normalize_keys_wan.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = LoRAOptimizer.INPUT_TYPES()
        # Replace MODEL with WANVIDEOMODEL
        base["required"]["model"] = ("WANVIDEOMODEL", {
            "tooltip": "Your WanVideo model from WanVideoModelLoader."
        })
        # Remove CLIP-related inputs (WanVideo doesn't use CLIP)
        base["optional"].pop("clip", None)
        base["optional"].pop("clip_strength_multiplier", None)
        base["optional"].pop("free_vram_between_passes", None)
        # Change defaults for video models
        base["optional"]["cache_patches"] = (["disabled", "enabled"], {
            "default": "disabled",
            "tooltip": "Keep the merge result in memory so re-running the workflow is instant. Disabled by default for large video models to save RAM."
        })
        base["optional"]["normalize_keys"] = (["enabled", "disabled"], {
            "default": "enabled",
            "tooltip": "Normalizes LoRA keys from different training tools (LyCORIS, diffusers, finetrainer, etc.) to a common format. Enabled by default for WanVideo LoRAs."
        })
        base["optional"]["architecture_preset"] = (["auto", "dit", "sd_unet", "llm"], {
            "default": "dit",
            "tooltip": "Architecture-aware threshold tuning. Default 'dit' for WanVideo models. "
                       "'auto' detects from LoRA keys."
        })
        base["optional"].pop("tuner_data", None)
        base["optional"].pop("settings_source", None)
        return base

    RETURN_TYPES = ("WANVIDEOMODEL", "STRING", "LORA_DATA")
    RETURN_NAMES = ("model", "analysis_report", "lora_data")
    FUNCTION = "optimize_merge"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "WanVideo LoRA Optimizer — merges multiple WanVideo LoRAs using "
        "conflict-aware algorithms (TIES, DARE, auto-strength). "
        "Connect after WanVideoModelLoader, before WanVideoSampler."
    )

    def _get_model_keys(self, model):
        model_keys = super()._get_model_keys(model)
        # WanVideo models have ._orig_mod. in state_dict keys from torch.compile.
        # The core model_lora_keys_unet creates entries WITH _orig_mod, but LoRA
        # files have prefixes WITHOUT it. Add stripped versions pointing to the
        # original target keys so prefixes match.
        augmented = {}
        for prefix, target in model_keys.items():
            if '._orig_mod.' in prefix or '._orig_mod' in prefix:
                stripped = prefix.replace('._orig_mod.', '.').replace('._orig_mod', '')
                if stripped not in model_keys:
                    augmented[stripped] = target
        model_keys.update(augmented)
        return model_keys

    def optimize_merge(self, model, lora_stack, output_strength, **kwargs):
        kwargs.pop("clip", None)
        kwargs.pop("clip_strength_multiplier", None)
        kwargs.pop("free_vram_between_passes", None)
        model_out, _clip, report, _, lora_data = super().optimize_merge(
            model, lora_stack, output_strength,
            clip=None, clip_strength_multiplier=1.0, **kwargs
        )
        return (model_out, report, lora_data)

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, **kwargs):
        kwargs.pop("clip", None)
        kwargs.pop("clip_strength_multiplier", None)
        kwargs.pop("free_vram_between_passes", None)
        return LoRAOptimizer.IS_CHANGED(
            model, lora_stack, output_strength,
            clip=None, clip_strength_multiplier=0, **kwargs
        )


class SaveMergedLoRA:
    """
    Saves merged LoRA patches from LoRA Optimizer as a standalone .safetensors
    file that can be loaded by any standard LoRA loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_folders = folder_paths.get_folder_paths("loras")
        folder_choices = lora_folders if lora_folders else [os.path.join(folder_paths.models_dir, "loras")]
        return {
            "required": {
                "lora_data": ("LORA_DATA", {"tooltip": "Connect the lora_data output from LoRA Optimizer here."}),
                "save_folder": (folder_choices, {"tooltip": "Which loras folder to save into. Lists all configured lora paths (from extra_model_paths.yaml and defaults)."}),
                "filename": ("STRING", {"default": "merged_lora", "tooltip": "Name for the saved file. Subdirectories allowed (e.g. 'merged/my_lora'). Extension .safetensors is added automatically."}),
                "save_rank": ("INT", {
                    "default": 0, "min": 0, "max": 2048, "step": 4,
                    "tooltip": "0 = auto (adaptively finds the rank needed for <5%% reconstruction error — recommended). Non-zero = force this rank for any layers that need compression. Higher values = more accurate but larger file."
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
    CATEGORY = "LoRA Optimizer"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves merged LoRA data as a standalone .safetensors file that can be loaded by any standard LoRA loader."

    def save_lora(self, lora_data, save_folder, filename, save_rank=0, bake_strength=True):
        if lora_data is None:
            logging.warning("[Save Merged LoRA] No lora_data received (optimizer may have returned early). Nothing to save.")
            return ("",)

        # Determine save path — subdirectories allowed, but must stay inside chosen folder
        save_dir = save_folder
        base = filename if filename.endswith('.safetensors') else f"{filename}.safetensors"
        save_path = os.path.join(save_dir, base)
        if not os.path.realpath(save_path).startswith(os.path.realpath(save_dir) + os.sep):
            raise ValueError(f"[Save Merged LoRA] Path escapes loras directory: {filename}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model_patches = lora_data["model_patches"]
        clip_patches = lora_data["clip_patches"]
        key_map = lora_data["key_map"]
        output_strength = lora_data["output_strength"]
        clip_strength = lora_data["clip_strength"]

        auto_rank = save_rank == 0

        # Auto mode: determine rank for compressing full-rank diffs.
        # LoRAAdapter patches keep their existing rank (already compressed during merge).
        # Full-rank diff patches (e.g. TIES) need adaptive estimation — element-wise
        # operations like TIES trim/elect destroy low-rank structure, so sum_rank
        # is not a reliable estimate of the effective rank.
        if auto_rank:
            # Check if there are any full-rank diffs that need compression
            has_diffs = any(
                isinstance(patch, tuple) and patch[0] == "diff"
                for patch in list(model_patches.values()) + list(clip_patches.values())
            )
            if has_diffs:
                initial_rank = lora_data.get("sum_rank", 128)
                fallback_rank = LoRAOptimizer._estimate_save_rank(initial_rank, model_patches, clip_patches)
                logging.info(f"[Save Merged LoRA] Auto rank for diffs: {fallback_rank} "
                             f"(initial estimate {initial_rank}, adapted from sample diffs)")
            else:
                # All patches are already low-rank (LoRAAdapter) — fallback_rank
                # is only used for LoKr/LoHa expansion, use sum_rank or 128
                fallback_rank = lora_data.get("sum_rank", 128)
                logging.info(f"[Save Merged LoRA] Auto rank: {fallback_rank} (no full-rank diffs to compress)")

        # Detect native storage dtype (fp16/bf16) from patches for output.
        # SVD compression produces float32 internally, but saved LoRAs should
        # match the original weight precision to avoid doubling file size.
        save_dtype = None
        for patch in list(model_patches.values()) + list(clip_patches.values()):
            if isinstance(patch, tuple) and patch[0] == "diff":
                dt = patch[1][0].dtype
            elif hasattr(patch, 'weights') and patch.weights[0] is not None:
                dt = patch.weights[0].dtype
            else:
                continue
            if dt not in (torch.float32, torch.float64):
                save_dtype = dt
                break
        if save_dtype is None:
            save_dtype = torch.float16
        logging.info(f"[Save Merged LoRA] Output dtype: {save_dtype}")

        state_dict = {}

        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            for target_key, patch in patches.items():
                # Look up by full target_key first (preserves offset info for split
                # Q/K/V patches), then fall back to base key string for non-tuple keys.
                lora_prefix = key_map.get(target_key)
                if lora_prefix is None:
                    tkey = target_key[0] if isinstance(target_key, tuple) else target_key
                    lora_prefix = key_map.get(tkey, tkey)

                if isinstance(patch, (LoKrAdapter, LoHaAdapter)):
                    # Expand LoKr/LoHa to full diff and compress to standard LoRA format
                    diff_tensor = _LoRAMergeBase._expand_patch_to_diff(patch)
                    rank = fallback_rank if auto_rank else save_rank
                    compressed = LoRAOptimizer._compress_to_lowrank(diff_tensor, rank)
                    mat_up, mat_down, alpha, mid, _, _ = compressed.weights
                    alpha = float(alpha)
                elif isinstance(patch, LoRAAdapter):
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

                state_dict[f"{lora_prefix}.lora_up.weight"] = mat_up.to(save_dtype).cpu().contiguous()
                state_dict[f"{lora_prefix}.lora_down.weight"] = mat_down.to(save_dtype).cpu().contiguous()
                state_dict[f"{lora_prefix}.alpha"] = torch.tensor(alpha)

        # Validate saved data before writing
        n_keys = len(state_dict) // 3
        unmapped_keys = []
        nan_keys = []
        zero_keys = []
        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            for target_key, patch in patches.items():
                direct = key_map.get(target_key)
                if direct is None:
                    tkey = target_key[0] if isinstance(target_key, tuple) else target_key
                    fallback = key_map.get(tkey)
                    label = f"{'CLIP' if is_clip else 'MODEL'} {tkey}"
                    if fallback is None:
                        unmapped_keys.append(f"{label} (using raw key as prefix)")
                    else:
                        unmapped_keys.append(f"{label} (fallback to base key)")

        for skey, tensor in state_dict.items():
            if skey.endswith('.alpha'):
                continue
            if torch.isnan(tensor).any():
                nan_keys.append(skey)
            if tensor.abs().max().item() == 0:
                zero_keys.append(skey)

        if unmapped_keys:
            logging.warning(f"[Save Merged LoRA] {len(unmapped_keys)} keys fell through to fallback mapping:")
            for k in unmapped_keys[:5]:
                logging.warning(f"  {k}")
        if nan_keys:
            logging.error(f"[Save Merged LoRA] {len(nan_keys)} tensors contain NaN!")
            for k in nan_keys[:5]:
                logging.error(f"  {k}")
        if zero_keys:
            logging.warning(f"[Save Merged LoRA] {len(zero_keys)} tensors are all zeros")

        # Log first few saved key prefixes for debugging
        prefixes = sorted(set(k.rsplit('.lora_', 1)[0] for k in state_dict if '.lora_' in k))
        if prefixes:
            logging.info(f"[Save Merged LoRA] Sample prefixes: {prefixes[:3]} ... ({len(prefixes)} total)")

        # Roundtrip SVD quality check: compare reconstructed diffs against originals
        svd_errors = []
        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            for target_key, patch in patches.items():
                lora_prefix = key_map.get(target_key)
                if lora_prefix is None:
                    continue
                up_key = f"{lora_prefix}.lora_up.weight"
                down_key = f"{lora_prefix}.lora_down.weight"
                alpha_key = f"{lora_prefix}.alpha"
                if up_key not in state_dict:
                    continue
                # Reconstruct diff from saved up/down
                saved_up = state_dict[up_key].float()
                saved_down = state_dict[down_key].float()
                saved_alpha = state_dict[alpha_key].item()
                rank = saved_down.shape[0]
                scale = saved_alpha / rank
                reconstructed = torch.mm(saved_up, saved_down) * scale
                # Get original diff for comparison
                if isinstance(patch, tuple) and patch[0] == "diff":
                    orig_diff = patch[1][0].float()
                    strength = clip_strength if is_clip else output_strength
                    if bake_strength:
                        orig_with_strength = orig_diff * strength
                    else:
                        orig_with_strength = orig_diff
                    orig_norm = orig_with_strength.norm().item()
                    if orig_norm > 0:
                        error = (reconstructed - orig_with_strength).norm().item() / orig_norm
                        svd_errors.append(error)
        if svd_errors:
            avg_err = sum(svd_errors) / len(svd_errors)
            max_err = max(svd_errors)
            logging.info(f"[Save Merged LoRA] SVD reconstruction error: "
                         f"avg={avg_err:.4f}, max={max_err:.4f} "
                         f"({len(svd_errors)} diffs checked)")

        save_file(state_dict, save_path)
        logging.info(f"[Save Merged LoRA] Saved {n_keys} LoRA keys to {save_path}")

        return (save_path,)


class SaveTunerData:
    """Saves AutoTuner results (TUNER_DATA) to a .tuner file for later reuse."""

    @classmethod
    def INPUT_TYPES(cls):
        tuner_folders = folder_paths.get_folder_paths("tuner_data")
        folder_choices = tuner_folders if tuner_folders else [TUNER_DATA_DIR]
        return {
            "required": {
                "tuner_data": ("TUNER_DATA", {"tooltip": "Connect the tuner_data output from LoRA AutoTuner."}),
                "save_folder": (folder_choices, {"tooltip": "Which tuner_data folder to save into. Lists all configured paths (from extra_model_paths.yaml and defaults)."}),
                "filename": ("STRING", {"default": "tuner_data", "tooltip": "Filename. Subdirectories allowed (e.g. 'results/experiment1'). Extension .tuner is added automatically (.json also accepted)."}),
                "overwrite": ("BOOLEAN", {"default": True, "tooltip": "When enabled, overwrites existing files. When disabled, appends _001, _002, etc. to avoid overwriting."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_tuner_data"
    CATEGORY = "LoRA Optimizer"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves AutoTuner results to a .tuner file so they can be reloaded later without re-running the tuner."

    def save_tuner_data(self, tuner_data, save_folder, filename, overwrite=True):
        if tuner_data is None:
            return ("",)
        save_dir = save_folder
        base = filename if filename.endswith((".json", ".tuner")) else f"{filename}.tuner"
        save_path = os.path.join(save_dir, base)
        if not os.path.realpath(save_path).startswith(os.path.realpath(save_dir) + os.sep):
            raise ValueError(f"[Save Tuner Data] Path escapes tuner_data directory: {filename}")
        if not overwrite and os.path.exists(save_path):
            stem, ext = os.path.splitext(save_path)
            counter = 1
            while os.path.exists(save_path):
                save_path = f"{stem}_{counter:03d}{ext}"
                counter += 1
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(tuner_data, f, indent=2)
        logging.info(f"[Save Tuner Data] Saved to: {save_path}")
        return (save_path,)


class LoadTunerData:
    """Loads previously saved AutoTuner results (TUNER_DATA) from a JSON file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tuner_data_file": (folder_paths.get_filename_list("tuner_data"), {"tooltip": "Select a saved tuner data file."}),
            }
        }

    RETURN_TYPES = ("TUNER_DATA",)
    RETURN_NAMES = ("tuner_data",)
    FUNCTION = "load_tuner_data"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Loads saved AutoTuner results from disk so they can be fed to Merge Selector without re-running the tuner."

    @classmethod
    def IS_CHANGED(cls, tuner_data_file):
        load_path = folder_paths.get_full_path_or_raise("tuner_data", tuner_data_file)
        return os.path.getmtime(load_path)

    def load_tuner_data(self, tuner_data_file):
        load_path = folder_paths.get_full_path_or_raise("tuner_data", tuner_data_file)
        with open(load_path, "r") as f:
            tuner_data = json.load(f)
        logging.info(f"[Load Tuner Data] Loaded from: {load_path} "
                     f"({len(tuner_data.get('top_n', []))} configs)")
        return (tuner_data,)


class MergedLoRAToHook:
    """Converts merged LoRA data into a conditioning hook for per-conditioning application."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_data": ("LORA_DATA", {
                    "tooltip": "Connect the lora_data output from LoRA Optimizer."
                }),
            },
            "optional": {
                "prev_hooks": ("HOOKS", {
                    "tooltip": "Optional: chain with existing hooks."
                }),
            },
        }

    RETURN_TYPES = ("HOOKS",)
    RETURN_NAMES = ("hooks",)
    FUNCTION = "convert"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Wraps merged LoRA patches as a conditioning hook (HOOKS). "
        "Connect to 'Cond Set Props' or similar nodes to apply the merged LoRA "
        "per-conditioning instead of globally."
    )

    def convert(self, lora_data, prev_hooks=None):
        import comfy.hooks

        if prev_hooks is None:
            prev_hooks = comfy.hooks.HookGroup()
        else:
            prev_hooks = prev_hooks.clone()

        if lora_data is None:
            logging.warning("[MergedLoRAToHook] No lora_data received. Returning empty hooks.")
            return (prev_hooks,)

        model_patches = lora_data.get("model_patches", {})
        clip_patches = lora_data.get("clip_patches", {})

        if not model_patches and not clip_patches:
            logging.warning("[MergedLoRAToHook] lora_data has no patches. Returning empty hooks.")
            return (prev_hooks,)

        strength_model = lora_data.get("output_strength", 1.0)
        strength_clip = lora_data.get("clip_strength", 1.0)
        if strength_clip is None:
            strength_clip = 1.0

        if strength_model == 0 and strength_clip == 0:
            return (prev_hooks,)

        hook = comfy.hooks.WeightHook(
            strength_model=strength_model,
            strength_clip=strength_clip,
        )
        hook.weights = model_patches
        hook.weights_clip = clip_patches
        hook.need_weight_init = False

        hook_group = comfy.hooks.HookGroup()
        hook_group.add(hook)
        return (prev_hooks.clone_and_combine(hook_group),)


class MergedLoRAToWanVideo:
    """Applies merged LoRA patches to a WanVideo wrapper model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wan_model": ("WANVIDEOMODEL",),
            },
            "optional": {
                "lora_data": ("LORA_DATA",),
            },
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patches"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Bridges merged LoRA patches from the LoRA Optimizer to a WanVideo wrapper model. "
        "Use this when the WanVideo wrapper exposes fewer keys than the core MODEL, "
        "ensuring all merged patches reach the sampler via set_lora_params."
    )

    def apply_patches(self, wan_model, lora_data=None):
        if lora_data is None:
            return (wan_model,)

        model_patches = lora_data.get("model_patches", {})
        if not model_patches:
            return (wan_model,)

        output_strength = lora_data.get("output_strength", 1.0)
        new_model = wan_model.clone()

        # Inject merged patches in the format set_lora_params expects:
        # patcher.patches[key] = [(strength, patch_obj, 1.0, None, None)]
        #
        # set_lora_params (custom_linear.py:101-106) looks up keys as
        # "diffusion_model.{module_prefix}weight" and falls back to
        # stripping "_orig_mod." — so our core-model keys will match.
        applied = 0
        for key, patch in model_patches.items():
            patch_key = key if isinstance(key, str) else key[0]
            current = new_model.patches.get(patch_key, [])
            current.append((output_strength, patch, 1.0, None, None))
            new_model.patches[patch_key] = current
            applied += 1

        import uuid
        new_model.patches_uuid = uuid.uuid4()

        logging.info(f"[MergedLoRAToWanVideo] Applied {applied} merged patches "
                     f"(output_strength={output_strength})")
        return (new_model,)


class LoRACompatibilityAnalyzer(LoRAOptimizer):
    """
    Standalone pre-merge planning tool. Analyzes pairwise LoRA interactions
    (cosine similarity, sign conflicts, magnitude ratios) and recommends
    which LoRAs to merge together vs. use independently.

    No merge happens — analysis only. Inherits from LoRAOptimizer to reuse
    _analyze_prefix, _compute_auto_strengths, and _auto_select_params.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False,
                    "tooltip": "Toggle on to run the analysis. Off by default so it doesn't slow down normal queue execution."}),
                "model": ("MODEL", {"tooltip": "Base model for key mapping and target shapes."}),
                "lora_stack": ("LORA_STACK", {"tooltip": "The LoRA stack to analyze for compatibility."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Text encoder. Connect for CLIP key analysis."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("report", "compatibility_map")
    FUNCTION = "analyze"
    CATEGORY = "LoRA Optimizer"
    OUTPUT_NODE = True
    DESCRIPTION = "Analyzes pairwise LoRA compatibility and recommends merge groups. No merge is performed — use this to plan which LoRAs to combine before merging."

    @classmethod
    def IS_CHANGED(cls, enabled, **kwargs):
        if not enabled:
            return "disabled"  # stable value → cached, effectively skipped
        return float("NaN")  # always re-execute when enabled

    def analyze(self, enabled, model, lora_stack, clip=None):
        if not enabled:
            return {"ui": {"groups": []}, "result": ("Analysis disabled. Toggle 'enabled' to run.", self._empty_image())}

        if not lora_stack or len(lora_stack) == 0:
            return {"ui": {"groups": []}, "result": ("No LoRAs in stack.", self._empty_image())}

        normalized_stack = self._normalize_stack(lora_stack)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]

        if len(active_loras) == 0:
            return {"ui": {"groups": []}, "result": ("No active LoRAs in stack (all zero strength).", self._empty_image())}

        n_loras = len(active_loras)
        detected_arch = getattr(self, '_detected_arch', None) or 'unknown'
        preset_key, arch_preset = _resolve_arch_preset("auto", detected_arch)

        if n_loras == 1:
            name = active_loras[0]["name"]
            report = (
                "=" * 55 + "\n"
                "  LoRA Compatibility Analysis\n"
                "=" * 55 + "\n\n"
                f"Single LoRA: {name}\n"
                "Nothing to compare — add more LoRAs to analyze compatibility.\n"
                "\n" + "=" * 55
            )
            return {"ui": {"groups": []}, "result": (report, self._empty_image())}

        # --- Setup ---
        logging.info(f"[Compatibility Analyzer] Analyzing {n_loras} LoRAs...")
        t_start = time.time()

        model_keys = self._get_model_keys(model)
        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})

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
        all_lora_prefixes = sorted(all_lora_prefixes)
        compute_device = self._get_compute_device()

        # --- Pass 1 Analysis ---
        analysis = self._run_pass1_analysis(active_loras, all_lora_prefixes,
                                            model_keys, clip_keys, model, clip, compute_device)
        prefix_count = analysis["prefix_count"]

        if prefix_count == 0:
            return {"ui": {"groups": []}, "result": (
                "No compatible LoRA keys found. "
                "LoRAs may be incompatible with this model architecture.",
                self._empty_image())}

        # --- Finalize stats ---
        per_lora_stats = analysis["per_lora_stats"]
        pair_accum = analysis["pair_accum"]
        pair_shared_prefixes = analysis["pair_shared_prefixes"]
        pairs = analysis["pairs"]

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
            })

        # Full-rank detection for compatibility analyzer
        ca_fr_preset = arch_preset.get("full_rank", {})
        ca_global_avg_rank = (sum(s["avg_rank"] for s in lora_stats) / len(lora_stats)) if lora_stats else 0
        ca_is_full_rank = ca_global_avg_rank >= ca_fr_preset.get("rank_threshold", 256)
        if ca_is_full_rank:
            logging.info(f"[Compatibility Analyzer] Full-rank LoRAs detected (avg rank {ca_global_avg_rank:.0f})")

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
                pair_label = (f"{name_i} [#{i+1}, str={active_loras[i]['strength']}] vs "
                              f"{name_j} [#{j+1}, str={active_loras[j]['strength']}]")
            else:
                pair_label = f"{name_i} vs {name_j}"
            pairwise_conflicts.append({
                "i": i, "j": j,
                "pair": pair_label,
                "overlap": pair_overlap,
                "conflicts": pair_conflict,
                "ratio": ratio,
                "cosine_sim": cos_sim,
                "shared_prefixes": pair_shared_prefixes[(i, j)],
            })

        avg_conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0
        valid_l2 = [m for m in l2_means if m > 0]
        magnitude_ratio = max(valid_l2) / min(valid_l2) if len(valid_l2) >= 2 else 1.0
        avg_cos_sim = (sum(pairwise_similarities.values()) / len(pairwise_similarities)
                       if pairwise_similarities else 0.0)

        collection_stats = {
            "n_loras": n_loras,
            "total_keys": prefix_count,
            "avg_conflict": avg_conflict_ratio,
            "avg_cos_sim": avg_cos_sim,
            "magnitude_ratio": magnitude_ratio,
        }

        # --- Compatibility matrix ---
        compat_matrix = self._compute_compat_matrix(pairwise_similarities, pairwise_conflicts, n_loras)

        # --- Clustering ---
        groups = self._cluster_loras(compat_matrix, threshold=0.05)

        # --- Per-group recommendations ---
        group_info = []
        for group in groups:
            info = {"indices": group}
            if len(group) == 1:
                idx = group[0]
                info["type"] = "solo"
                info["strength"] = active_loras[idx]["strength"]
                # Find opposing LoRAs
                opposing = []
                for other_idx in range(n_loras):
                    if other_idx == idx:
                        continue
                    key = (min(idx, other_idx), max(idx, other_idx))
                    cs = pairwise_similarities.get(key, 0.0)
                    if cs < -0.05:
                        opposing.append((active_loras[other_idx]["name"], cs))
                opposing.sort(key=lambda x: x[1])
                info["opposing"] = opposing
            else:
                info["type"] = "merge"
                # Compute intra-group stats
                conflict_by_pair = {(pc["i"], pc["j"]): pc["ratio"] for pc in pairwise_conflicts}
                intra_cos = []
                intra_conflict = []
                for a_pos in range(len(group)):
                    for b_pos in range(a_pos + 1, len(group)):
                        key = (min(group[a_pos], group[b_pos]), max(group[a_pos], group[b_pos]))
                        cs = pairwise_similarities.get(key, 0.0)
                        cr = conflict_by_pair.get(key, 0.0)
                        intra_cos.append(cs)
                        intra_conflict.append(cr)
                info["avg_cos_sim"] = sum(intra_cos) / len(intra_cos) if intra_cos else 0.0
                info["avg_conflict"] = sum(intra_conflict) / len(intra_conflict) if intra_conflict else 0.0
                avg_compat = sum(cs * (1.0 - cr) for cs, cr in zip(intra_cos, intra_conflict)) / len(intra_cos) if intra_cos else 0.0
                if avg_compat > 0.2:
                    info["confidence"] = "High"
                elif avg_compat > 0.05:
                    info["confidence"] = "Moderate"
                else:
                    info["confidence"] = "Low"

                # Auto-strengths for group
                group_loras = [active_loras[idx] for idx in group]
                group_lora_stats = [lora_stats[idx] for idx in group]
                group_pw_sim = {}
                for a_pos in range(len(group)):
                    for b_pos in range(a_pos + 1, len(group)):
                        orig_key = (min(group[a_pos], group[b_pos]), max(group[a_pos], group[b_pos]))
                        group_pw_sim[(a_pos, b_pos)] = pairwise_similarities.get(orig_key, 0.0)
                new_strengths, _ = self._compute_auto_strengths(
                    group_loras, group_lora_stats,
                    pairwise_similarities=group_pw_sim,
                    arch_preset=arch_preset, detected_arch=detected_arch,
                    is_full_rank=ca_is_full_rank)
                info["strengths"] = {group[k]: new_strengths[k] for k in range(len(group))}

                # Suggested merge strategy
                group_avg_cos = info["avg_cos_sim"]
                group_avg_conflict = info["avg_conflict"]
                mag_samples = analysis["all_magnitude_samples"]
                mode, _, _, _ = self._auto_select_params(
                    group_avg_conflict,
                    magnitude_ratio,
                    magnitude_samples=mag_samples,
                    avg_cos_sim=group_avg_cos,
                    arch_preset=arch_preset)
                info["suggested_merge"] = mode

            group_info.append(info)

        # --- Warnings ---
        def _disambiguate(idx_a, idx_b):
            """Return display names, adding [#N] suffix when names collide."""
            na = active_loras[idx_a]["name"]
            nb = active_loras[idx_b]["name"]
            if na == nb:
                na = f"{na} [#{idx_a+1}]"
                nb = f"{nb} [#{idx_b+1}]"
            return na, nb

        warnings = []
        for pc in pairwise_conflicts:
            if pc["cosine_sim"] < -0.1:
                na, nb = _disambiguate(pc["i"], pc["j"])
                warnings.append({
                    "type": "opposing",
                    "name_i": na,
                    "name_j": nb,
                    "cos_sim": pc["cosine_sim"],
                })
        # Magnitude imbalance warnings — consolidated per strong LoRA
        # Compute per-LoRA average magnitude ratio vs all others
        lora_avg_ratios = {}  # idx -> avg ratio relative to mean
        if len(l2_means) >= 2:
            valid_l2 = [m for m in l2_means if m > 0]
            mean_l2 = sum(valid_l2) / len(valid_l2) if valid_l2 else 1.0
            for idx, l2 in enumerate(l2_means):
                if l2 > 0 and mean_l2 > 0:
                    lora_avg_ratios[idx] = l2 / mean_l2

        strong_loras = {}  # strong_idx -> [(weak_name, ratio)]
        for pc in pairwise_conflicts:
            i, j = pc["i"], pc["j"]
            l2_i = l2_means[i]
            l2_j = l2_means[j]
            if l2_i > 0 and l2_j > 0:
                ratio = max(l2_i, l2_j) / min(l2_i, l2_j)
                if ratio > 3.0:
                    strong_idx = i if l2_i > l2_j else j
                    weak_idx = j if l2_i > l2_j else i
                    weak_name = active_loras[weak_idx]["name"]
                    if strong_idx not in strong_loras:
                        strong_loras[strong_idx] = []
                    if weak_name not in [w for w, _ in strong_loras[strong_idx]]:
                        strong_loras[strong_idx].append((weak_name, ratio))
        for strong_idx in sorted(strong_loras.keys()):
            weak_list = strong_loras[strong_idx]
            strong_name = active_loras[strong_idx]["name"]
            avg_ratio = lora_avg_ratios.get(strong_idx, 0.0)
            warnings.append({
                "type": "magnitude_group",
                "stronger": strong_name,
                "avg_ratio": avg_ratio,
                "weaker_list": [w for w, _ in weak_list],
            })

        elapsed = time.time() - t_start
        logging.info(f"[Compatibility Analyzer] Analysis complete ({elapsed:.1f}s)")

        # --- Build report ---
        report = self._build_compatibility_report(
            active_loras, groups, group_info, lora_stats, pairwise_conflicts,
            collection_stats, warnings, detected_arch, prefix_count)

        # --- Generate heatmap (use raw compat without baseline for visual differentiation) ---
        raw_matrix = self._compute_raw_compat_matrix(pairwise_similarities, pairwise_conflicts, n_loras)
        display_names = [os.path.splitext(os.path.basename(item["name"]))[0] for item in active_loras]
        heatmap = self._generate_heatmap(raw_matrix, display_names, groups)

        # --- Build group data for UI (auto-create LoRA Stack nodes) ---
        groups_for_ui = []
        for gi in group_info:
            if gi["type"] != "merge":
                continue
            groups_for_ui.append({
                "loras": [
                    {"name": active_loras[idx]["name"], "strength": round(gi["strengths"][idx], 4)}
                    for idx in gi["indices"]
                ],
                "suggested_merge": gi.get("suggested_merge", "weighted_average"),
                "confidence": gi.get("confidence", "Low"),
            })

        return {"ui": {"groups": groups_for_ui}, "result": (report, heatmap)}

    @staticmethod
    def _empty_image():
        """Return a minimal 1x1 black IMAGE tensor."""
        return torch.zeros(1, 1, 1, 3, dtype=torch.float32)

    @staticmethod
    def _compute_compat_matrix(pairwise_similarities, pairwise_conflicts, n_loras):
        """Compute N×N compatibility matrix: compat[i][j] = cos_sim * (1 - conflict_ratio) + baseline.

        Non-opposing LoRAs (cos_sim > -0.1) get a +0.1 baseline because orthogonal
        LoRAs are safe to merge — they don't interfere with each other.
        """
        conflict_by_pair = {}
        for pc in pairwise_conflicts:
            conflict_by_pair[(pc["i"], pc["j"])] = pc["ratio"]

        matrix = [[0.0] * n_loras for _ in range(n_loras)]
        for (i, j), cos_sim in pairwise_similarities.items():
            conflict_ratio = conflict_by_pair.get((i, j), 0.0)
            compat = cos_sim * (1.0 - conflict_ratio)
            if cos_sim > -0.1:
                compat += 0.1  # non-opposing LoRAs are safe to merge
            matrix[i][j] = compat
            matrix[j][i] = compat
        return matrix

    @staticmethod
    def _compute_raw_compat_matrix(pairwise_similarities, pairwise_conflicts, n_loras):
        """Compute N×N raw compatibility matrix without baseline (for heatmap display)."""
        conflict_by_pair = {}
        for pc in pairwise_conflicts:
            conflict_by_pair[(pc["i"], pc["j"])] = pc["ratio"]

        matrix = [[0.0] * n_loras for _ in range(n_loras)]
        for (i, j), cos_sim in pairwise_similarities.items():
            conflict_ratio = conflict_by_pair.get((i, j), 0.0)
            compat = cos_sim * (1.0 - conflict_ratio)
            matrix[i][j] = compat
            matrix[j][i] = compat
        return matrix

    @staticmethod
    def _cluster_loras(compat_matrix, threshold=0.05):
        """Greedy agglomerative clustering. Returns list of groups (lists of indices)."""
        n = len(compat_matrix)
        groups = [[i] for i in range(n)]

        while len(groups) > 1:
            best_score = -float('inf')
            best_pair = None
            for ga_idx in range(len(groups)):
                for gb_idx in range(ga_idx + 1, len(groups)):
                    ga, gb = groups[ga_idx], groups[gb_idx]
                    total = 0.0
                    count = 0
                    for a in ga:
                        for b in gb:
                            total += compat_matrix[a][b]
                            count += 1
                    avg_compat = total / count if count > 0 else 0.0
                    if avg_compat > best_score:
                        best_score = avg_compat
                        best_pair = (ga_idx, gb_idx)

            if best_score < threshold:
                break

            # Merge the best pair
            ga_idx, gb_idx = best_pair
            merged = groups[ga_idx] + groups[gb_idx]
            groups = [g for idx, g in enumerate(groups) if idx != ga_idx and idx != gb_idx]
            groups.append(merged)

        # Sort groups: multi-member first (by size desc), then singletons
        groups.sort(key=lambda g: (-len(g), g[0]))
        return groups

    @staticmethod
    def _generate_heatmap(compat_matrix, lora_names, groups):
        """Generate N×N heatmap as torch IMAGE tensor using PIL."""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        n = len(compat_matrix)
        if n == 0:
            return torch.zeros(1, 1, 1, 3, dtype=torch.float32)

        # Reorder indices by group membership so group borders are contiguous
        display_order = []
        for g in groups:
            display_order.extend(g)

        cell_size = 80

        # Compute margin from longest label
        tmp_img = Image.new("RGB", (1, 1))
        tmp_draw = ImageDraw.Draw(tmp_img)
        max_label_w = max(tmp_draw.textlength(name) for name in lora_names) if lora_names else 100
        margin = int(max_label_w) + 20  # padding
        col_header_h = margin  # same space for rotated column headers
        img_w = margin + n * cell_size
        img_h = col_header_h + n * cell_size

        img = Image.new("RGB", (img_w, img_h), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        # Color mapping: continuous red (-0.4) → yellow (0.05) → green (+0.5)
        def compat_to_color(val):
            # Normalize to 0..1 range: -0.4 → 0, +0.5 → 1
            t = max(0.0, min(1.0, (val + 0.4) / 0.9))
            if t < 0.5:
                # Red to yellow (t: 0→0.5)
                s = t / 0.5  # 0..1
                r = int(200 + 20 * s)
                g = int(180 * s)
                return (r, g, 0)
            else:
                # Yellow to green (t: 0.5→1)
                s = (t - 0.5) / 0.5  # 0..1
                r = int(220 * (1 - s))
                g = int(180 + 20 * s)
                return (r, g, 0)

        # Draw cells (using reordered indices)
        for row in range(n):
            for col in range(n):
                orig_i = display_order[row]
                orig_j = display_order[col]
                x = margin + col * cell_size
                y = col_header_h + row * cell_size
                if orig_i == orig_j:
                    color = (80, 80, 80)
                    val_text = "-"
                else:
                    val = compat_matrix[orig_i][orig_j]
                    color = compat_to_color(val)
                    val_text = f"{val:.2f}"

                draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=color)
                # Center text in cell
                bbox = draw.textbbox((0, 0), val_text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x + (cell_size - tw) // 2
                ty = y + (cell_size - th) // 2
                # Text color: white on dark, black on light
                brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                draw.text((tx, ty), val_text, fill=text_color)

        # Draw row labels (full name, right-aligned against grid)
        for pos in range(n):
            label = lora_names[display_order[pos]]
            y = col_header_h + pos * cell_size + cell_size // 2 - 5
            draw.text((5, y), label, fill=(220, 220, 220))

        # Draw column labels (rotated 90° for readability)
        for pos in range(n):
            label = lora_names[display_order[pos]]
            # Render text onto a temporary image, rotate, paste
            bbox = tmp_draw.textbbox((0, 0), label)
            tw = bbox[2] - bbox[0] + 4
            th = bbox[3] - bbox[1] + 4
            txt_img = Image.new("RGBA", (tw, th), (30, 30, 30, 255))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((2, 2), label, fill=(220, 220, 220, 255))
            rotated = txt_img.rotate(90, expand=True)
            paste_x = margin + pos * cell_size + (cell_size - rotated.width) // 2
            paste_y = col_header_h - rotated.height - 5
            img.paste(rotated.convert("RGB"), (paste_x, max(0, paste_y)))

        # Draw group borders (contiguous after reordering)
        pos = 0
        for group in groups:
            if len(group) > 1:
                x0 = margin + pos * cell_size - 1
                y0 = col_header_h + pos * cell_size - 1
                x1 = margin + (pos + len(group)) * cell_size
                y1 = col_header_h + (pos + len(group)) * cell_size
                for offset in range(3):
                    draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                                   outline=(255, 255, 255))
            pos += len(group)

        # Convert PIL → torch tensor [1, H, W, 3] float32
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]
        return tensor

    def _build_compatibility_report(self, active_loras, groups, group_info,
                                     lora_stats, pairwise_conflicts,
                                     collection_stats, warnings,
                                     detected_arch, prefix_count):
        """Build human-readable compatibility report."""
        n_loras = len(active_loras)
        lines = []

        def _dn(name):
            """Display name: strip .safetensors extension."""
            return name.replace(".safetensors", "")

        lines.append("=" * 55)
        lines.append("  LoRA Compatibility Analysis")
        lines.append("=" * 55)
        lines.append("")
        lines.append(f"Analyzed: {n_loras} LoRAs across {prefix_count} prefixes")
        arch_display = detected_arch.upper() if detected_arch and detected_arch != 'unknown' else "Unknown"
        lines.append(f"Architecture: {arch_display}")
        lines.append(f"Avg conflict: {collection_stats['avg_conflict']:.0%} | "
                     f"Avg cosine similarity: {collection_stats['avg_cos_sim']:.2f}")
        lines.append("")

        # --- Recommended Groups ---
        lines.append("-" * 30 + " Recommended Groups " + "-" * 30)
        lines.append("")

        group_num = 0
        solo_entries = []
        for gi in group_info:
            indices = gi["indices"]
            if gi["type"] == "solo":
                solo_entries.append(gi)
                continue

            group_num += 1
            confidence = gi["confidence"]
            label = "Safe to combine" if confidence == "Low" else "Merge together"
            lines.append(f"Group {group_num} -- {label} (compatibility: {confidence})")
            for idx in indices:
                strength = gi["strengths"].get(idx, active_loras[idx]["strength"])
                name = _dn(active_loras[idx]["name"])
                lines.append(f"  * {name:<28s} strength {strength:.2f}")
            lines.append(f"  Avg cosine sim: {gi['avg_cos_sim']:.2f} | "
                        f"Avg conflict: {gi['avg_conflict']:.0%}")
            lines.append(f"  Suggested merge: {gi['suggested_merge']}")
            lines.append("")

        if solo_entries:
            lines.append("Solo -- Use independently")
            for gi in solo_entries:
                idx = gi["indices"][0]
                name = _dn(active_loras[idx]["name"])
                strength = gi["strength"]
                lines.append(f"  * {name:<28s} strength {strength:.2f}")
                if gi["opposing"]:
                    opp_strs = [f"{_dn(oname)} (cos: {cs:.2f})" for oname, cs in gi["opposing"][:3]]
                    lines.append(f"    Opposes: {', '.join(opp_strs)}")
            lines.append("")

        # --- Warnings ---
        if warnings:
            lines.append("-" * 30 + " Warnings " + "-" * 30)
            lines.append("")
            for w in warnings:
                if w["type"] == "opposing":
                    lines.append(f"! {_dn(w['name_i'])} vs {_dn(w['name_j'])}: Opposing (cos_sim: {w['cos_sim']:.2f})")
                    lines.append(f"  These cancel each other out -- using both will degrade quality.")
                    lines.append("")
                elif w["type"] == "magnitude_group":
                    lines.append(f"! {_dn(w['stronger'])} ({w['avg_ratio']:.1f}x avg) overshadows:")
                    lines.append(f"  {', '.join(_dn(n) for n in w['weaker_list'])}")
                    lines.append(f"  Consider lowering its strength or raising weaker LoRAs.")
                    lines.append("")
                elif w["type"] == "magnitude":
                    lines.append(f"! {_dn(w['stronger'])} is {w['ratio']:.1f}x stronger than {_dn(w['weaker'])}")
                    lines.append(f"  {_dn(w['weaker'])} may be overshadowed at equal strengths.")
                    lines.append("")

        # --- Pairwise Compatibility Table ---
        lines.append("-" * 30 + " Pairwise Compatibility " + "-" * 30)
        lines.append("")

        # Sort by compat score descending
        sorted_pairs = sorted(pairwise_conflicts, key=lambda p: p["cosine_sim"] * (1.0 - p["ratio"]), reverse=True)
        # Compute max pair label width for right-aligned columns
        pair_labels = [_dn(pc["pair"]) for pc in sorted_pairs]
        max_pair_w = max((len(p) for p in pair_labels), default=45)
        for pc, pair_label in zip(sorted_pairs, pair_labels):
            conflict = pc["ratio"]
            cos = round(pc["cosine_sim"], 2) + 0.0  # avoid -0.00
            compat = round(pc["cosine_sim"] * (1.0 - conflict), 2) + 0.0  # avoid -0.00
            indicator = ""
            if compat > 0.2:
                indicator = " [OK]"
            elif compat < -0.05:
                indicator = " [!!]"
            if pc.get("shared_prefixes", 1) == 0:
                indicator += " (no shared keys)"
            lines.append(f"  {pair_label:<{max_pair_w}s}  "
                        f"cos:{cos:6.2f}  conflict:{conflict:5.0%}  compat:{compat:6.2f}{indicator}")
        lines.append("")
        lines.append("=" * 55)

        return "\n".join(lines)


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
                "merge_strategy": (["auto", "ties", "consensus", "slerp", "weighted_average", "weighted_sum"], {
                    "default": "auto",
                    "tooltip": "Merge strategy to pass to the optimizer. "
                               "'auto': let the optimizer decide based on conflict analysis. "
                               "'ties': force TIES merging (good for high-conflict stacks). "
                               "'consensus': force Fisher-proxy + magnitude calibration + spectral cleanup (good for highly similar LoRAs). "
                               "'slerp': force spherical interpolation (magnitude-preserving, good for low-conflict stacks). "
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
    CATEGORY = "LoRA Optimizer"
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
                    cs = float(entry[2]) if entry[2] is not None else -1.0
                    entries.append((str(entry[0]), float(entry[1]), cs, cm))
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
            # Find the original stack position (may not be slot 1 if earlier LoRAs have strength=0)
            orig_pos = next(i for i, item in enumerate(normalized) if item["strength"] != 0)
            cm = kwargs.get(f"conflict_mode_{orig_pos + 1}", "auto")
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

        # Read per-LoRA conflict_mode widgets, mapping by original stack position
        # (not active-only index) so widgets align with what the user sees
        widget_modes = {}
        active_idx = 0
        for orig_idx, item in enumerate(normalized):
            if item["strength"] != 0:
                widget_key = f"conflict_mode_{orig_idx + 1}"
                widget_modes[active_idx] = kwargs.get(widget_key, "auto")
                active_idx += 1

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

        # Release GPU memory after analysis
        compute_device = self._get_compute_device()
        if compute_device.type != "cpu":
            torch.cuda.empty_cache()

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

        # Free loaded LoRA state dicts — the editor only needs them during analysis,
        # and keeping them would leak hundreds of MB per run for large models.
        self.loaded_loras.clear()

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
    "LoRAOptimizerSimple": LoRAOptimizerSimple,
    "SaveMergedLoRA": SaveMergedLoRA,
    "LoRAConflictEditor": LoRAConflictEditor,
    "MergedLoRAToHook": MergedLoRAToHook,
    "MergedLoRAToWanVideo": MergedLoRAToWanVideo,
    "WanVideoLoRAOptimizer": WanVideoLoRAOptimizer,
    "LoRAAutoTuner": LoRAAutoTuner,
    "LoRAMergeSelector": LoRAMergeSelector,
    "SaveTunerData": SaveTunerData,
    "LoadTunerData": LoadTunerData,
    "LoRACompatibilityAnalyzer": LoRACompatibilityAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAStack": "LoRA Stack",
    "LoRAStackDynamic": "LoRA Stack (Dynamic)",
    "LoRAOptimizer": "LoRA Optimizer (Advanced)",
    "LoRAOptimizerSimple": "LoRA Optimizer",
    "SaveMergedLoRA": "Save Merged LoRA",
    "LoRAConflictEditor": "LoRA Conflict Editor",
    "MergedLoRAToHook": "Merged LoRA to Hook",
    "MergedLoRAToWanVideo": "(WIP) Merged LoRA → WanVideo",
    "WanVideoLoRAOptimizer": "(WIP) WanVideo LoRA Optimizer",
    "LoRAAutoTuner": "LoRA AutoTuner",
    "LoRAMergeSelector": "Merge Selector",
    "SaveTunerData": "Save Tuner Data",
    "LoadTunerData": "Load Tuner Data",
    "LoRACompatibilityAnalyzer": "LoRA Compatibility Analyzer",
}
