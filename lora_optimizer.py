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
import glob
import importlib
import importlib.util
import concurrent.futures
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
from comfy.weight_adapter.lora import LoRAAdapter
try:
    from comfy.weight_adapter.lokr import LoKrAdapter
except Exception:
    class LoKrAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights

try:
    from comfy.weight_adapter.loha import LoHaAdapter
except Exception:
    class LoHaAdapter:
        def __init__(self, loaded_keys, weights):
            self.loaded_keys = loaded_keys
            self.weights = weights
from safetensors import safe_open
from safetensors.torch import save_file

# --- Triton SVD kernel (optional) ---
_kernel_path = None
try:
    _kernel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")
    _kernel_spec = importlib.util.spec_from_file_location("lora_optimizer_kernel", _kernel_path)
    _kernel_mod = importlib.util.module_from_spec(_kernel_spec)
    _kernel_spec.loader.exec_module(_kernel_mod)
    _batched_svd = _kernel_mod.batched_svd
    _batched_procrustes = _kernel_mod.batched_procrustes
    _HAS_SVD_KERNEL = True
    _HAS_TRITON = _kernel_mod.HAS_TRITON
    logging.info(f"[LoRA Optimizer] SVD kernel loaded (Triton={_HAS_TRITON})")
except Exception as e:
    _batched_svd = None
    _batched_procrustes = None
    _HAS_SVD_KERNEL = False
    _HAS_TRITON = False
    if _kernel_path and os.path.exists(_kernel_path):
        logging.warning(f"[LoRA Optimizer] kernel.py found but failed to load: {e}")


def _triton_svdvals(mat2d: torch.Tensor, n_sv: int) -> torch.Tensor:
    """Single 2D matrix → singular values, using kernel when available.
    Handles transpose internally — callers can pass any 2D tensor."""
    if mat2d.dim() != 2:
        return torch.linalg.svdvals(mat2d)[..., :n_sv]
    m, n = mat2d.shape
    if m < n:
        mat2d = mat2d.T
    if _batched_svd is not None and min(m, n) <= 32:
        try:
            _, s, _ = _batched_svd(mat2d.unsqueeze(0))
            return s.squeeze(0)[:n_sv]
        except Exception:
            pass
    return torch.linalg.svdvals(mat2d)[:n_sv]


TUNER_DATA_DIR = os.path.join(folder_paths.models_dir, "tuner_data")
os.makedirs(TUNER_DATA_DIR, exist_ok=True)
folder_paths.add_model_folder_path("tuner_data", TUNER_DATA_DIR)

AUTOTUNER_MEMORY_DIR = os.path.join(folder_paths.models_dir, "autotuner_memory")
os.makedirs(AUTOTUNER_MEMORY_DIR, exist_ok=True)
AUTOTUNER_MEMORY_VERSION = 1
AUTOTUNER_ALGO_VERSION = "1.4.5"  # Bump when scoring/analysis logic changes




def _read_safetensors_metadata(filepath):
    """Read metadata header from a safetensors file without loading tensors."""
    try:
        if not filepath.endswith(".safetensors"):
            return {}
        with safe_open(filepath, framework="pt") as f:
            return dict(f.metadata()) if f.metadata() else {}
    except Exception:
        return {}


def _resolve_safe_output_path(base_dir, filename, suffix, label):
    """Resolve a save path under `base_dir`, allowing subdirectories but blocking traversal."""
    if filename is None:
        raise ValueError(f"[{label}] Filename is required.")
    base_dir_real = os.path.realpath(base_dir)
    base_name = filename if filename.endswith(suffix) else f"{filename}{suffix}"
    candidate = os.path.realpath(os.path.join(base_dir_real, base_name))
    try:
        if os.path.commonpath([base_dir_real, candidate]) != base_dir_real:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"[{label}] Path escapes base directory: {filename}") from exc
    return candidate


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

_VIDEO_ARCH_ORTHOGONAL_FLOOR = {"wan": 1.0, "ltx": 1.0}

_ARCH_TO_PRESET = {
    "sdxl": "sd_unet", "unknown": "sd_unet",
    "flux": "dit", "wan": "dit", "zimage": "dit", "ltx": "dit",
    "acestep": "dit",
    "qwen_image": "llm",
}


def _resolve_arch_preset(arch_override, detected_arch):
    """Resolve architecture preset from override or detected architecture."""
    if arch_override and arch_override != "auto" and arch_override in _ARCH_PRESETS:
        key = arch_override
    else:
        key = _ARCH_TO_PRESET.get(detected_arch, "sd_unet")
    return key, _ARCH_PRESETS[key]


def _parse_merge_formula(formula_str, n_loras):
    """
    Parse a merge formula string into a tree structure.

    Syntax:
        expr   = term (('+') term)*
        term   = atom (':' weight)?
        atom   = NUMBER | '(' expr ')'
        weight = FLOAT

    Numbers are 1-indexed LoRA positions. Returns a tree of:
        {"type": "leaf", "index": int, "weight": float|None}
        {"type": "group", "children": list, "weight": float|None}

    Raises ValueError on malformed input or out-of-range indices.
    """
    formula_str = formula_str.strip()
    if not formula_str:
        raise ValueError("Empty merge formula")

    pos = [0]  # mutable position cursor

    def _skip_ws():
        while pos[0] < len(formula_str) and formula_str[pos[0]] == ' ':
            pos[0] += 1

    def _parse_weight():
        _skip_ws()
        if pos[0] < len(formula_str) and formula_str[pos[0]] == ':':
            pos[0] += 1  # skip ':'
            _skip_ws()
            start = pos[0]
            while pos[0] < len(formula_str) and (formula_str[pos[0]].isdigit() or formula_str[pos[0]] == '.'):
                pos[0] += 1
            if pos[0] == start:
                raise ValueError(f"Expected weight after ':' at position {pos[0]}")
            return float(formula_str[start:pos[0]])
        return None

    def _parse_atom():
        _skip_ws()
        if pos[0] >= len(formula_str):
            raise ValueError("Unexpected end of formula")

        if formula_str[pos[0]] == '(':
            pos[0] += 1  # skip '('
            node = _parse_expr()
            _skip_ws()
            if pos[0] >= len(formula_str) or formula_str[pos[0]] != ')':
                raise ValueError(f"Expected ')' at position {pos[0]}")
            pos[0] += 1  # skip ')'
            weight = _parse_weight()
            if weight is not None:
                node["weight"] = weight
            return node

        # Must be a number
        start = pos[0]
        while pos[0] < len(formula_str) and formula_str[pos[0]].isdigit():
            pos[0] += 1
        if pos[0] == start:
            raise ValueError(f"Unexpected character '{formula_str[pos[0]]}' at position {pos[0]}")
        index_1based = int(formula_str[start:pos[0]])
        if index_1based < 1 or index_1based > n_loras:
            raise ValueError(f"LoRA index {index_1based} out of range (have {n_loras} LoRAs)")
        weight = _parse_weight()
        return {"type": "leaf", "index": index_1based - 1, "weight": weight}

    def _parse_expr():
        children = [_parse_atom()]
        while True:
            _skip_ws()
            if pos[0] < len(formula_str) and formula_str[pos[0]] == '+':
                pos[0] += 1  # skip '+'
                children.append(_parse_atom())
            else:
                break
        if len(children) == 1:
            return children[0]
        return {"type": "group", "children": children, "weight": None}

    result = _parse_expr()
    _skip_ws()
    if pos[0] != len(formula_str):
        raise ValueError(f"Unexpected content at position {pos[0]}: '{formula_str[pos[0]:]}'")
    return result


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
            "metadata": _read_safetensors_metadata(lora_path),
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
            import tempfile, atexit, weakref
            self._cache_dir = tempfile.mkdtemp(prefix="lora_diff_cache_",
                                                   dir=folder_paths.get_temp_directory())
            # Clean up temp directory on process exit (guards against crash-orphaned files)
            _dir = self._cache_dir
            def _cleanup(d=_dir):
                import shutil
                shutil.rmtree(d, ignore_errors=True)
            atexit.register(_cleanup)
            weakref.finalize(self, _cleanup)
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
            return val.to(device) if device is not None else val
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
        self._ram_store[key] = cached
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

        # Qwen-Image: transformer_blocks with img_mlp/txt_mlp/img_mod/txt_mod
        # Must be checked BEFORE FLUX — both use transformer.transformer_blocks
        # but Qwen has dual-stream markers (img_mlp, txt_mlp, img_mod, txt_mod, add_q_proj).
        # Also detect Qwen LoRAs that only target attention (to_q/to_k/to_v) without
        # dual-stream markers — these have transformer.transformer_blocks but lack
        # FLUX-specific double_blocks/single_blocks patterns.
        _has_qwen_markers = any('transformer_blocks' in k and
               any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod', 'add_q_proj'])
               for k in keys)
        if _has_qwen_markers:
            return 'qwen_image'
        # Qwen attention-only LoRAs: transformer.transformer_blocks with to_q/to_k/to_v
        # but NO double_blocks/single_blocks (which would indicate FLUX)
        _has_transformer_blocks = any('transformer.transformer_blocks' in k for k in keys)
        _has_flux_blocks = any('double_blocks' in k or 'single_blocks' in k for k in keys)
        if _has_transformer_blocks and not _has_flux_blocks:
            # transformer.transformer_blocks without FLUX block patterns = Qwen-Image
            return 'qwen_image'

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
        # ~4x weight amplification (rank 16) to ~5.66x (rank 32). When alpha
        # is missing, _get_lora_key_info defaults alpha=rank (scale=1.0),
        # which is correct for standard LoRAs and only slightly weak for
        # RS-LoRA.

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

    _ACESTEP_COMPOUND_NAMES = sorted([
        "self_attn", "cross_attn",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lora_up", "lora_down", "lora_A", "lora_B", "lora_mid",
    ], key=len, reverse=True)

    @classmethod
    def _acestep_underscore_to_dot(cls, rest):
        """Convert Kohya-style ACE-Step keys to dotted form while preserving
        compound component names like self_attn and q_proj."""
        protected = []
        for i, name in enumerate(cls._ACESTEP_COMPOUND_NAMES):
            placeholder = f"\x01{i}\x01"
            if name in rest:
                rest = rest.replace(name, placeholder)
                protected.append((placeholder, name))
        rest = rest.replace("_", ".")
        for placeholder, name in protected:
            rest = rest.replace(placeholder, name)
        return rest

    @classmethod
    def _normalize_keys_acestep(cls, lora_sd):
        """
        Normalize ACE-Step LoRA keys to canonical
        diffusion_model.layers.N.{self,cross}_attn.* form.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            if new_k.startswith("base_model.model."):
                new_k = new_k[len("base_model.model."):]

            if new_k.startswith("lora_unet_"):
                rest = new_k[len("lora_unet_"):]
                rest = cls._acestep_underscore_to_dot(rest)
                new_k = f"diffusion_model.{rest}"

            new_k = re.sub(r"^transformer\.", "diffusion_model.", new_k)
            new_k = re.sub(r"^model\.", "diffusion_model.", new_k)

            if new_k.startswith("layers."):
                new_k = "diffusion_model." + new_k

            if new_k.startswith("diffusion_model.layers."):
                new_k = re.sub(r"\.to_(q|k|v|out)\.", lambda m: f".{m.group(1)}_proj.", new_k)

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
        elif architecture == 'acestep':
            return cls._normalize_keys_acestep(lora_sd)
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
                    store_dtype = q_diff.dtype
                    fused_diff = torch.cat([q_diff, k_diff, v_diff], dim=0)
                    if store_dtype not in (torch.float32, torch.float64):
                        fused_diff = fused_diff.to(store_dtype)
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
                        if store_dtype not in (torch.float32, torch.float64):
                            fused_diff = fused_diff.to(store_dtype)
                        fused[fused_key] = ("diff", (fused_diff,))
                elif any(hasattr(p, "weights") for p in [q_patch, k_patch, v_patch]):
                    store_dtype = torch.float16
                    for candidate in [q_patch, k_patch, v_patch]:
                        if hasattr(candidate, "weights") and candidate.weights[0] is not None:
                            dtype = candidate.weights[0].dtype
                            if dtype not in (torch.float32, torch.float64):
                                store_dtype = dtype
                                break
                    parts = [_LoRAMergeBase._expand_patch_to_diff(comp_patch) for comp_patch in [q_patch, k_patch, v_patch]]
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

    @staticmethod
    def _collect_lora_prefixes(active_loras):
        """Collect all LoRA key prefixes from a stack in deterministic order."""
        all_lora_prefixes = set()
        suffixes = [
            ".lora_up.weight", ".lora_down.weight",
            "_lora.up.weight", "_lora.down.weight",
            ".lora_B.weight", ".lora_A.weight",
            ".lora.up.weight", ".lora.down.weight",
            # LoKr (Kronecker)
            ".lokr_w1", ".lokr_w2",
            ".lokr_w1_a", ".lokr_w1_b",
            ".lokr_w2_a", ".lokr_w2_b",
            ".lokr_t2",
            # LoHa (Hadamard)
            ".hada_w1_a", ".hada_w1_b",
            ".hada_w2_a", ".hada_w2_b",
            ".hada_t1", ".hada_t2",
            ".alpha",
        ]
        for item in active_loras:
            if item.get("_precomputed_diffs"):
                for key in item["lora"]:
                    all_lora_prefixes.add(key)
                continue
            for key in item["lora"].keys():
                for suffix in suffixes:
                    if key.endswith(suffix):
                        all_lora_prefixes.add(key[:-len(suffix)])
                        break
        return sorted(all_lora_prefixes, key=lambda x: (str(x),))

    @staticmethod
    def _resolve_target_key(lora_prefix, model_keys, clip_keys):
        """Resolve a LoRA prefix to a model/CLIP target key."""
        if lora_prefix in model_keys:
            return (model_keys[lora_prefix], False)
        if lora_prefix in clip_keys:
            return (clip_keys[lora_prefix], True)
        return (None, False)

    @staticmethod
    def _make_target_group_id(target_key, is_clip):
        return (bool(is_clip), target_key)

    @staticmethod
    def _choose_canonical_prefix(aliases):
        """Pick a stable human-readable label for an alias group."""
        if not aliases:
            return ""
        def _sort_key(alias):
            return (
                alias.startswith("lora_"),
                alias.count("_"),
                len(alias),
                alias,
            )
        return min(sorted(set(aliases)), key=_sort_key)

    def _build_target_groups(self, all_lora_prefixes, model_keys, clip_keys):
        """
        Group aliases by resolved (is_clip, target_key) so analysis/merge operates
        on actual model weights rather than raw trainer-specific prefixes.
        """
        grouped = {}
        # Build reverse lookup: target_key → True for fast virtual-key detection
        model_target_keys = set()
        for v in (model_keys.values() if model_keys else []):
            model_target_keys.add(v)
            if isinstance(v, tuple):
                model_target_keys.add(v[0])
        clip_target_keys = set()
        for v in (clip_keys.values() if clip_keys else []):
            clip_target_keys.add(v)
            if isinstance(v, tuple):
                clip_target_keys.add(v[0])
        for prefix in all_lora_prefixes:
            target_key, is_clip = self._resolve_target_key(prefix, model_keys, clip_keys)
            if target_key is None:
                # Virtual LoRA keys are already target keys — map to themselves
                if prefix in model_target_keys:
                    target_key, is_clip = prefix, False
                elif prefix in clip_target_keys:
                    target_key, is_clip = prefix, True
                else:
                    continue
            group_id = self._make_target_group_id(target_key, is_clip)
            entry = grouped.setdefault(group_id, {
                "target_key": target_key,
                "is_clip": is_clip,
                "aliases": [],
            })
            entry["aliases"].append(prefix)

        ordered = {}
        prepared = []
        for entry in grouped.values():
            aliases = sorted(a for a in set(entry["aliases"]) if isinstance(a, str))
            if not aliases:
                canonical = str(entry["target_key"])
                aliases = [canonical]
            else:
                canonical = self._choose_canonical_prefix(aliases)
            prepared.append((entry["is_clip"], canonical, {
                "target_key": entry["target_key"],
                "is_clip": entry["is_clip"],
                "aliases": aliases,
                "label_prefix": canonical,
            }))
        for _is_clip, canonical, entry in sorted(prepared, key=lambda item: (item[0], item[1])):
            ordered[canonical] = entry
        return ordered

    @staticmethod
    def _group_target_key(target_group):
        return target_group["target_key"]

    def _resolve_target_shape(self, target_key, is_clip, model, clip):
        """Resolve the actual target tensor shape for a target key."""
        offset = None
        if isinstance(target_key, tuple):
            actual_key = target_key[0]
            if len(target_key) > 1:
                offset = target_key[1]
        else:
            actual_key = target_key

        if is_clip:
            target_weight = comfy.utils.get_attr(clip.cond_stage_model, actual_key)
        else:
            target_weight = comfy.utils.get_attr(model.model, actual_key)

        target_shape = list(target_weight.shape)
        if offset is not None:
            target_shape[offset[0]] = offset[2]
        return torch.Size(target_shape)

    @staticmethod
    def _resolve_branch_strength(item, is_clip, clip_strength_multiplier):
        """Base per-LoRA strength for the target branch before auto-scaling."""
        if is_clip:
            if item["clip_strength"] is not None:
                return item["clip_strength"]
            return item["strength"]
        return item["strength"]

    def _apply_conflict_modes(self, diffs, eff_strengths, active_loras, merge_refinement="none"):
        """Apply per-LoRA conflict_mode masking to already-aggregated diffs."""
        if len(diffs) <= 1:
            return diffs
        if not any(active_loras[idx].get("conflict_mode", "all") != "all" for idx in diffs):
            return diffs

        indices = sorted(diffs.keys())
        ref = diffs[indices[0]]
        if merge_refinement != "none" and ref.dim() >= 2:
            out_dim = ref.shape[0]
            sign_sum = torch.zeros(out_dim, device=ref.device, dtype=torch.float32)
            for idx in indices:
                effective = diffs[idx] if eff_strengths[idx] >= 0 else -diffs[idx]
                sign_sum += effective.reshape(out_dim, -1).to(dtype=torch.float32).sum(dim=1).sign()
            majority_sign = torch.where(sign_sum >= 0, 1.0, -1.0)
            majority_sign = majority_sign.reshape(-1, *([1] * (ref.dim() - 1))).expand_as(ref)
        else:
            sign_sum = torch.zeros_like(ref, dtype=torch.float32)
            for idx in indices:
                effective = diffs[idx] if eff_strengths[idx] >= 0 else -diffs[idx]
                sign_sum += effective.sign()
            majority_sign = torch.where(sign_sum >= 0, 1.0, -1.0)

        masked = {}
        for idx in indices:
            diff = diffs[idx]
            effective = diff if eff_strengths[idx] >= 0 else -diff
            cm = active_loras[idx].get("conflict_mode", "all")
            if cm == "low_conflict":
                diff = diff * ((effective * majority_sign) > 0).float()
            elif cm == "high_conflict":
                diff = diff * ((effective * majority_sign) < 0).float()
            masked[idx] = diff
        return masked

    def _prepare_group_diffs(self, target_group, active_loras, model, clip, device,
                             clip_strength_multiplier=1.0, merge_refinement="none",
                             diff_cache=None, auto_scale=1.0):
        """
        Aggregate all alias contributions that resolve to the same target weight.
        Returns metadata plus one diff per contributing LoRA after key_filter and
        conflict_mode are applied.
        """
        target_key = self._group_target_key(target_group)
        is_clip = target_group["is_clip"]

        try:
            target_shape = self._resolve_target_shape(target_key, is_clip, model, clip)
        except (AttributeError, RuntimeError, IndexError):
            return None

        use_gpu = device is not None and device.type != "cpu"
        aggregated = {}
        ranks = {}
        raw_contributors = set()
        storage_dtype = None
        skip_count = 0

        for i, item in enumerate(active_loras):
            diff_accum = None
            rank_sum = 0

            # Virtual LoRAs from sub-merges store pre-computed diffs keyed by target key
            if item.get("_precomputed_diffs"):
                tkey = target_key
                raw = item["lora"].get(tkey)
                if raw is None and isinstance(tkey, tuple):
                    raw = item["lora"].get(tkey[0])
                if raw is not None:
                    if isinstance(raw, torch.Tensor):
                        diff = raw.float()
                    else:
                        diff = self._expand_patch_to_diff(raw)
                    if device is not None and diff.device != device:
                        diff = diff.to(device)
                    try:
                        diff = diff.reshape(target_shape)
                    except RuntimeError:
                        diff = None
                    if diff is not None:
                        raw_contributors.add(i)
                        rank_sum += 1
                        if storage_dtype is None:
                            storage_dtype = raw.dtype if isinstance(raw, torch.Tensor) else diff.dtype
                        diff_accum = diff
                if diff_accum is not None:
                    aggregated[i] = diff_accum
                    ranks[i] = rank_sum
                continue

            for alias in target_group["aliases"]:
                cache_key = (alias, i)
                if diff_cache is not None and cache_key in diff_cache:
                    diff = diff_cache.get(cache_key, device=device if use_gpu else None)
                    if diff is not None:
                        diff = diff.float()
                        raw_contributors.add(i)
                        rank_sum += 1  # rank unknown from cache
                        diff_accum = diff if diff_accum is None else diff_accum + diff
                    continue

                lora_info = self._get_lora_key_info(item["lora"], alias)
                if lora_info is not None:
                    mat_up, mat_down, alpha, mid = lora_info
                    rank_sum += mat_down.shape[0]
                    raw_contributors.add(i)
                    if storage_dtype is None:
                        storage_dtype = mat_up.dtype
                    diff = self._compute_lora_diff(
                        mat_up, mat_down, alpha, mid, target_shape,
                        device=device if use_gpu else None,
                        to_cpu=not use_gpu,
                    )
                else:
                    # Try LoKr / LoHa formats
                    alt = self._get_lokr_diff(
                        item["lora"], alias,
                        device=device if use_gpu else None, to_cpu=not use_gpu,
                    )
                    if alt is None:
                        alt = self._get_loha_diff(
                            item["lora"], alias,
                            device=device if use_gpu else None, to_cpu=not use_gpu,
                        )
                    if alt is not None:
                        diff, alt_rank, alt_dtype = alt
                        try:
                            diff = diff.reshape(target_shape)
                        except RuntimeError:
                            diff = None
                        if diff is not None:
                            rank_sum += alt_rank
                            raw_contributors.add(i)
                            if storage_dtype is None:
                                storage_dtype = alt_dtype
                    else:
                        diff = None

                if diff is not None:
                    if diff_cache is not None:
                        diff_cache.put(cache_key, diff)
                    diff = diff.float()
                    diff_accum = diff if diff_accum is None else diff_accum + diff

            if diff_accum is not None:
                aggregated[i] = diff_accum
                ranks[i] = rank_sum
            elif i in raw_contributors:
                pass  # contributed via cache but diff_accum ended up None (shouldn't happen)
            else:
                skip_count += 1

        raw_n = len(aggregated)
        if raw_n == 0:
            if skip_count > 0:
                return {
                    "label_prefix": target_group["label_prefix"],
                    "target_key": target_key,
                    "is_clip": is_clip,
                    "raw_n_loras": 0,
                    "diffs": {},
                    "eff_strengths": {},
                    "rank_sums": {},
                    "target_shape": target_shape,
                    "storage_dtype": storage_dtype,
                    "skip_count": skip_count,
                }
            return None

        filtered = {}
        eff_strengths = {}
        rank_sums = {}
        for i, diff in aggregated.items():
            kf = active_loras[i].get("key_filter", "all")
            if kf == "shared_only" and raw_n < 2:
                continue
            if kf == "unique_only" and raw_n != 1:
                continue
            filtered[i] = diff
            eff_strengths[i] = self._resolve_branch_strength(
                active_loras[i], is_clip, clip_strength_multiplier
            ) * auto_scale
            rank_sums[i] = ranks.get(i, 0)

        if not filtered:
            return {
                "label_prefix": target_group["label_prefix"],
                "target_key": target_key,
                "is_clip": is_clip,
                "raw_n_loras": raw_n,
                "diffs": {},
                "eff_strengths": {},
                "rank_sums": {},
                "target_shape": target_shape,
                "storage_dtype": storage_dtype,
                "skip_count": skip_count,
            }

        filtered = self._apply_conflict_modes(
            filtered, eff_strengths, active_loras, merge_refinement=merge_refinement
        )

        return {
            "label_prefix": target_group["label_prefix"],
            "target_key": target_key,
            "is_clip": is_clip,
            "raw_n_loras": raw_n,
            "diffs": filtered,
            "eff_strengths": eff_strengths,
            "rank_sums": rank_sums,
            "target_shape": target_shape,
            "storage_dtype": storage_dtype,
            "skip_count": skip_count,
        }

    def _get_lora_key_info(self, lora_dict, key_prefix):
        """
        Extracts LoRA information for the given key.
        Returns (mat_up, mat_down, alpha, mid) or None for standard LoRA.
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

    @staticmethod
    def _has_lokr_keys(lora_dict, key_prefix):
        """Check if lora_dict has LoKr keys for the given prefix (no tensor loading)."""
        p = key_prefix
        has_w1 = f"{p}.lokr_w1" in lora_dict or (f"{p}.lokr_w1_a" in lora_dict and f"{p}.lokr_w1_b" in lora_dict)
        has_w2 = f"{p}.lokr_w2" in lora_dict or (f"{p}.lokr_w2_a" in lora_dict and f"{p}.lokr_w2_b" in lora_dict)
        return has_w1 and has_w2

    @staticmethod
    def _has_loha_keys(lora_dict, key_prefix):
        """Check if lora_dict has LoHa keys for the given prefix (no tensor loading)."""
        p = key_prefix
        return (f"{p}.hada_w1_a" in lora_dict and f"{p}.hada_w1_b" in lora_dict
                and f"{p}.hada_w2_a" in lora_dict and f"{p}.hada_w2_b" in lora_dict)

    def _get_lokr_diff(self, lora_dict, key_prefix, device=None, to_cpu=True):
        """
        Extract LoKr (Kronecker) factors and compute full diff.
        Returns (diff, rank, dtype) or None.
        """
        p = key_prefix
        w1 = lora_dict.get(f"{p}.lokr_w1")
        w2 = lora_dict.get(f"{p}.lokr_w2")
        w1_a = lora_dict.get(f"{p}.lokr_w1_a")
        w1_b = lora_dict.get(f"{p}.lokr_w1_b")
        w2_a = lora_dict.get(f"{p}.lokr_w2_a")
        w2_b = lora_dict.get(f"{p}.lokr_w2_b")
        t2 = lora_dict.get(f"{p}.lokr_t2")

        has_w1 = w1 is not None or (w1_a is not None and w1_b is not None)
        has_w2 = w2 is not None or (w2_a is not None and w2_b is not None)
        if not (has_w1 and has_w2):
            return None

        alpha = lora_dict.get(f"{p}.alpha")
        if alpha is not None:
            alpha = alpha.item()

        dim = None
        ref_tensor = w1 if w1 is not None else (w1_a if w1_a is not None else w2_a)
        dtype = ref_tensor.dtype

        if device is not None:
            w1 = w1.to(device) if w1 is not None else None
            w2 = w2.to(device) if w2 is not None else None
            w1_a = w1_a.to(device) if w1_a is not None else None
            w1_b = w1_b.to(device) if w1_b is not None else None
            w2_a = w2_a.to(device) if w2_a is not None else None
            w2_b = w2_b.to(device) if w2_b is not None else None
            t2 = t2.to(device) if t2 is not None else None

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
        diff = torch.kron(w1, w2) * scale
        rank = dim if dim is not None else min(w1.shape)

        if to_cpu and diff.device.type != "cpu":
            diff = diff.cpu()
        return (diff, rank, dtype)

    def _get_loha_diff(self, lora_dict, key_prefix, device=None, to_cpu=True):
        """
        Extract LoHa (Hadamard) factors and compute full diff.
        Returns (diff, rank, dtype) or None.
        """
        p = key_prefix
        w1a = lora_dict.get(f"{p}.hada_w1_a")
        w1b = lora_dict.get(f"{p}.hada_w1_b")
        w2a = lora_dict.get(f"{p}.hada_w2_a")
        w2b = lora_dict.get(f"{p}.hada_w2_b")
        if w1a is None or w1b is None or w2a is None or w2b is None:
            return None

        t1 = lora_dict.get(f"{p}.hada_t1")
        t2 = lora_dict.get(f"{p}.hada_t2")
        alpha = lora_dict.get(f"{p}.alpha")
        if alpha is not None:
            alpha = alpha.item()

        dtype = w1a.dtype
        rank = w1b.shape[0]

        if device is not None:
            w1a = w1a.to(device)
            w1b = w1b.to(device)
            w2a = w2a.to(device)
            w2b = w2b.to(device)
            t1 = t1.to(device) if t1 is not None else None
            t2 = t2.to(device) if t2 is not None else None

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
        diff = (m1 * m2) * scale

        if to_cpu and diff.device.type != "cpu":
            diff = diff.cpu()
        return (diff, rank, dtype)
    
    def _compute_lora_diff(self, mat_up, mat_down, alpha, mid, target_shape, device=None, to_cpu=True):
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
        if to_cpu and device is not None and device.type != "cpu":
            return diff.cpu()
        return diff

    @staticmethod
    def _stable_data_hash(value):
        """Create a compact stable hash for nested JSON-like data and tensors."""
        def _normalize(obj):
            if isinstance(obj, torch.Tensor):
                t = obj.detach().float().cpu()
                sample = t.flatten()[:16].tolist()
                return {
                    "__tensor__": True,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "sample": sample,
                }
            if isinstance(obj, dict):
                return {str(k): _normalize(obj[k]) for k in sorted(obj.keys(), key=str)}
            if isinstance(obj, (list, tuple)):
                return [_normalize(v) for v in obj]
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return repr(obj)

        payload = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _safe_unit_clamp(value):
        return max(-1.0, min(1.0, float(value)))


    @staticmethod
    def _compute_subspace_basis(diff, rank_hint=None, max_rank=8):
        if diff is None:
            return None

        if diff.dim() >= 2:
            mat = diff.reshape(diff.shape[0], -1).float()
        else:
            mat = diff.float().reshape(1, -1)

        if min(mat.shape) <= 0:
            return None

        rank_cap = min(max_rank, min(mat.shape))
        if rank_hint is not None:
            rank_cap = min(rank_cap, max(1, int(rank_hint)))
        rank_cap = max(rank_cap, 1)

        try:
            q = min(rank_cap, min(mat.shape))
            U, _S, V = torch.svd_lowrank(mat, q=q)
            return {"left": U[:, :q], "right": V[:, :q]}
        except Exception:
            try:
                left, _ = torch.linalg.qr(mat, mode="reduced")
                right, _ = torch.linalg.qr(mat.T, mode="reduced")
                return {
                    "left": left[:, :rank_cap],
                    "right": right[:, :rank_cap],
                }
            except Exception:
                return None

    @classmethod
    def _compute_subspace_overlap(cls, basis_a, basis_b):
        if not basis_a or not basis_b:
            return 0.0

        def _project_overlap(q_a, q_b):
            if q_a is None or q_b is None or q_a.numel() == 0 or q_b.numel() == 0:
                return None
            cross = torch.mm(q_a.transpose(0, 1), q_b)
            denom = max(1, min(q_a.shape[1], q_b.shape[1]))
            value = (cross.square().sum() / denom).item()
            return min(max(value, 0.0), 1.0)

        left = _project_overlap(basis_a.get("left"), basis_b.get("left"))
        right = _project_overlap(basis_a.get("right"), basis_b.get("right"))
        values = [v for v in (left, right) if v is not None]
        return sum(values) / len(values) if values else 0.0

    def _sample_pair_metrics(self, diff_a, diff_b, basis_a=None, basis_b=None, device=None):
        """
        Pairwise overlap/conflict metrics with a magnitude-aware noise floor,
        excess-conflict baseline, and optional subspace overlap.
        """
        flat_a = diff_a.flatten()
        flat_b = diff_b.flatten()
        if device is not None:
            flat_a = flat_a.to(device=device, dtype=torch.float32)
            flat_b = flat_b.to(device=device, dtype=torch.float32)
        elif flat_a.dtype != torch.float32:
            flat_a = flat_a.float()
            flat_b = flat_b.float()

        if flat_a.numel() != flat_b.numel():
            return {
                "overlap": 0,
                "conflict": 0,
                "dot": 0.0,
                "norm_a_sq": 0.0,
                "norm_b_sq": 0.0,
                "weighted_total": 0.0,
                "weighted_conflict": 0.0,
                "expected_conflict": 0.0,
                "excess_conflict": 0.0,
                "subspace_overlap": 0.0,
                "subspace_weight": 0.0,
            }

        n = flat_a.numel()
        if n > 100000:
            target_device = flat_a.device
            g = torch.Generator(device=target_device).manual_seed(42)
            indices = torch.randperm(n, device=target_device, generator=g)[:100000]
            flat_a = flat_a[indices]
            flat_b = flat_b[indices]

        mask = (flat_a != 0) & (flat_b != 0)
        n_overlap = mask.sum().item()
        if n_overlap == 0:
            return {
                "overlap": 0,
                "conflict": 0,
                "dot": 0.0,
                "norm_a_sq": 0.0,
                "norm_b_sq": 0.0,
                "weighted_total": 0.0,
                "weighted_conflict": 0.0,
                "expected_conflict": 0.0,
                "excess_conflict": 0.0,
                "subspace_overlap": 0.0,
                "subspace_weight": 0.0,
            }

        a_overlap = flat_a[mask]
        b_overlap = flat_b[mask]
        dot = (a_overlap * b_overlap).sum().item()
        norm_a_sq = (a_overlap * a_overlap).sum().item()
        norm_b_sq = (b_overlap * b_overlap).sum().item()
        n_conflict = (a_overlap.sign() != b_overlap.sign()).sum().item()

        a_rms = a_overlap.square().mean().sqrt()
        b_rms = b_overlap.square().mean().sqrt()
        noise_floor = max(a_rms.item(), b_rms.item()) * 0.05
        strong_mask = (a_overlap.abs() > noise_floor) & (b_overlap.abs() > noise_floor)
        if strong_mask.any():
            a_strong = a_overlap[strong_mask]
            b_strong = b_overlap[strong_mask]
        else:
            a_strong = a_overlap
            b_strong = b_overlap

        weights = torch.minimum(a_strong.abs(), b_strong.abs())
        weighted_total = weights.sum().item()
        mismatch = a_strong.sign() != b_strong.sign()
        weighted_conflict = weights[mismatch].sum().item() if weighted_total > 0 else 0.0

        denom = math.sqrt(norm_a_sq * norm_b_sq) if norm_a_sq > 0 and norm_b_sq > 0 else 0.0
        cos_sim = dot / denom if denom > 0 else 0.0
        cos_sim = self._safe_unit_clamp(cos_sim)
        expected_conflict = math.acos(cos_sim) / math.pi if denom > 0 else 0.0
        weighted_ratio = (weighted_conflict / weighted_total) if weighted_total > 0 else 0.0
        excess_conflict = max(weighted_ratio - expected_conflict, 0.0)

        subspace_overlap = self._compute_subspace_overlap(basis_a, basis_b)
        subspace_weight = math.sqrt(norm_a_sq * norm_b_sq) if norm_a_sq > 0 and norm_b_sq > 0 else 0.0

        return {
            "overlap": n_overlap,
            "conflict": n_conflict,
            "dot": dot,
            "norm_a_sq": norm_a_sq,
            "norm_b_sq": norm_b_sq,
            "weighted_total": weighted_total,
            "weighted_conflict": weighted_conflict,
            "expected_conflict": expected_conflict,
            "excess_conflict": excess_conflict,
            "subspace_overlap": subspace_overlap,
            "subspace_weight": subspace_weight,
        }

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
            return tensor
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
            weights = patch.weights
            w1, w2, alpha = weights[0], weights[1], weights[2]
            w1_a, w1_b = weights[3], weights[4]
            w2_a, w2_b, t2 = weights[5], weights[6], weights[7]
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
            weights = patch.weights
            w1a, w1b, alpha = weights[0], weights[1], weights[2]
            w2a, w2b = weights[3], weights[4]
            t1, t2 = weights[5], weights[6]
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
            if isinstance(patch, LoHaAdapter):
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
            consensus_d = torch.where(selfish_mask, torch.zeros(1, device=d.device, dtype=d.dtype), d)
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
        # Operates in-place on directions (not used after this loop)
        ortho = []
        for i in range(len(directions)):
            q = directions[i]
            for j in range(len(ortho)):
                proj = torch.dot(q, ortho[j])
                q = q - proj * ortho[j]
            q_norm = q.norm()
            if q_norm > 1e-8:
                q = q / q_norm
            else:
                q = torch.zeros_like(q)
            ortho.append(q)
        del directions

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
    def _procrustes_align(diffs_with_weights, compute_device=None, svd_device=None):
        """
        Procrustes alignment: rotate each LoRA diff toward the weighted-mean
        reference via optimal rotation. Uses batched_procrustes from kernel.py
        with whiten=False so Frobenius norms are preserved (pure rotation).

        Treats each diff as (n_samples=out_dim, N=in_dim): aligns input-space
        directions. Batches all diffs into a single (B, out_dim, in_dim) call.
        """
        if _batched_procrustes is None:
            return diffs_with_weights
        if len(diffs_with_weights) < 2:
            return diffs_with_weights

        ref = diffs_with_weights[0][0]
        if ref.dim() < 2 or min(ref.shape) < 2:
            return diffs_with_weights

        n = len(diffs_with_weights)
        out_dim = ref.shape[0]
        in_dim = ref.reshape(out_dim, -1).shape[1]
        original_shape = ref.shape
        original_dtype = ref.dtype

        dev = svd_device if svd_device is not None else (compute_device or ref.device)
        output_device = compute_device if compute_device is not None else ref.device

        # Weighted mean reference
        total_w = sum(abs(w) for _, w in diffs_with_weights)
        if total_w < 1e-12:
            return diffs_with_weights
        ref_mat = sum(
            d.reshape(out_dim, in_dim).to(device=dev, dtype=torch.float32) * (abs(w) / total_w)
            for d, w in diffs_with_weights
        )

        source_batch = torch.stack([
            d.reshape(out_dim, in_dim).to(device=dev, dtype=torch.float32)
            for d, _ in diffs_with_weights
        ], dim=0)
        target_batch = ref_mat.unsqueeze(0).expand(n, -1, -1)

        try:
            # Get rotation from batched_procrustes, then apply to uncentered source.
            # batched_procrustes centers internally, which corrupts LoRA diffs.
            # We extract R and compute src @ R ourselves.
            _, info = _batched_procrustes(
                source_batch, target_batch, whiten=False)
            R = info.get('rotation') if 'rotation' in info else info.get('rotation_k')
            if R is None:
                del source_batch, target_batch, ref_mat
                return diffs_with_weights
            if 'projection' in info:
                # Subspace path: R_k is in projected space, need to lift back
                P = info['projection']  # (B, N, k)
                P_T = P.transpose(1, 2)
                src_in = torch.bmm(source_batch, P)  # (B, out, k)
                src_perp = source_batch - torch.bmm(src_in, P_T)
                aligned_batch = torch.bmm(torch.bmm(src_in, R), P_T) + src_perp
            else:
                aligned_batch = torch.bmm(source_batch, R)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logging.warning(f"[LoRA Optimizer] Procrustes align failed ({e}), skipping")
            del source_batch, target_batch, ref_mat
            return diffs_with_weights
        del source_batch, target_batch, ref_mat

        result = []
        for i, (_, w) in enumerate(diffs_with_weights):
            aligned_diff = aligned_batch[i].reshape(original_shape)
            if aligned_diff.device != output_device:
                aligned_diff = aligned_diff.to(output_device)
            result.append((aligned_diff.to(dtype=original_dtype), w))
        del aligned_batch

        return result

    @staticmethod
    @torch.no_grad()
    def _compress_to_lowrank(diff, rank, svd_device=None, output_dtype=None):
        """
        Re-compress a full-rank diff tensor to low-rank via truncated SVD.
        Returns ("lora", (mat_up, mat_down, alpha=rank, None)) so ComfyUI
        computes up @ down * (rank/rank) = up @ down (no extra scaling).

        svd_device: where to run SVD. GPU is ~10-50x faster. CPU if None.
        output_dtype: cast output to this dtype. None = same as input.
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
        if rank > min(mat.shape) // 2:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            del mat
            U = U[:, :rank]
            S = S[:rank]
            V = Vh[:rank, :].T
            del Vh
        else:
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
        mat_up = (U * sqrt_S.unsqueeze(0)).to(output_dtype).cpu()
        mat_down = ((V * sqrt_S.unsqueeze(0)).T).to(output_dtype).cpu()
        del U, S, V, sqrt_S
        # alpha=rank so ComfyUI computes: up @ down * (rank/rank) = up @ down
        return LoRAAdapter(set(), (mat_up, mat_down, float(rank), None, None, None))

    @staticmethod
    @torch.no_grad()
    def _estimate_save_rank(initial_rank, model_patches, clip_patches,
                            max_error=0.05, n_samples=3):
        """
        Estimate the minimum SVD rank needed to reconstruct sample diff patches
        within `max_error` relative Frobenius error.
        """
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
            singular_values = _triton_svdvals(mat, n_sv=min(mat.shape))
            total_sq = (singular_values ** 2).sum().item()
            if total_sq == 0:
                continue
            threshold_sq = (max_error * max_error) * total_sq
            cumulative_sq = 0.0
            needed = len(singular_values)
            for idx in range(len(singular_values)):
                cumulative_sq += singular_values[idx].item() ** 2
                if total_sq - cumulative_sq <= threshold_sq:
                    needed = idx + 1
                    break
            if needed > rank:
                rank = needed
            del singular_values, mat
        return rank

    @torch.no_grad()
    def _merge_diffs(self, diffs_with_weights, mode, density=0.5, majority_sign_method="frequency",
                     compute_device=None, sparsification="disabled",
                     sparsification_density=0.7, sparsification_generator=None,
                     merge_refinement="none", dare_dampening=0.0,
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

                # Guard: if conflict mask covers >40% of positions, the "conflicts"
                # are likely base-rate noise from orthogonal LoRAs (expected ~50%
                # sign disagreement for uncorrelated vectors).  Skip sparsification
                # entirely — there are no real conflicts to resolve.
                conflict_frac = conflict_mask.float().mean().item()
                if conflict_frac > 0.40:
                    del conflict_mask
                    is_conflict = False
                    self._sparsification_skipped = getattr(self, '_sparsification_skipped', 0) + 1

            if is_conflict:
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

        # Refine/full merge refinement pipeline (non-TIES modes)
        # TIES has its own enhancement path below (after trim)
        # Order matters: TALL-masks must run BEFORE orthogonalization.
        # Orthogonalized diffs have uncorrelated element-wise distributions,
        # which causes TALL-masks to classify every position as "selfish"
        # (agreement=1 everywhere), zeroing out all consensus diffs.
        selfish_additions = None
        if merge_refinement != "none" and len(diffs_with_weights) >= 2 and mode != "ties":
            diffs_with_weights, selfish_additions = self._tall_masks(diffs_with_weights)
            first = diffs_with_weights[0][0]
            if first.dim() >= 2:
                diffs_with_weights = self._do_orthogonalize(diffs_with_weights)
            if merge_refinement == "full":
                first = diffs_with_weights[0][0]
                if first.dim() >= 2 and min(first.shape) >= 2:
                    if _batched_procrustes is not None:
                        diffs_with_weights = self._procrustes_align(
                            diffs_with_weights, compute_device=dev, svd_device=dev)
                    else:
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

            # Refine/full merge refinement pipeline for TIES
            # TALL-masks before orthogonalization (see non-TIES comment above)
            ties_selfish = None
            if merge_refinement != "none" and len(trimmed) >= 2:
                # Re-pair trimmed diffs with abs_weights for enhancement pipeline
                trimmed_pairs = list(zip(trimmed, abs_weights))
                trimmed_pairs, ties_selfish = self._tall_masks(trimmed_pairs)
                first = trimmed_pairs[0][0]
                if first.dim() >= 2:
                    trimmed_pairs = self._do_orthogonalize(trimmed_pairs)
                if merge_refinement == "full":
                    first = trimmed_pairs[0][0]
                    if first.dim() >= 2 and min(first.shape) >= 2:
                        if _batched_procrustes is not None:
                            trimmed_pairs = self._procrustes_align(
                                trimmed_pairs, compute_device=dev, svd_device=dev)
                        else:
                            trimmed_pairs = self._knots_align(
                                trimmed_pairs, compute_device=dev, svd_device=dev)
                trimmed = [d for d, _ in trimmed_pairs]
                abs_weights = [w for _, w in trimmed_pairs]
                del trimmed_pairs

            # Step 2: Elect majority sign
            if merge_refinement != "none":
                majority_sign = self._columnwise_elect_sign(trimmed, majority_sign_method)
            else:
                majority_sign = self._ties_elect_sign(trimmed, majority_sign_method)

            # Step 3: Disjoint merge
            if merge_refinement != "none":
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

        # Filter out formula metadata entries (safety net — optimize_merge
        # also strips these before calling _normalize_stack)
        lora_stack = [item for item in lora_stack
                      if not (isinstance(item, dict) and "_merge_formula" in item)]
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
                    lora_path = None
                else:
                    try:
                        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                        lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        self.loaded_loras[lora_name] = lora_dict
                    except Exception as e:
                        logging.warning(f"[LoRA Optimizer] Failed to load LoRA '{lora_name}': {e}")
                        continue

                metadata = {}
                if lora_path is not None:
                    metadata = _read_safetensors_metadata(lora_path)

                normalized.append({
                    "name": lora_name,
                    "lora": lora_dict,
                    "strength": model_str,
                    "clip_strength": clip_str,
                    "conflict_mode": conflict_mode,
                    "key_filter": key_filter,
                    "metadata": metadata,
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
                    "clip_strength": item.get("clip_strength", None),
                    "conflict_mode": item.get("conflict_mode", "all"),
                    "key_filter": item.get("key_filter", "all"),
                    "metadata": item.get("metadata", {}),
                    "_precomputed_diffs": item.get("_precomputed_diffs", False),
                })

        else:
            logging.warning("[LoRA Optimizer] Unrecognized stack format")
            return []

        # Always detect architecture (used for preset selection even without key normalization)
        if len(normalized) > 0:
            arch = "unknown"
            for item in normalized:
                if item.get("_precomputed_diffs"):
                    continue  # virtual LoRAs have model-space keys, not LoRA keys
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
                        if not item.get("_precomputed_diffs"):
                            item["lora"] = self._normalize_keys(item["lora"], arch)
                else:
                    logging.info("[LoRA Optimizer] Architecture: unknown (no key normalization applied)")

            # Update loaded_loras cache to point at normalized dicts so the
            # pre-normalization copies can be garbage-collected.  This avoids
            # keeping both raw and normalized state dicts in memory (saves
            # 500MB-3GB for large models like Qwen).
            for item in normalized:
                name = item["name"]
                if name in self.loaded_loras:
                    self.loaded_loras[name] = item["lora"]
        else:
            self._detected_arch = None

        return normalized

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
    qualities = ["none", "refine", "full"]
    auto_strengths = ["enabled", "disabled"]
    strategy_sets = ["full", "no_slerp", "basic"]

    for spars in sparsifications:
        density_vals = densities if spars != "disabled" else [0.7]
        for density in density_vals:
            damp_vals = dampenings if spars in ("dare", "dare_conflict") else [0.0]
            for dampening in damp_vals:
                for quality in qualities:
                    for auto_str in auto_strengths:
                        # per_prefix: optimizer auto-selects mode per prefix,
                        # merge_mode is irrelevant — emit one per strategy_set
                        for strat_set in strategy_sets:
                            grid.append({
                                "merge_mode": "per_prefix_auto",
                                "sparsification": spars,
                                "sparsification_density": density,
                                "dare_dampening": dampening,
                                "merge_refinement": quality,
                                "auto_strength": auto_str,
                                "optimization_mode": "per_prefix",
                                "strategy_set": strat_set,
                            })
                        # global: merge_mode matters — emit one per mode
                        # strategy_set is irrelevant for global (mode is explicit)
                        for mode in merge_modes:
                            grid.append({
                                "merge_mode": mode,
                                "sparsification": spars,
                                "sparsification_density": density,
                                "dare_dampening": dampening,
                                "merge_refinement": quality,
                                "auto_strength": auto_str,
                                "optimization_mode": "global",
                                "strategy_set": "full",
                            })
    return grid


def _score_config_heuristic(config, avg_conflict_ratio, avg_cos_sim,
                            magnitude_ratio, prefix_stats, arch_preset=None,
                            avg_excess_conflict=None, avg_subspace_overlap=0.0):
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
    quality = config["merge_refinement"]
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

    is_orthogonal = abs(avg_cos_sim) < ortho_cos and avg_subspace_overlap < 0.35

    effective_conflict = avg_conflict_ratio
    if avg_excess_conflict is not None:
        effective_conflict = max(avg_excess_conflict, 0.0)
        if avg_subspace_overlap > 0:
            effective_conflict *= (0.5 + 0.5 * avg_subspace_overlap)

    decision_conflicts = [
        ps.get("decision_conflict", ps.get("excess_conflict", ps.get("conflict_ratio", 0.0)))
        for ps in prefix_stats.values()
        if ps.get("n_loras", 0) > 1
    ] if prefix_stats else []

    # --- Mode fit score (0-0.4) ---
    if opt_mode == "per_prefix":
        if abs(avg_cos_sim) < ortho_cos and avg_subspace_overlap < 0.35 and effective_conflict < ortho_conf:
            score += 0.40
        elif decision_conflicts:
            if max(decision_conflicts) - min(decision_conflicts) > 0.10:
                score += 0.35
            else:
                score += 0.30
        else:
            score += 0.30
    elif mode == "consensus":
        if avg_cos_sim > consensus_cos and effective_conflict < consensus_conf and avg_subspace_overlap >= 0.35:
            score += 0.4
        elif avg_cos_sim > consensus_cos * 0.6 and effective_conflict < ties_thresh:
            score += 0.25
        else:
            score += 0.05
    elif mode == "slerp":
        if effective_conflict < ties_thresh * 1.2:
            score += 0.35
        elif effective_conflict < ortho_conf * 0.83:
            score += 0.20
        else:
            score += 0.10
    elif mode == "weighted_average":
        if abs(avg_cos_sim) < ortho_cos and avg_subspace_overlap < 0.35 and effective_conflict < ortho_conf:
            score += 0.30
        elif effective_conflict < ortho_conf * 0.67:
            score += 0.20
        else:
            score += 0.10
    elif mode == "ties":
        if effective_conflict > ties_thresh:
            if abs(avg_cos_sim) < ortho_cos and avg_subspace_overlap < 0.35:
                score += 0.15
            else:
                score += 0.35
        elif effective_conflict > ties_thresh * 0.6:
            score += 0.20
        else:
            score += 0.10

    # --- Sparsification fit (0-0.15) ---
    if spars != "disabled":
        conflict_benefit = min(effective_conflict / 0.5, 1.0) * 0.10
        score += conflict_benefit
        density_penalty = abs(density - ideal_density) * 0.05
        score += 0.05 - density_penalty
        if "_conflict" in spars and decision_conflicts and len(decision_conflicts) > 1:
            variance = max(decision_conflicts) - min(decision_conflicts)
            score += variance * 0.05
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
    if quality == "full":
        if is_orthogonal:
            score += min(effective_conflict / 0.3, 1.0) * 0.10
        else:
            score += 0.05 + min(effective_conflict / 0.3, 1.0) * 0.10
    elif quality == "refine":
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

    # --- Optimization mode fit (0-0.10) ---
    # Per-prefix benefit already scored in mode fit section above.
    if opt_mode == "per_prefix":
        score += 0.10
    else:
        score += 0.07

    # --- Strategy set fit (0-0.05, per_prefix only) ---
    strat_set = config.get("strategy_set", "full")
    if opt_mode == "per_prefix":
        if is_orthogonal:
            # Orthogonal LoRAs: strategy_set impact is architecture-dependent
            # (SLERP helps WAN but hurts Z-Image) — stay neutral, let Phase 2 decide
            score += 0.04
        elif effective_conflict < ties_thresh:
            # Low conflict, non-orthogonal: SLERP upgrade can help
            if strat_set == "full":
                score += 0.05
            elif strat_set == "no_slerp":
                score += 0.03
            else:
                score += 0.02
        else:
            # High conflict: strategy_set has less impact (TIES dominates)
            score += 0.03

    return score


def _score_merge_result(model_patches, clip_patches, compute_svd=True,
                        score_device=None, arch_preset=None, lora_svd=False):
    """
    Score an actual merge result by measuring output quality metrics.
    Returns dict with individual metrics and composite score in [0, 1].

    When compute_svd=False, skips the expensive SVD-based effective rank
    computation and scores using norm consistency + sparsity only.
    When score_device is set (e.g. "cuda"), tensors are moved there for
    faster norm/SVD/sparsity computation.
    When arch_preset is provided, uses arch-aware ideal sparsity from
    dare_ideal_density instead of hardcoded 40%.
    """
    norms = []
    importance_values = []
    effective_ranks = []
    sparsities = []
    _svd_tasks = []  # (gram_up [rank,rank], rank_int) — batched after loop

    all_patches = (
        [(False, key, patch) for key, patch in model_patches.items()] +
        [(True, key, patch) for key, patch in clip_patches.items()]
    )
    total = len(all_patches)
    device_label = f", device={score_device}" if score_device else ""
    logging.info(f"[LoRA AutoTuner]   Scoring merge quality ({total} patches"
                 f"{', +SVD' if compute_svd else ''}{device_label})")
    log_interval = max(1, total // 4)  # Log at ~25%, 50%, 75%, 100%
    for patch_idx, (_is_clip, patch_key, patch) in enumerate(all_patches):
        if (patch_idx + 1) % log_interval == 0 or patch_idx + 1 == total:
            logging.info(f"[LoRA AutoTuner]     Scored {patch_idx + 1}/{total} patches")
        if patch is None:
            continue
        target_key = patch_key[0] if isinstance(patch_key, tuple) else patch_key
        if isinstance(patch, tuple) and len(patch) >= 2:
            tensor = patch[1][0] if isinstance(patch[1], tuple) else patch[1]
        elif isinstance(patch, LoKrAdapter):
            weights = patch.weights
            w1, w2, alpha = weights[0], weights[1], weights[2]
            w1_a, w1_b = weights[3], weights[4]
            w2_a, w2_b, t2 = weights[5], weights[6], weights[7]
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
            importance_values.append(fro_norm)
            del w1, w2
            continue
        elif isinstance(patch, LoHaAdapter):
            diff = _LoRAMergeBase._expand_patch_to_diff(patch)
            if score_device is not None:
                diff = diff.to(score_device)
            fro_norm = diff.norm().item()
            norms.append(fro_norm)
            importance_values.append(fro_norm)
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
                importance_values.append(fro_norm)
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
                # Defer effective-rank SVD to batched post-loop computation.
                # gram_up is [rank, rank] — tiny; batch all same-rank grams
                # into a single eigvalsh call instead of 240 individual svdvals.
                if lora_svd and compute_svd and rank > 0 and rank <= 64:
                    _svd_tasks.append((gram_up, rank))
            continue
        else:
            continue

        if tensor is None:
            continue

        t = tensor.float()
        if score_device is not None:
            t = t.to(score_device)

        # Frobenius norm
        fro_norm = t.norm().item()
        norms.append(fro_norm)
        importance_values.append(fro_norm)

        # Effective rank via spectral analysis (optional, expensive)
        if compute_svd and t.dim() == 2 and min(t.shape) > 1:
            try:
                thin_dim = min(t.shape)
                s = _triton_svdvals(t, n_sv=min(thin_dim, 64))
                s_norm = s / (s.sum() + 1e-10)
                entropy = -(s_norm * (s_norm + 1e-10).log()).sum().item()
                eff_rank = min(math.exp(entropy), thin_dim)
                effective_ranks.append(eff_rank)
            except Exception:
                pass

        # Sparsity
        threshold = t.abs().max().item() * 0.01
        if threshold > 0:
            sparsity = (t.abs() < threshold).float().mean().item()
            sparsities.append(sparsity)

    # Batched effective-rank from deferred gram matrices.
    # Group by rank, stack, single eigvalsh per group.
    if _svd_tasks:
        from collections import defaultdict
        _groups = defaultdict(list)
        for _g, _r in _svd_tasks:
            _groups[_g.shape[0]].append((_g, _r))
        for _items in _groups.values():
            try:
                _G = torch.stack([g for g, _ in _items])
                _eigs = torch.linalg.eigvalsh(_G).flip(-1).clamp(min=0)
                _S = _eigs.sqrt()
                for _i, (_, _r) in enumerate(_items):
                    _s = _S[_i]
                    _s_norm = _s / (_s.sum() + 1e-10)
                    _ent = -(_s_norm * (_s_norm + 1e-10).log()).sum().item()
                    effective_ranks.append(min(math.exp(_ent), float(_r)))
            except Exception:
                pass
        del _svd_tasks

    metrics = {}

    if norms:
        metrics["norm_mean"] = sum(norms) / len(norms)
        metrics["norm_std"] = (sum((n - metrics["norm_mean"])**2 for n in norms)
                               / len(norms)) ** 0.5
        metrics["norm_cv"] = metrics["norm_std"] / (metrics["norm_mean"] + 1e-10)
    else:
        metrics["norm_mean"] = 0.0
        metrics["norm_cv"] = 1.0

    if importance_values:
        metrics["importance_mean"] = sum(importance_values) / len(importance_values)
        metrics["importance_std"] = (sum((n - metrics["importance_mean"])**2 for n in importance_values)
                                     / len(importance_values)) ** 0.5
        metrics["importance_cv"] = metrics["importance_std"] / (metrics["importance_mean"] + 1e-10)
    else:
        metrics["importance_mean"] = 0.0
        metrics["importance_cv"] = metrics["norm_cv"]

    if effective_ranks:
        metrics["effective_rank_mean"] = sum(effective_ranks) / len(effective_ranks)
    else:
        metrics["effective_rank_mean"] = 0.0

    if sparsities:
        avg_sparsity = sum(sparsities) / len(sparsities)
        metrics["sparsity_mean"] = avg_sparsity
        # Arch-aware ideal sparsity: derive from dare_ideal_density
        if arch_preset is not None:
            ideal_sparsity = 1.0 - arch_preset.get("dare_ideal_density", 0.7)
        else:
            ideal_sparsity = 0.4  # legacy default
        metrics["sparsity_fit"] = max(0.0, 1.0 - abs(avg_sparsity - ideal_sparsity) * 2.0)
    else:
        metrics["sparsity_mean"] = 0.0
        metrics["sparsity_fit"] = 0.5

    # Total squared energy of merged output (sum of squared norms)
    metrics["norm_energy_sq"] = sum(n ** 2 for n in norms)

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


def _load_python_callable(module_path, callable_name):
    """Load a callable from either a Python file path or importable module name."""
    if not module_path or not callable_name:
        raise ValueError("module_path and callable_name are required")

    if os.path.isfile(module_path):
        module_name = f"lora_optimizer_eval_{hashlib.sha256(module_path.encode()).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load evaluator module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    fn = getattr(module, callable_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Callable '{callable_name}' not found in {module_path}")
    return fn


def _run_autotuner_evaluator(evaluator, model, clip, lora_data, config, analysis_summary):
    """
    Run an optional external evaluator. The callable receives keyword args:
    model, clip, lora_data, config, context. It may return a float score or
    a dict with {'score': float, 'details': ...}.
    """
    if not evaluator:
        return None

    module_path = evaluator.get("module_path")
    callable_name = evaluator.get("callable_name")
    if not module_path or not callable_name:
        return None

    try:
        fn = _load_python_callable(module_path, callable_name)
        result = fn(
            model=model,
            clip=clip,
            lora_data=lora_data,
            config=config,
            context=evaluator.get("context", {}),
            analysis_summary=analysis_summary,
        )
    except Exception as exc:
        logging.warning(f"[LoRA AutoTuner] External evaluator failed: {exc}")
        return {"score": None, "details": {"error": str(exc)}}

    if isinstance(result, dict):
        score = result.get("score")
        details = result.get("details")
    else:
        score = result
        details = None

    try:
        if score is None:
            return {"score": None, "details": details}
        score = float(score)
    except (TypeError, ValueError):
        logging.warning(f"[LoRA AutoTuner] External evaluator returned invalid score: {score!r}")
        return {"score": None, "details": details}

    return {
        "score": max(0.0, min(1.0, score)),
        "details": details,
    }


class LoRAMergeFormula:
    """
    Passthrough node that attaches a merge formula to the LoRA stack.
    The formula defines hierarchical merge order, e.g., "(1+2) + 3"
    merges LoRAs 1 & 2 first, then merges the result with LoRA 3.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "The LoRA stack to apply the merge formula to."
                }),
                "formula": ("STRING", {
                    "default": "",
                    "tooltip": "Merge formula defining hierarchical merge order. "
                               "Numbers reference 1-indexed LoRA positions in the stack. "
                               "Use + to combine and () to group sub-merges. "
                               "Example: '(1+2) + 3' merges LoRAs 1 & 2 first, then blends with 3. "
                               "Optional weights: '(1+2):0.6 + 3:0.4'. "
                               "Leave empty for default flat merge."
                }),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "apply_formula"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Attaches a merge formula to the LoRA stack to control hierarchical merge order"

    def apply_formula(self, lora_stack, formula):
        output = list(lora_stack) if lora_stack else []
        formula = formula.strip()
        if not formula:
            return (output,)

        # Count actual LoRAs (exclude any existing formula metadata)
        n_loras = sum(1 for item in output
                      if not (isinstance(item, dict) and "_merge_formula" in item))

        # Validate formula
        try:
            _parse_merge_formula(formula, n_loras)
        except ValueError as e:
            logging.warning(f"[LoRA Optimizer] Invalid merge formula: {e} — using flat merge")
            return (output,)

        # Remove any existing formula metadata (in case of chaining)
        output = [item for item in output
                  if not (isinstance(item, dict) and "_merge_formula" in item)]
        output.append({"_merge_formula": formula})
        return (output,)


class LoRAOptimizer(_LoRAMergeBase):
    """
    Auto-optimizer that analyzes a LoRA stack (sign conflicts, magnitude
    distributions, overlap) and automatically selects merge modes
    and parameters, then performs the merge.

    Outputs the merged model/clip plus an analysis report explaining
    what was chosen and why.

    Two-pass streaming architecture:
      Pass 1 — Analysis: resolves aliases to target groups, computes diffs
        per group, samples conflict and magnitude statistics, then discards
        diffs immediately. Only lightweight scalars and small sample tensors
        are kept in memory.
      Pass 2 — Merge: recomputes diffs per group and merges with the
        auto-selected strategy. Each group's diffs are freed after merging.
    Peak memory is roughly one target group at a time, but the exact peak
    still depends on layer size, overlap, and enabled quality/compression
    options.

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
                    "tooltip": "Minimum auto-strength scale factor for orthogonal LoRAs. -1 = architecture-aware default (1.0 for motion-heavy video architectures, lower for image models). Set manually to preserve more or less independent LoRA energy."
                }),
                "free_vram_between_passes": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Frees GPU memory between processing steps. Enable if you're running out of VRAM. Barely affects speed."
                }),
                "vram_budget": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of free VRAM to use for storing merged patches. 0 = all CPU (default), 1.0 = use all free VRAM. Reduces RAM usage on GPU systems."
                }),
                "optimization_mode": (["per_prefix", "global", "additive"], {
                    "default": "per_prefix",
                    "tooltip": "How the optimizer decides to combine LoRAs. 'per_prefix' (recommended): automatically picks the best method for each layer. 'global': uses one method everywhere. 'additive': simple weighted addition with no conflict resolution — preserves all weights exactly. Use for edit, distillation, or DPO LoRAs. (Previously: 'additive' was called 'weighted_sum_only'.)"
                }),
                "cache_patches": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": "Keep the merge result in memory so re-running the workflow is instant (no re-merge needed). Disable to save RAM — recommended for large video models like Wan or LTX."
                }),
                "patch_compression": (["smart", "aggressive", "disabled"], {
                    "default": "smart",
                    "tooltip": "Shrink the merged result to use less memory. "
                               "'smart' (recommended): compresses layers where it's lossless, skips layers that already went through rank reduction. "
                               "'aggressive': compresses everything including rank-reduced layers — saves the most memory but slightly lossy. "
                               "'disabled': no compression, uses more RAM. "
                               "(Previously: this setting was called 'compress_patches' with values non_ties/all/disabled.)"
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
                "merge_refinement": (["none", "refine", "full"], {
                    "default": "none",
                    "tooltip": "Optional preprocessing steps applied to weight diffs before merging. "
                               "none: merge as-is, no extra processing. "
                               "refine: adds direction orthogonalization + selfish weight protection "
                               "(TALL-masks) to reduce interference between LoRAs (minimal extra compute). "
                               "full: adds SVD alignment (KnOTS) on top of refine for maximum "
                               "interference reduction (uses more VRAM for SVD decomposition). "
                               "Higher levels help most when LoRAs have high conflict; "
                               "for low-conflict or orthogonal LoRAs, 'none' is usually fine. "
                               "(Previously: this setting was called 'merge_quality' with values standard/enhanced/maximum.)"
                }),
                "strategy_set": (["full", "no_slerp", "basic"], {
                    "default": "full",
                    "tooltip": "Which merge strategies the auto-selector can choose from. "
                               "'full': all strategies available (consensus, SLERP, orthogonal detection). "
                               "'no_slerp': same detection logic but SLERP is excluded (weighted_average stays as-is). "
                               "'basic': only TIES vs weighted_average, no advanced strategy selection. "
                               "(Previously: this setting was called 'behavior_profile' with values v1.2/no_slerp/classic.)"
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
                "decision_smoothing": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Smooth per-group strategy metrics toward each block's average before Pass 2 decisions. 0 disables smoothing; 0.2-0.4 usually removes noisy mode flips without washing out real differences."
                }),
                "smooth_slerp_gate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, uses smoothed cosine (decision_cosine) for SLERP gate instead of raw avg_cos_sim. Can affect SLERP/weighted_average ratio."
                }),
                "tuner_data": ("TUNER_DATA", {
                    "tooltip": "Connect from the LoRA AutoTuner's tuner_data output. Used when settings_source is 'from_autotuner'."
                }),
                "settings_source": (["manual", "from_autotuner", "from_tuner_data"], {
                    "default": "manual",
                    "tooltip": "manual: use widget settings. from_autotuner: passthrough when chained with a live AutoTuner. from_tuner_data: apply the top-ranked config from loaded tuner data (use with Load Tuner Data)."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "TUNER_DATA", "LORA_DATA")
    RETURN_NAMES = ("model", "clip", "analysis_report", "tuner_data", "lora_data")
    FUNCTION = "execute_node"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Auto-analyzes a LoRA stack and selects heuristic merge strategies per weight group. Outputs merged model + analysis report. Best for style/character LoRAs — apply edit, distillation (LCM/Turbo/Hyper), or DPO LoRAs via a standard Load LoRA node instead."

    @staticmethod
    def _compute_cache_key(lora_stack, output_strength, clip_strength_multiplier, auto_strength, optimization_mode="per_prefix", patch_compression="smart", svd_device="gpu", normalize_keys="disabled", sparsification="disabled", sparsification_density=0.7, dare_dampening=0.0, merge_strategy_override="", merge_refinement="none", strategy_set="full", architecture_preset="auto", auto_strength_floor=-1.0, decision_smoothing=0.25, smooth_slerp_gate=False):
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
        h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}|om={optimization_mode}|cp={patch_compression}|sd={svd_device}|nk={normalize_keys}|sp={sparsification}|spd={sparsification_density}|dd={dare_dampening}|mso={merge_strategy_override}|mq={merge_refinement}|bp={strategy_set}|ap={architecture_preset}|asf={auto_strength_floor}|ds={decision_smoothing}|ssg={smooth_slerp_gate}".encode())
        return h.hexdigest()[:16]

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, clip=None,
                   clip_strength_multiplier=1.0, auto_strength="disabled",
                   auto_strength_floor=-1.0,
                   free_vram_between_passes="disabled", vram_budget=0.0,
                   optimization_mode="per_prefix",
                   cache_patches="enabled", patch_compression="smart",
                   svd_device="gpu", normalize_keys="disabled",
                   sparsification="disabled", sparsification_density=0.7,
                   dare_dampening=0.0,
                   merge_strategy_override="", merge_refinement="none",
                   strategy_set="full", architecture_preset="auto",
                   decision_smoothing=0.25, smooth_slerp_gate=False,
                   tuner_data=None, settings_source="manual"):
        base_key = cls._compute_cache_key(lora_stack, output_strength,
                                          clip_strength_multiplier, auto_strength,
                                          optimization_mode, patch_compression,
                                          svd_device, normalize_keys,
                                          sparsification, sparsification_density,
                                          dare_dampening,
                                          merge_strategy_override, merge_refinement,
                                          strategy_set, architecture_preset,
                                          auto_strength_floor,
                                          decision_smoothing, smooth_slerp_gate)
        cache_key = f"{base_key}|mid={id(model)}|ss={settings_source}"
        if settings_source in ("from_autotuner", "from_tuner_data") and tuner_data is not None:
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
            # Virtual LoRAs from sub-merges store pre-computed diffs keyed by target key
            if item.get("_precomputed_diffs"):
                tkey = target_key
                raw = item["lora"].get(tkey)
                if raw is None and isinstance(tkey, tuple):
                    raw = item["lora"].get(tkey[0])
                if raw is not None:
                    if isinstance(raw, torch.Tensor):
                        diff = raw.float()
                    else:
                        diff = self._expand_patch_to_diff(raw)
                    if device is not None and diff.device != device:
                        diff = diff.to(device)
                    try:
                        diff = diff.reshape(target_shape)
                    except RuntimeError:
                        diff = None
                    rank = 1
                else:
                    continue
            elif (lora_info := self._get_lora_key_info(item["lora"], lora_prefix)) is not None:
                mat_up, mat_down, alpha, mid = lora_info
                rank = mat_down.shape[0]
                diff = self._compute_lora_diff(mat_up, mat_down, alpha, mid, target_shape, device=device)
            else:
                # Try LoKr / LoHa formats
                alt = self._get_lokr_diff(item["lora"], lora_prefix, device=device)
                if alt is None:
                    alt = self._get_loha_diff(item["lora"], lora_prefix, device=device)
                if alt is not None:
                    diff, rank, _ = alt
                    try:
                        diff = diff.reshape(target_shape)
                    except RuntimeError:
                        diff = None
                        rank = 0
                else:
                    continue

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
            # Virtual LoRAs from sub-merges store pre-computed diffs keyed by target key
            if item.get("_precomputed_diffs"):
                tkey = target_key
                raw = item["lora"].get(tkey)
                if raw is None and isinstance(tkey, tuple):
                    raw = item["lora"].get(tkey[0])
                if raw is None:
                    continue
                if isinstance(raw, torch.Tensor):
                    diff = raw.float()
                else:
                    diff = self._expand_patch_to_diff(raw)
                if device is not None and diff.device != device:
                    diff = diff.to(device)
                try:
                    diff = diff.reshape(target_shape)
                except RuntimeError:
                    skip_count += 1
                    continue
                if use_gpu and diff.device.type == "cpu":
                    diff = diff.to(device)
                rank = 1  # unknown rank for virtual LoRAs
            elif (lora_info := self._get_lora_key_info(item["lora"], lora_prefix)) is not None:
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
            else:
                # Try LoKr / LoHa formats
                alt = self._get_lokr_diff(
                    item["lora"], lora_prefix,
                    device=device if use_gpu else None, to_cpu=False,
                )
                if alt is None:
                    alt = self._get_loha_diff(
                        item["lora"], lora_prefix,
                        device=device if use_gpu else None, to_cpu=False,
                    )
                if alt is None:
                    continue
                diff, rank, _ = alt
                try:
                    diff = diff.reshape(target_shape)
                except RuntimeError:
                    skip_count += 1
                    continue

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

    @torch.no_grad()
    def _analyze_target_group(self, target_group, active_loras, model, clip, device,
                              clip_strength_multiplier=1.0, merge_refinement="none", n_magnitude_samples=1000):
        """
        Pass 1 analysis for one resolved target group. All aliases that hit the
        same underlying weight are aggregated per LoRA before statistics are
        computed, so mixed-trainer overlaps are analyzed as one merge unit.
        """
        prepared = self._prepare_group_diffs(
            target_group, active_loras, model, clip, device,
            clip_strength_multiplier=clip_strength_multiplier,
            merge_refinement=merge_refinement,
        )
        if prepared is None:
            return None

        diffs = prepared["diffs"]
        eff_strengths = prepared["eff_strengths"]
        rank_sums = prepared["rank_sums"]
        skip_count = prepared["skip_count"]
        target_key = prepared["target_key"]
        is_clip = prepared["is_clip"]
        raw_n = prepared["raw_n_loras"]

        if len(diffs) == 0:
            if skip_count > 0 or raw_n > 0:
                return (
                    target_group["label_prefix"], [], {}, [], (target_key, is_clip),
                    skip_count, raw_n, {}, {}
                )
            return None

        partial_stats = []
        per_lora_norm_sq = {}
        bases = {}
        for i, diff in diffs.items():
            norm_sq = diff.float().square().sum().item()
            per_lora_norm_sq[i] = norm_sq
            display_l2 = math.sqrt(norm_sq) * abs(active_loras[i]["strength"])
            partial_stats.append((i, rank_sums.get(i, 0), display_l2, norm_sq))
            bases[i] = self._compute_subspace_basis(diff, rank_hint=rank_sums.get(i, 1))

        pair_conflicts = {}
        lora_indices = sorted(diffs.keys())
        for ai in range(len(lora_indices)):
            for bi in range(ai + 1, len(lora_indices)):
                i, j = lora_indices[ai], lora_indices[bi]
                diff_i = diffs[i] if eff_strengths[i] >= 0 else -diffs[i]
                diff_j = diffs[j] if eff_strengths[j] >= 0 else -diffs[j]
                pair_conflicts[(i, j)] = self._sample_pair_metrics(
                    diff_i, diff_j, basis_a=bases.get(i), basis_b=bases.get(j),
                    device=device
                )

        magnitude_samples = []
        seed = hash(target_group["label_prefix"]) & 0xFFFFFFFF
        sample_dev = diffs[lora_indices[0]].device
        mag_g = torch.Generator(device=sample_dev).manual_seed(seed)
        for i in lora_indices:
            flat = diffs[i].flatten().abs().float() * abs(eff_strengths[i])
            del diffs[i]
            n = flat.numel()
            if n > n_magnitude_samples:
                indices = torch.randint(0, n, (n_magnitude_samples,),
                                        device=sample_dev, generator=mag_g)
                flat = flat[indices]
            magnitude_samples.append(flat.cpu())

        return (
            target_group["label_prefix"],
            partial_stats,
            pair_conflicts,
            magnitude_samples,
            (target_key, is_clip),
            skip_count,
            raw_n,
            per_lora_norm_sq,
        )

    def _run_group_analysis(self, target_groups, active_loras, model, clip,
                            compute_device, clip_strength_multiplier=1.0,
                            merge_refinement="none",
                            decision_smoothing=0.0, progress_cb=None):
        """
        Shared Pass 1 runner used by both the optimizer and AutoTuner.
        Returns the same lightweight accumulators both call sites need.
        """
        use_gpu = compute_device.type != "cpu"
        per_lora_stats = [{
            "name": item["name"],
            "strength": item["strength"],
            "ranks": [],
            "key_count": 0,
            "l2_norms": [],
        } for item in active_loras]

        pairs = [(i, j) for i in range(len(active_loras))
                         for j in range(i + 1, len(active_loras))]
        branch_energy = {
            "model": {
                "norm_sq": [0.0] * len(active_loras),
                "dot": {(i, j): 0.0 for i, j in pairs},
                "importance": [0.0] * len(active_loras),
            },
            "clip": {
                "norm_sq": [0.0] * len(active_loras),
                "dot": {(i, j): 0.0 for i, j in pairs},
                "importance": [0.0] * len(active_loras),
            },
        }
        pair_accum = {
            (i, j): {
                "overlap": 0,
                "conflict": 0,
                "dot": 0.0,
                "norm_a_sq": 0.0,
                "norm_b_sq": 0.0,
                "weighted_total": 0.0,
                "weighted_conflict": 0.0,
                "expected_conflict_weighted": 0.0,
                "excess_conflict_weighted": 0.0,
                "subspace_num": 0.0,
                "subspace_den": 0.0,
            } for i, j in pairs
        }
        all_magnitude_samples = []
        all_key_targets = {}
        prefix_stats = {}
        skipped_keys = 0
        prefix_count = 0

        def _collect_analysis_result(result):
            nonlocal skipped_keys, prefix_count
            if result is None:
                return
            (prefix, partial_stats, pair_conflicts, mag_samples, target_info, skips,
             raw_n, per_lora_norm_sq) = result
            is_clip = target_info[1]
            branch_name = "clip" if is_clip else "model"
            skipped_keys += skips
            if len(partial_stats) > 0:
                all_key_targets[prefix] = target_info
                prefix_count += 1
            for (idx, rank, l2, norm_sq) in partial_stats:
                per_lora_stats[idx]["ranks"].append(rank)
                per_lora_stats[idx]["key_count"] += 1
                per_lora_stats[idx]["l2_norms"].append(l2)
                branch_energy[branch_name]["norm_sq"][idx] += norm_sq
            for (i, j), metrics in pair_conflicts.items():
                acc = pair_accum[(i, j)]
                acc["overlap"] += metrics["overlap"]
                acc["conflict"] += metrics["conflict"]
                acc["dot"] += metrics["dot"]
                acc["norm_a_sq"] += metrics["norm_a_sq"]
                acc["norm_b_sq"] += metrics["norm_b_sq"]
                acc["weighted_total"] += metrics["weighted_total"]
                acc["weighted_conflict"] += metrics["weighted_conflict"]
                acc["expected_conflict_weighted"] += metrics["expected_conflict"] * metrics["weighted_total"]
                acc["excess_conflict_weighted"] += metrics["excess_conflict"] * metrics["weighted_total"]
                acc["subspace_num"] += metrics["subspace_overlap"] * metrics["subspace_weight"]
                acc["subspace_den"] += metrics["subspace_weight"]
                branch_energy[branch_name]["dot"][(i, j)] += metrics["dot"]
            all_magnitude_samples.extend(mag_samples)

            if len(partial_stats) > 0:
                pf_overlap = sum(m["overlap"] for m in pair_conflicts.values())
                pf_conflict = sum(m["conflict"] for m in pair_conflicts.values())
                pf_conflict_ratio = pf_conflict / pf_overlap if pf_overlap > 0 else 0.0
                pf_weighted_total = sum(m["weighted_total"] for m in pair_conflicts.values())
                pf_weighted_conflict = sum(m["weighted_conflict"] for m in pair_conflicts.values())
                pf_weighted_ratio = (pf_weighted_conflict / pf_weighted_total) if pf_weighted_total > 0 else pf_conflict_ratio
                pf_expected_conflict = (
                    sum(m["expected_conflict"] * m["weighted_total"] for m in pair_conflicts.values()) / pf_weighted_total
                ) if pf_weighted_total > 0 else 0.0
                pf_excess_conflict = (
                    sum(m["excess_conflict"] * m["weighted_total"] for m in pair_conflicts.values()) / pf_weighted_total
                ) if pf_weighted_total > 0 else 0.0

                pf_l2s = [math.sqrt(v) for v in per_lora_norm_sq.values() if v > 0]
                pf_mag_ratio = (max(pf_l2s) / min(pf_l2s)) if len(pf_l2s) >= 2 else 1.0
                pf_activation_ratio = pf_mag_ratio

                pf_cos_sims = []
                for metrics in pair_conflicts.values():
                    denom = (metrics["norm_a_sq"] ** 0.5) * (metrics["norm_b_sq"] ** 0.5)
                    if denom > 0:
                        pf_cos_sims.append(metrics["dot"] / denom)
                avg_cos_sim = sum(pf_cos_sims) / len(pf_cos_sims) if pf_cos_sims else 0.0
                pf_subspace_den = sum(m["subspace_weight"] for m in pair_conflicts.values())
                avg_subspace_overlap = (
                    sum(m["subspace_overlap"] * m["subspace_weight"] for m in pair_conflicts.values()) / pf_subspace_den
                ) if pf_subspace_den > 0 else 0.0

                prefix_stats[prefix] = {
                    "n_loras": len(partial_stats),
                    "raw_n_loras": raw_n,
                    "conflict_ratio": pf_conflict_ratio,
                    "weighted_conflict_ratio": pf_weighted_ratio,
                    "expected_conflict": pf_expected_conflict,
                    "excess_conflict": pf_excess_conflict,
                    "magnitude_ratio": pf_mag_ratio,
                    "activation_ratio": pf_activation_ratio,
                    "magnitude_samples": list(mag_samples),
                    "avg_cos_sim": avg_cos_sim,
                    "avg_subspace_overlap": avg_subspace_overlap,
                    "per_lora_norm_sq": dict(per_lora_norm_sq),
                    "pairwise_dots": {
                        pair: vals["dot"] for pair, vals in pair_conflicts.items()
                    },
                }

        group_items = list(target_groups.values())
        if use_gpu:
            for target_group in group_items:
                result = self._analyze_target_group(
                    target_group, active_loras, model, clip, compute_device,
                    clip_strength_multiplier=clip_strength_multiplier,
                    merge_refinement=merge_refinement,
                )
                _collect_analysis_result(result)
                if progress_cb is not None:
                    progress_cb()
        else:
            max_workers = min(4, max(1, len(group_items)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._analyze_target_group, target_group, active_loras,
                        model, clip, compute_device, clip_strength_multiplier,
                        merge_refinement
                    ): target_group["label_prefix"]
                    for target_group in group_items
                }
                for future in concurrent.futures.as_completed(futures):
                    _collect_analysis_result(future.result())
                    if progress_cb is not None:
                        progress_cb()

        prefix_stats = self._apply_block_smoothing(prefix_stats, strength=decision_smoothing)

        return {
            "all_key_targets": all_key_targets,
            "target_groups": dict(target_groups),
            "prefix_stats": prefix_stats,
            "per_lora_stats": per_lora_stats,
            "pair_accum": pair_accum,
            "branch_energy": branch_energy,
            "all_magnitude_samples": all_magnitude_samples,
            "prefix_count": prefix_count,
            "skipped_keys": skipped_keys,
            "pairs": pairs,
        }

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

    def _compute_branch_auto_scale(self, branch_name, strengths, norm_sq, dot_accum,
                                   arch_preset=None, detected_arch=None,
                                   auto_strength_floor=-1.0, is_full_rank=False):
        """Exact streamed auto-strength scale for one branch."""
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
        n = len(strengths)
        effective = [abs(strengths[i]) * math.sqrt(max(norm_sq[i], 0.0)) for i in range(n)]
        nonzero = [e for e in effective if e > 0]
        reasoning = []
        if len(nonzero) <= 1:
            reasoning.append(f"{branch_name}: single contributing branch or none — no adjustment needed")
            return {
                "scale": 1.0,
                "new_strengths": list(strengths),
                "reasoning": reasoning,
            }

        energy_sq = 0.0
        orthogonal_energy_sq = 0.0
        pairwise_cos = []
        for i in range(n):
            energy_sq += (strengths[i] ** 2) * norm_sq[i]
            orthogonal_energy_sq += (strengths[i] ** 2) * norm_sq[i]
        for (i, j), dot in dot_accum.items():
            if norm_sq[i] <= 0 or norm_sq[j] <= 0:
                continue
            energy_sq += 2.0 * strengths[i] * strengths[j] * dot
            denom = math.sqrt(norm_sq[i]) * math.sqrt(norm_sq[j])
            if denom > 0:
                pairwise_cos.append(dot / denom)

        energy_sq = max(energy_sq, 0.0)
        current_energy = math.sqrt(energy_sq)
        orthogonal_energy = math.sqrt(max(orthogonal_energy_sq, 0.0))
        reference_energy = max(effective)
        scale = min(reference_energy / current_energy, 1.0) if current_energy > 0 else 1.0

        floor_applied = False
        floor = None
        if pairwise_cos:
            avg_cos = sum(pairwise_cos) / len(pairwise_cos)
            alignment_thresh = arch_preset["alignment_threshold"]
            if abs(avg_cos) <= alignment_thresh:
                if auto_strength_floor >= 0:
                    floor = auto_strength_floor
                elif is_full_rank:
                    floor = arch_preset.get("full_rank", {}).get("auto_strength_floor", 1.0)
                else:
                    floor = _VIDEO_ARCH_ORTHOGONAL_FLOOR.get(
                        detected_arch,
                        arch_preset.get("auto_strength_orthogonal_floor", 0.85),
                    )
                if scale < floor:
                    scale = floor
                    floor_applied = True

        new_strengths = [s * scale if effective[i] > 0 else s for i, s in enumerate(strengths)]

        reasoning.append(f"{branch_name}: scale factor {scale:.4f}")
        if pairwise_cos:
            avg_cos = sum(pairwise_cos) / len(pairwise_cos)
            alignment_thresh = arch_preset["alignment_threshold"]
            if avg_cos > alignment_thresh:
                alignment_desc = "mostly aligned (reinforcing)"
            elif avg_cos < -alignment_thresh:
                alignment_desc = "mostly opposing (cancelling)"
            else:
                alignment_desc = "mostly orthogonal (independent)"
            if floor_applied:
                arch_label = detected_arch or "unknown"
                if is_full_rank:
                    reasoning.append(
                        f"{branch_name}: full-rank orthogonal floor {floor:.2f} applied "
                        f"to preserve complete weight deltas"
                    )
                else:
                    reasoning.append(
                        f"{branch_name}: orthogonal floor {floor:.2f} applied for {arch_label} "
                        f"to preserve independent contributions"
                    )
            reasoning.append(
                f"{branch_name}: exact streamed energy {current_energy:.4f} "
                f"(orthogonal baseline {orthogonal_energy:.4f}, avg cos {avg_cos:.3f} — {alignment_desc})"
            )
        return {
            "scale": scale,
            "new_strengths": new_strengths,
            "reasoning": reasoning,
        }

    def _compute_auto_strengths(self, active_loras, branch_energy,
                                clip_strength_multiplier=1.0, arch_preset=None,
                                detected_arch=None, auto_strength_floor=-1.0,
                                is_full_rank=False):
        """
        Compute exact streamed auto-strength scaling separately for model and
        CLIP branches, using accumulated Frobenius norms and pairwise dots.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]

        model_strengths = [item["strength"] for item in active_loras]
        clip_strengths = [
            item["clip_strength"] if item["clip_strength"] is not None else item["strength"]
            for item in active_loras
        ]

        model_info = self._compute_branch_auto_scale(
            "Model",
            model_strengths,
            branch_energy["model"]["norm_sq"],
            branch_energy["model"]["dot"],
            arch_preset=arch_preset,
            detected_arch=detected_arch,
            auto_strength_floor=auto_strength_floor,
            is_full_rank=is_full_rank,
        )
        clip_info = self._compute_branch_auto_scale(
            "CLIP",
            clip_strengths,
            branch_energy["clip"]["norm_sq"],
            branch_energy["clip"]["dot"],
            arch_preset=arch_preset,
            detected_arch=detected_arch,
            auto_strength_floor=auto_strength_floor,
            is_full_rank=is_full_rank,
        )

        reasoning = []
        reasoning.extend(model_info["reasoning"])
        if any(v > 0 for v in branch_energy["clip"]["norm_sq"]):
            reasoning.extend(clip_info["reasoning"])

        return {
            "model_scale": model_info["scale"],
            "clip_scale": clip_info["scale"],
            "model_strengths": model_info["new_strengths"],
            "clip_strengths": clip_info["new_strengths"],
            "original_model_strengths": model_strengths,
            "original_clip_strengths": clip_strengths,
            "names": [item["name"] for item in active_loras],
            "clip_uses_global_multiplier": [
                item["clip_strength"] is None for item in active_loras
            ],
            "clip_strength_multiplier": clip_strength_multiplier,
            "reasoning": reasoning,
        }

    def _build_exact_linear_patch(self, target_group, active_loras, raw_n_loras,
                                  mode, is_clip_key=False, model_scale=1.0):
        """
        Build an exact low-rank patch for linear merges by concatenating factors
        instead of materializing a dense diff. Falls back to None when the group
        contains unsupported parameterizations (for example LoCon mid matrices).
        """
        if mode not in ("weighted_sum", "weighted_average", "normalize"):
            return None

        pieces = []
        lora_weights = {}
        has_conflict_modes = False

        for i, item in enumerate(active_loras):
            kf = item.get("key_filter", "all")
            if kf == "shared_only" and raw_n_loras < 2:
                continue
            if kf == "unique_only" and raw_n_loras != 1:
                continue
            if item.get("conflict_mode", "all") != "all":
                has_conflict_modes = True
                break

            if is_clip_key:
                base_weight = item["clip_strength"] if item["clip_strength"] is not None else item["strength"]
            else:
                base_weight = item["strength"] * model_scale

            contributed = False
            for alias in target_group["aliases"]:
                lora_info = self._get_lora_key_info(item["lora"], alias)
                if lora_info is None:
                    # Check if this alias has LoKr/LoHa keys — can't represent
                    # as low-rank factors, fall through to dense diff path
                    if self._has_lokr_keys(item["lora"], alias) or self._has_loha_keys(item["lora"], alias):
                        return None
                    continue
                mat_up, mat_down, alpha, mid = lora_info
                if mid is not None:
                    return None
                pieces.append((i, mat_up, mat_down, alpha))
                contributed = True

            if contributed:
                lora_weights[i] = base_weight

        if has_conflict_modes or not pieces:
            return None

        if mode == "weighted_average":
            total_weight = sum(abs(w) for w in lora_weights.values())
            if total_weight == 0:
                return None
            per_lora_scales = {idx: w / total_weight for idx, w in lora_weights.items()}
        elif mode == "normalize":
            denom = math.sqrt(sum(w * w for w in lora_weights.values()))
            if denom == 0:
                return None
            per_lora_scales = {idx: w / denom for idx, w in lora_weights.items()}
        else:
            per_lora_scales = dict(lora_weights)

        up_parts = []
        down_parts = []
        total_rank = 0
        for lora_idx, mat_up, mat_down, alpha in pieces:
            weight = per_lora_scales[lora_idx]
            rank = mat_down.shape[0]
            total_rank += rank
            piece_scale = weight * (alpha / rank)
            up_parts.append(mat_up * piece_scale)
            down_parts.append(mat_down)

        if total_rank <= 0:
            return None

        fused_up = torch.cat(up_parts, dim=1)
        fused_down = torch.cat(down_parts, dim=0)
        return {
            "patch": LoRAAdapter(set(), (fused_up, fused_down, float(total_rank), None, None, None)),
            "weights": per_lora_scales,
        }

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

    def _apply_block_smoothing(self, prefix_stats, strength=0.0):
        """
        Smooth noisy per-group decision metrics toward the average of their
        surrounding logical block. Raw metrics are preserved; decision_* fields
        are what Pass 2 should consume.
        """
        if not prefix_stats:
            return prefix_stats

        strength = max(0.0, min(1.0, float(strength or 0.0)))
        block_groups = {}
        for prefix, stat in prefix_stats.items():
            block_name = self._extract_block_name(prefix)
            stat["block_name"] = block_name
            block_groups.setdefault(block_name, []).append((prefix, stat))

        metric_keys = (
            "conflict_ratio",
            "weighted_conflict_ratio",
            "expected_conflict",
            "excess_conflict",
            "avg_cos_sim",
            "avg_subspace_overlap",
            "magnitude_ratio",
            "activation_ratio",
        )

        for entries in block_groups.values():
            if not entries:
                continue

            weights = []
            for _prefix, stat in entries:
                weight = sum(stat.get("per_lora_norm_sq", {}).values())
                weights.append(weight if weight > 0 else 1.0)

            total_weight = sum(weights) if weights else 0.0
            block_means = {}
            for key in metric_keys:
                values = [stat.get(key) for _prefix, stat in entries if stat.get(key) is not None]
                if not values:
                    continue
                weighted_sum = 0.0
                for weight, (_prefix, stat) in zip(weights, entries):
                    weighted_sum += stat.get(key, 0.0) * weight
                block_means[key] = weighted_sum / total_weight if total_weight > 0 else sum(values) / len(values)

            for prefix, stat in entries:
                stat["block_size"] = len(entries)
                stat["block_smoothing_strength"] = strength
                for key, mean_value in block_means.items():
                    raw_value = stat.get(key, mean_value)
                    smoothed = raw_value if strength <= 0 or len(entries) == 1 else ((1.0 - strength) * raw_value + strength * mean_value)
                    stat[f"smoothed_{key}"] = smoothed
                stat["decision_conflict"] = stat.get("smoothed_excess_conflict", stat.get("excess_conflict", stat.get("conflict_ratio", 0.0)))
                stat["decision_cosine"] = stat.get("smoothed_avg_cos_sim", stat.get("avg_cos_sim", 0.0))
                stat["decision_subspace_overlap"] = stat.get("smoothed_avg_subspace_overlap", stat.get("avg_subspace_overlap", 0.0))
                stat["decision_magnitude_ratio"] = stat.get("smoothed_activation_ratio", stat.get("activation_ratio", stat.get("magnitude_ratio", 1.0)))
        return prefix_stats

    def _auto_select_params(self, avg_conflict_ratio, magnitude_ratio, all_key_diffs=None,
                            magnitude_samples=None, avg_cos_sim=0.0,
                            avg_excess_conflict=None, avg_subspace_overlap=0.0,
                            strategy_set="full", arch_preset=None,
                            precomputed_density=None):
        """
        Decision logic for auto-selecting merge parameters.
        Returns (mode, density, sign_method, reasoning_lines).

        Density can be estimated from either all_key_diffs (legacy bulk path)
        or magnitude_samples (streaming path). Thresholds come from arch_preset.
        """
        if arch_preset is None:
            arch_preset = _ARCH_PRESETS["sd_unet"]
        reasoning = []

        effective_conflict = avg_conflict_ratio
        if avg_excess_conflict is not None:
            effective_conflict = max(avg_excess_conflict, 0.0)
            if avg_subspace_overlap > 0:
                effective_conflict *= (0.5 + 0.5 * avg_subspace_overlap)

        # High similarity + low conflict → consensus mode (Fisher-proxy + magnitude calibration)
        if (strategy_set == "full"
                and avg_cos_sim > arch_preset["consensus_cos_sim_min"]
                and effective_conflict < arch_preset["consensus_conflict_max"]
                and avg_subspace_overlap >= 0.35):
            mode = "consensus"
            reasoning.append(f"Cosine similarity {avg_cos_sim:.2f} > {arch_preset['consensus_cos_sim_min']} "
                             f"and excess conflict {effective_conflict:.1%} < {arch_preset['consensus_conflict_max']:.0%} -> consensus mode")
            reasoning.append("  Fisher-proxy importance weighting + magnitude calibration + spectral cleanup")
            density = 0.5  # unused
            sign_method = "frequency"  # unused
            return (mode, density, sign_method, reasoning)

        # Near-orthogonal LoRAs: ~50% sign conflict is the base rate for
        # independent vectors, not actual semantic conflict. TIES trimming
        # destroys both signals. Use weighted_average as the global mode,
        # upgraded to SLERP per-prefix to preserve magnitude (important for
        # video LoRAs where motion energy matters).
        if (strategy_set in ("full", "no_slerp")
                and abs(avg_cos_sim) < arch_preset["orthogonal_cos_sim_max"]
                and effective_conflict < arch_preset["orthogonal_conflict_max"]
                and avg_subspace_overlap < 0.35):
            mode = "weighted_average"
            reasoning.append(f"Cosine similarity {avg_cos_sim:.2f} near zero (orthogonal LoRAs) — "
                             f"sign conflict {avg_conflict_ratio:.1%} is base-rate noise, not real conflict")
            if strategy_set == "full":
                reasoning.append("  Using weighted_average (upgraded to SLERP per-prefix to preserve magnitude)")
            else:
                reasoning.append("  Using weighted_average to preserve both signals (SLERP upgrade disabled by profile)")
            density = 0.5
            sign_method = "frequency"
            return (mode, density, sign_method, reasoning)

        # Select mode based on sign conflict
        if effective_conflict > arch_preset["ties_conflict_threshold"]:
            mode = "ties"
            reasoning.append(f"Excess conflict {effective_conflict:.1%} > {arch_preset['ties_conflict_threshold']:.0%} threshold -> TIES mode selected")
            if avg_subspace_overlap > 0:
                reasoning.append(f"  Subspace overlap {avg_subspace_overlap:.2f} suggests the conflicting LoRAs target similar directions")
            reasoning.append("  TIES resolves sign conflicts via trim + elect sign + disjoint merge")
        else:
            mode = "weighted_average"
            reasoning.append(f"Excess conflict {effective_conflict:.1%} <= {arch_preset['ties_conflict_threshold']:.0%} threshold -> weighted_average mode selected")
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
                      merge_refinement="none",
                      compatibility_warnings=None,
                      strategy_set="full",
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
                'acestep': 'ACE-Step',
                'sdxl': 'SDXL',
                'ltx': 'LTX Video',
                'qwen_image': 'Qwen-Image',
            }
            lines.append(f"Architecture: {arch_names.get(detected_arch, detected_arch)} (auto-detected)")
            lines.append(f"Key normalization: enabled")
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
                orig = auto_strength_info["original_model_strengths"][i]
                new = auto_strength_info["model_strengths"][i]
                line = f"  {name}: model {orig} -> {new:.4f}"
                if i < len(auto_strength_info.get("original_clip_strengths", [])):
                    clip_orig = auto_strength_info["original_clip_strengths"][i]
                    clip_new = auto_strength_info["clip_strengths"][i]
                    line += f", clip {clip_orig} -> {clip_new:.4f}"
                    if auto_strength_info.get("clip_uses_global_multiplier", [False] * len(auto_strength_info["names"]))[i]:
                        line += " (pre-global multiplier)"
                lines.append(line)
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
                if 'weighted_ratio' in pc:
                    lines.append(f"    Magnitude-weighted conflict: {pc['weighted_ratio']:.1%}")
                if 'expected_conflict' in pc:
                    lines.append(f"    Excess conflict over cosine baseline: {pc.get('excess_conflict', 0.0):.1%} (expected {pc['expected_conflict']:.1%})")
                if 'cosine_sim' in pc:
                    lines.append(f"    Cosine similarity: {pc['cosine_sim']:.3f}")
                if 'subspace_overlap' in pc:
                    lines.append(f"    Subspace overlap: {pc['subspace_overlap']:.2f}")

        # Collection Statistics
        lines.append("")
        lines.append("--- Collection Statistics ---")
        lines.append(f"  Total LoRAs: {collection_stats['n_loras']}")
        lines.append(f"  Total target groups: {collection_stats['total_keys']}")
        lines.append(f"  Avg sign conflict ratio: {collection_stats['avg_conflict']:.1%}")
        if "avg_weighted_conflict" in collection_stats:
            lines.append(f"  Avg weighted conflict ratio: {collection_stats['avg_weighted_conflict']:.1%}")
        if "avg_excess_conflict" in collection_stats:
            lines.append(f"  Avg excess conflict: {collection_stats['avg_excess_conflict']:.1%}")
        if "avg_subspace_overlap" in collection_stats:
            lines.append(f"  Avg subspace overlap: {collection_stats['avg_subspace_overlap']:.2f}")
        lines.append(f"  Importance ratio (max/min frobenius): {collection_stats['magnitude_ratio']:.2f}x")
        if collection_stats.get("decision_smoothing", 0.0) > 0:
            lines.append(f"  Decision smoothing: {collection_stats['decision_smoothing']:.2f}")

        # Auto-Selected Parameters
        lines.append("")
        lines.append("--- Auto-Selected Parameters ---")
        if optimization_mode == "additive":
            lines.append(f"  Merge mode: weighted_sum (forced by additive)")
            lines.append(f"  Auto-detected mode: {mode} (overridden)")
        else:
            lines.append(f"  Merge mode: {mode}")
        if mode == "ties":
            lines.append(f"  Density: {density:.2f}")
            lines.append(f"  Sign method: {sign_method}")
        if optimization_mode == "per_prefix":
            lines.append("  (global fallback — each target group uses its own parameters)")
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
            elif optimization_mode == "additive":
                lines.append(f"  Applied as preprocessing before weighted_sum")
            elif mode == "ties":
                lines.append(f"  Note: {display_name} replaces TIES trim step")
            else:
                lines.append(f"  Applied as preprocessing before {mode}")

        if merge_refinement != "none":
            quality_desc = {
                "refine": "Refine (orthogonalize + TALL-masks)",
                "full": "Full (orthogonalize + KnOTS SVD alignment + TALL-masks)",
            }
            lines.append(f"  Merge refinement: {quality_desc.get(merge_refinement, merge_refinement)}")

        if strategy_set != "full":
            profile_desc = {
                "no_slerp": "no_slerp (full detection, no SLERP upgrade)",
                "basic": "basic (TIES vs weighted_average only)",
            }
            lines.append(f"  Strategy set: {profile_desc.get(strategy_set, strategy_set)}")

        # Per-Prefix Strategy breakdown (only in per_prefix mode)
        if optimization_mode == "per_prefix" and strategy_counts:
            lines.append("")
            lines.append("--- Per-Group Strategy ---")
            total_pf = sum(strategy_counts.values())
            if total_pf > 0:
                if strategy_counts.get("weighted_sum", 0) > 0:
                    n = strategy_counts["weighted_sum"]
                    lines.append(f"  weighted_sum (single LoRA):      {n:>4} groups ({n/total_pf:.0%})")
                if strategy_counts.get("slerp", 0) > 0:
                    n = strategy_counts["slerp"]
                    lines.append(f"  slerp (low conflict):            {n:>4} groups ({n/total_pf:.0%})")
                if strategy_counts.get("weighted_average", 0) > 0:
                    n = strategy_counts["weighted_average"]
                    lines.append(f"  weighted_average (orthogonal):    {n:>4} groups ({n/total_pf:.0%})")
                if strategy_counts.get("consensus", 0) > 0:
                    n = strategy_counts["consensus"]
                    lines.append(f"  consensus (high similarity):     {n:>4} groups ({n/total_pf:.0%})")
                if strategy_counts.get("ties", 0) > 0:
                    n = strategy_counts["ties"]
                    lines.append(f"  ties (high conflict):            {n:>4} groups ({n/total_pf:.0%})")
                lines.append(f"  Total:                           {total_pf:>4} groups")

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
                     patch_compression="smart", svd_device="gpu",
                     normalize_keys="enabled", sparsification="disabled",
                     sparsification_density=0.7, dare_dampening=0.0,
                     merge_strategy_override="", merge_refinement="none",
                     strategy_set="full", architecture_preset="auto",
                     decision_smoothing=0.25, smooth_slerp_gate=False,
                     tuner_data=None, settings_source="manual"):
        """
        ComfyUI entry point. Supports an AutoTuner bridge mode:
        - from_autotuner: passthrough, using the upstream AutoTuner merge and
          syncing the optimizer widgets from tuner_data for manual refinement.
        - manual: run the optimizer normally using the current widget values.
        """
        if settings_source == "from_autotuner":
            if tuner_data is None or "top_n" not in tuner_data or len(tuner_data["top_n"]) == 0:
                logging.warning("[AutoTuner Bridge] No valid tuner_data — falling back to manual merge")
            else:
                entry = tuner_data["top_n"][0]
                config = entry["config"]
                applied_config = dict(config)
                if "auto_strength_floor" in tuner_data:
                    applied_config["auto_strength_floor"] = tuner_data["auto_strength_floor"]
                score = entry.get("score_final", entry.get("score_measured", entry.get("score_heuristic", 0.0)))
                mode_display = ("per-prefix (auto)" if config["optimization_mode"] == "per_prefix"
                                else config.get("merge_mode", "unknown"))

                report_lines = [
                    f"[AutoTuner Bridge] Passthrough — model merged by AutoTuner (rank #1, score: {score:.3f}):",
                    f"  optimization_mode: {config['optimization_mode']}",
                    f"  merge mode: {mode_display}",
                    f"  merge_refinement: {config['merge_refinement']}",
                ]
                if config["sparsification"] != "disabled":
                    report_lines.append(
                        f"  sparsification: {config['sparsification']} "
                        f"(density: {config['sparsification_density']}, "
                        f"dampening: {config['dare_dampening']})"
                    )
                report_lines.append(f"  auto_strength: {config['auto_strength']}")
                if "auto_strength_floor" in tuner_data:
                    report_lines.append(f"  auto_strength_floor: {tuner_data['auto_strength_floor']}")
                if "decision_smoothing" in tuner_data:
                    report_lines.append(f"  decision_smoothing: {tuner_data['decision_smoothing']}")
                report_lines.append("")
                report_lines.append("Switch settings_source to 'manual' to tweak from these settings.")
                report = "\n".join(report_lines)

                logging.info("[AutoTuner Bridge] Passthrough mode — model already merged by AutoTuner")
                return {"result": (model, clip, report, tuner_data, None),
                        "ui": {"applied_settings": [json.dumps(applied_config)]}}

        if settings_source == "from_tuner_data":
            if tuner_data is None or "top_n" not in tuner_data or len(tuner_data["top_n"]) == 0:
                logging.warning("[Tuner Data] No valid tuner_data — falling back to manual merge")
            else:
                entry = tuner_data["top_n"][0]
                config = entry["config"]
                strategy_override = config["merge_mode"] if config["optimization_mode"] == "global" else ""
                resolved_smoothing = tuner_data.get("decision_smoothing", decision_smoothing)
                result = self.optimize_merge(
                    model, lora_stack, output_strength,
                    clip=clip,
                    clip_strength_multiplier=clip_strength_multiplier,
                    auto_strength=config["auto_strength"],
                    auto_strength_floor=tuner_data.get("auto_strength_floor", auto_strength_floor),
                    free_vram_between_passes=free_vram_between_passes,
                    vram_budget=vram_budget,
                    optimization_mode=config["optimization_mode"],
                    cache_patches=cache_patches,
                    patch_compression=patch_compression,
                    svd_device=svd_device,
                    normalize_keys=tuner_data.get("normalize_keys", normalize_keys),
                    sparsification=config["sparsification"],
                    sparsification_density=config["sparsification_density"],
                    dare_dampening=config["dare_dampening"],
                    merge_strategy_override=strategy_override,
                    merge_refinement=config["merge_refinement"],
                    strategy_set=tuner_data.get("strategy_set", strategy_set),
                    architecture_preset=tuner_data.get("architecture_preset", architecture_preset),
                    decision_smoothing=resolved_smoothing,
                    smooth_slerp_gate=smooth_slerp_gate,
                )
                applied_config = dict(config)
                return {"result": result, "ui": {"applied_settings": [json.dumps(applied_config)]}}

        return self.optimize_merge(
            model, lora_stack, output_strength,
            clip=clip,
            clip_strength_multiplier=clip_strength_multiplier,
            auto_strength=auto_strength,
            auto_strength_floor=auto_strength_floor,
            free_vram_between_passes=free_vram_between_passes,
            vram_budget=vram_budget,
            optimization_mode=optimization_mode,
            cache_patches=cache_patches,
            patch_compression=patch_compression,
            svd_device=svd_device,
            normalize_keys=normalize_keys,
            sparsification=sparsification,
            sparsification_density=sparsification_density,
            dare_dampening=dare_dampening,
            merge_strategy_override=merge_strategy_override,
            merge_refinement=merge_refinement,
            strategy_set=strategy_set,
            architecture_preset=architecture_preset,
            decision_smoothing=decision_smoothing,
            smooth_slerp_gate=smooth_slerp_gate,
        )

    def optimize_merge(self, model, lora_stack, output_strength, clip=None, clip_strength_multiplier=1.0, auto_strength="disabled", auto_strength_floor=-1.0, free_vram_between_passes="disabled", vram_budget=0.0, optimization_mode="per_prefix", cache_patches="enabled", patch_compression="smart", svd_device="gpu", normalize_keys="disabled", sparsification="disabled", sparsification_density=0.7, dare_dampening=0.0, merge_strategy_override="", merge_refinement="none", strategy_set="full", architecture_preset="auto", decision_smoothing=0.25, smooth_slerp_gate=False, _analysis_cache=None, _diff_cache=None, _skip_report=False, _skip_qkv_refusion=False):
        """
        Main entry point. Two-pass streaming architecture:
        Pass 1: Resolve aliases to target groups, compute diffs, sample metrics, discard diffs
        Decision: Finalize stats, auto-select params from lightweight accumulators
        Pass 2: Recompute diffs per target group, merge immediately, discard
        Peak memory tracks the largest active target group, not the whole stack.
        """
        # Free stale cached models when the input model changes — prevents
        # the old patched model from staying in RAM after switching models.
        current_mid = id(model) if model is not None else None
        prev_mid = getattr(self, '_cached_model_id', None)
        if prev_mid is not None and current_mid != prev_mid:
            if hasattr(self, '_merge_cache') and self._merge_cache:
                self._merge_cache.clear()
            if hasattr(self, '_autotuner_cache') and getattr(self, '_autotuner_cache', None):
                self._autotuner_cache.clear()
            delegate = getattr(self, '_autotuner_delegate', None)
            if delegate is not None:
                if hasattr(delegate, '_merge_cache') and delegate._merge_cache:
                    delegate._merge_cache.clear()
                if hasattr(delegate, '_autotuner_cache') and getattr(delegate, '_autotuner_cache', None):
                    delegate._autotuner_cache.clear()
                delegate._cached_model_id = None
            gc.collect()
        self._cached_model_id = current_mid

        # Normalize stack format (standard tuples or LoRAStack dicts)
        if not lora_stack or len(lora_stack) == 0:
            return (model, clip, "No LoRAs in stack.", None, None)

        # Extract merge formula metadata before normalization
        merge_formula = None
        clean_stack = []
        for item in lora_stack:
            if isinstance(item, dict) and "_merge_formula" in item:
                merge_formula = item["_merge_formula"]
            else:
                clean_stack.append(item)
        if merge_formula:
            lora_stack = clean_stack

        normalized_stack = self._normalize_stack(lora_stack, normalize_keys=normalize_keys)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]

        if len(active_loras) == 0:
            return (model, clip, "No LoRAs in stack (all zero strength or malformed).", None, None)

        # Resolve architecture preset from override or auto-detection
        preset_key, arch_preset = _resolve_arch_preset(
            architecture_preset, getattr(self, '_detected_arch', None) or 'unknown')
        logging.info(f"[LoRA Optimizer] Architecture preset: {preset_key} ({arch_preset['display_name']})")

        # Formula-based hierarchical merge
        if merge_formula and len(active_loras) >= 2:
            try:
                tree = _parse_merge_formula(merge_formula, len(normalized_stack))
            except ValueError as e:
                logging.warning(f"[LoRA Optimizer] Invalid merge formula: {e} — using flat merge")
                tree = None

            if tree is not None and tree["type"] == "group":
                logging.info(f"[LoRA Optimizer] Using merge formula: {merge_formula}")
                merge_kwargs = {
                    "clip_strength_multiplier": clip_strength_multiplier,
                    "auto_strength": auto_strength,
                    "auto_strength_floor": auto_strength_floor,
                    "optimization_mode": optimization_mode,
                    "patch_compression": patch_compression,
                    "svd_device": svd_device,
                    "normalize_keys": normalize_keys,
                    "sparsification": sparsification,
                    "sparsification_density": sparsification_density,
                    "dare_dampening": dare_dampening,
                    "merge_strategy_override": merge_strategy_override,
                    "merge_refinement": merge_refinement,
                    "strategy_set": strategy_set,
                    "architecture_preset": preset_key,  # resolved, not "auto" — prevents wrong detection in all-virtual sub-merges
                    "decision_smoothing": decision_smoothing,
                    "smooth_slerp_gate": smooth_slerp_gate,
                    "cache_patches": "disabled",  # sub-merges must not thrash parent cache
                    "free_vram_between_passes": free_vram_between_passes,
                    "vram_budget": vram_budget,
                    "_skip_qkv_refusion": True,  # sub-merge patches must stay unfused for outer merge compatibility
                }
                return self._execute_merge_tree(
                    tree, normalized_stack, model, clip, output_strength,
                    _orig_cache_patches=cache_patches, **merge_kwargs)

        # Single LoRA: skip analysis, apply directly via ComfyUI's standard
        # additive LoRA application (faster than diff-based pipeline).
        # auto_strength is a no-op with a single LoRA (scale would be 1.0).
        # Skip fast path for Z-Image: normalized keys (to_q/to_k/to_v) won't
        # match the model's fused qkv keys — need full pipeline + re-fusion.
        if len(active_loras) == 1 and getattr(self, '_detected_arch', None) != 'zimage':
            item = active_loras[0]
            lora_dict = item["lora"]
            strength = item["strength"]
            resolved_output_strength = 1.0 if output_strength < 0 else output_strength

            if item["clip_strength"] is not None:
                clip_str = item["clip_strength"]
            else:
                clip_str = strength * clip_strength_multiplier
            new_model, new_clip = comfy.sd.load_lora_for_models(
                model, clip, lora_dict, resolved_output_strength * strength, resolved_output_strength * clip_str
            )

            report = (
                "=" * 50 + "\n"
                "LORA OPTIMIZER - ANALYSIS REPORT\n"
                "=" * 50 + "\n\n"
                "Single LoRA detected — bypassing analysis.\n"
                f"  Name: {item['name']}\n"
                f"  Strength: {strength}\n"
                f"  Applied directly with output_strength={resolved_output_strength}\n"
                "\n" + "=" * 50
            )
            return (new_model, new_clip, report, None, None)

        # Check instance-level patch cache (survives ComfyUI re-execution
        # triggered by downstream seed changes or similar non-LoRA changes)
        cache_key = self._compute_cache_key(lora_stack, output_strength,
                                            clip_strength_multiplier, auto_strength,
                                            optimization_mode, patch_compression,
                                            svd_device, normalize_keys,
                                            sparsification, sparsification_density,
                                            dare_dampening,
                                            merge_strategy_override, merge_refinement,
                                            strategy_set, architecture_preset,
                                            auto_strength_floor,
                                            decision_smoothing)
        cache_key = f"{cache_key}|mid={id(model)}"
        if cache_patches == "enabled" and cache_key in self._merge_cache:
            model_patches, clip_patches, report, clip_strength_out, lora_data = self._merge_cache[cache_key]
            cached_output_strength = output_strength
            if cached_output_strength < 0 and lora_data and lora_data.get("suggested_max_strength") is not None:
                cached_output_strength = lora_data["suggested_max_strength"]
                clip_strength_out = cached_output_strength * clip_strength_multiplier
            elif cached_output_strength < 0:
                cached_output_strength = 1.0
            new_model = model
            new_clip = clip
            if model is not None and len(model_patches) > 0:
                new_model = model.clone()
                new_model.add_patches(model_patches, cached_output_strength)
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

        # Build target groups so all aliases for the same underlying weight are
        # analyzed and merged together.
        all_lora_prefixes = self._collect_lora_prefixes(active_loras)
        target_groups = self._build_target_groups(all_lora_prefixes, model_keys, clip_keys)
        if not target_groups:
            return (model, clip, "No compatible LoRA keys found. "
                    "LoRAs may be incompatible with this model architecture.", None, None)

        compute_device = self._get_compute_device()
        use_gpu = compute_device.type != "cpu"

        if _analysis_cache is not None:
            # Use pre-computed Pass 1 data (from AutoTuner)
            all_key_targets = _analysis_cache["all_key_targets"]
            target_groups = _analysis_cache.get("target_groups") or {
                prefix: {
                    "target_key": target_info[0],
                    "is_clip": target_info[1],
                    "aliases": [prefix],
                    "label_prefix": prefix,
                }
                for prefix, target_info in all_key_targets.items()
            }
            prefix_stats = _analysis_cache["prefix_stats"]
            per_lora_stats = _analysis_cache["per_lora_stats"]
            pair_accum = _analysis_cache["pair_accum"]
            branch_energy = _analysis_cache.get("branch_energy", {
                "model": {"norm_sq": [0.0] * len(active_loras), "dot": {pair: 0.0 for pair in _analysis_cache.get("pair_accum", {}).keys()}},
                "clip": {"norm_sq": [0.0] * len(active_loras), "dot": {pair: 0.0 for pair in _analysis_cache.get("pair_accum", {}).keys()}},
            })
            all_magnitude_samples = _analysis_cache["all_magnitude_samples"]
            prefix_count = _analysis_cache["prefix_count"]
            skipped_keys = _analysis_cache["skipped_keys"]
            pairs = _analysis_cache.get("pairs") or [
                (i, j) for i in range(len(active_loras))
                for j in range(i + 1, len(active_loras))
            ]
            logging.info(f"[LoRA Optimizer] Using cached analysis ({prefix_count} target groups, skipping Pass 1)")

        else:
            # =====================================================================
            # Pass 1 — Analysis (streaming: diffs computed, sampled, and discarded)
            # =====================================================================
            logging.info("[LoRA Optimizer] Pass 1: Analyzing weight diffs (streaming)...")
            logging.info(f"[LoRA Optimizer]   {len(target_groups)} target groups from {len(all_lora_prefixes)} aliases across {len(active_loras)} LoRAs")
            logging.info(f"[LoRA Optimizer]   Compute device: {compute_device}"
                         f" ({'sequential' if use_gpu else 'threaded'})")
            t_pass1 = time.time()
            analysis_data = self._run_group_analysis(
                target_groups, active_loras, model, clip, compute_device,
                clip_strength_multiplier=clip_strength_multiplier,
                merge_refinement=merge_refinement,
                decision_smoothing=decision_smoothing,
            )
            all_key_targets = analysis_data["all_key_targets"]
            target_groups = analysis_data["target_groups"]
            prefix_stats = analysis_data["prefix_stats"]
            per_lora_stats = analysis_data["per_lora_stats"]
            pair_accum = analysis_data["pair_accum"]
            branch_energy = analysis_data["branch_energy"]
            all_magnitude_samples = analysis_data["all_magnitude_samples"]
            prefix_count = analysis_data["prefix_count"]
            skipped_keys = analysis_data["skipped_keys"]
            pairs = analysis_data["pairs"]

            if prefix_count == 0:
                return (model, clip, "No compatible LoRA keys found. "
                        "LoRAs may be incompatible with this model architecture.", None, None)

            # Log per-LoRA summaries
            for i, stat in enumerate(per_lora_stats):
                avg_r = sum(stat["ranks"]) / len(stat["ranks"]) if stat["ranks"] else 0
                logging.info(f"[LoRA Optimizer]   {stat['name']} ({i+1}/{len(active_loras)}): "
                             f"{stat['key_count']} groups, avg rank {avg_r:.0f}")
            logging.info(f"[LoRA Optimizer]   Total: {prefix_count} target groups ({time.time() - t_pass1:.1f}s)")

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

        fr_preset = arch_preset.get("full_rank", {})
        fr_rank_threshold = fr_preset.get("rank_threshold", 512)
        global_avg_rank = (sum(s["avg_rank"] for s in lora_stats) / len(lora_stats)) if lora_stats else 0
        is_full_rank = global_avg_rank >= fr_rank_threshold
        logging.info(f"[LoRA Optimizer] Global avg rank: {global_avg_rank:.0f} (full-rank threshold: {fr_rank_threshold})")
        if is_full_rank:
            logging.info(
                f"[LoRA Optimizer] Full-rank LoRAs detected "
                f"(avg rank {global_avg_rank:.0f} >= {fr_rank_threshold})"
            )
            if fr_preset.get("disable_slerp_upgrade", False):
                logging.info("[LoRA Optimizer]   SLERP upgrade disabled for full-rank patches")
            if fr_preset.get("prefer_sum_orthogonal", False):
                logging.info("[LoRA Optimizer]   Orthogonal full-rank patches will use weighted_sum (additive)")

        # Pairwise conflict stats and cosine similarity from accumulated counts
        total_overlap = 0
        total_conflict = 0
        total_weighted_total = 0.0
        total_weighted_conflict = 0.0
        total_expected_conflict_weighted = 0.0
        total_excess_conflict_weighted = 0.0
        total_subspace_num = 0.0
        total_subspace_den = 0.0
        pairwise_conflicts = []
        pairwise_similarities = {}
        for i, j in pairs:
            pair_metrics = pair_accum[(i, j)]
            pair_overlap = pair_metrics["overlap"]
            pair_conflict = pair_metrics["conflict"]
            pair_dot = pair_metrics["dot"]
            pair_na_sq = pair_metrics["norm_a_sq"]
            pair_nb_sq = pair_metrics["norm_b_sq"]
            total_overlap += pair_overlap
            total_conflict += pair_conflict
            total_weighted_total += pair_metrics["weighted_total"]
            total_weighted_conflict += pair_metrics["weighted_conflict"]
            total_expected_conflict_weighted += pair_metrics["expected_conflict_weighted"]
            total_excess_conflict_weighted += pair_metrics["excess_conflict_weighted"]
            total_subspace_num += pair_metrics["subspace_num"]
            total_subspace_den += pair_metrics["subspace_den"]
            ratio = pair_conflict / pair_overlap if pair_overlap > 0 else 0
            weighted_ratio = (pair_metrics["weighted_conflict"] / pair_metrics["weighted_total"]) if pair_metrics["weighted_total"] > 0 else ratio
            expected_conflict = (pair_metrics["expected_conflict_weighted"] / pair_metrics["weighted_total"]) if pair_metrics["weighted_total"] > 0 else 0.0
            excess_conflict = (pair_metrics["excess_conflict_weighted"] / pair_metrics["weighted_total"]) if pair_metrics["weighted_total"] > 0 else 0.0
            subspace_overlap = (pair_metrics["subspace_num"] / pair_metrics["subspace_den"]) if pair_metrics["subspace_den"] > 0 else 0.0
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
                "weighted_ratio": weighted_ratio,
                "expected_conflict": expected_conflict,
                "excess_conflict": excess_conflict,
                "cosine_sim": cos_sim,
                "subspace_overlap": subspace_overlap,
            })
            logging.info(f"[LoRA Optimizer]   {pair_label} -> raw={ratio:.1%}, excess={excess_conflict:.1%}, cos_sim={cos_sim:.3f}, subspace={subspace_overlap:.2f}")

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
        avg_weighted_conflict_ratio = total_weighted_conflict / total_weighted_total if total_weighted_total > 0 else avg_conflict_ratio
        avg_expected_conflict = total_expected_conflict_weighted / total_weighted_total if total_weighted_total > 0 else 0.0
        avg_excess_conflict = total_excess_conflict_weighted / total_weighted_total if total_weighted_total > 0 else 0.0
        avg_subspace_overlap = total_subspace_num / total_subspace_den if total_subspace_den > 0 else 0.0
        logging.info(f"[LoRA Optimizer]   Average conflict ratio: {avg_conflict_ratio:.1%}")
        logging.info(f"[LoRA Optimizer]   Excess conflict: {avg_excess_conflict:.1%} | subspace overlap: {avg_subspace_overlap:.2f}")

        # Magnitude ratio
        branch_measure = branch_energy["model"]["norm_sq"]
        model_effective = [
            abs(active_loras[i]["strength"]) * math.sqrt(max(branch_measure[i], 0.0))
            for i in range(len(active_loras))
        ]
        valid_l2 = [m for m in model_effective if m > 0]
        if len(valid_l2) >= 2:
            magnitude_ratio = max(valid_l2) / min(valid_l2)
        else:
            magnitude_ratio = 1.0

        collection_stats = {
            "n_loras": len(active_loras),
            "total_keys": prefix_count,
            "avg_conflict": avg_conflict_ratio,
            "avg_weighted_conflict": avg_weighted_conflict_ratio,
            "avg_expected_conflict": avg_expected_conflict,
            "avg_excess_conflict": avg_excess_conflict,
            "avg_subspace_overlap": avg_subspace_overlap,
            "magnitude_ratio": magnitude_ratio,
            "decision_smoothing": decision_smoothing,
        }

        # Auto-select parameters (density estimated from pre-sampled magnitudes)
        global_avg_cos_sim = (sum(ps.get("cosine_sim", 0.0) for ps in pairwise_conflicts)
                              / len(pairwise_conflicts)) if pairwise_conflicts else 0.0
        mode, density, sign_method, reasoning = self._auto_select_params(
            avg_conflict_ratio, magnitude_ratio, magnitude_samples=all_magnitude_samples,
            avg_cos_sim=global_avg_cos_sim, strategy_set=strategy_set,
            avg_excess_conflict=avg_excess_conflict,
            avg_subspace_overlap=avg_subspace_overlap,
            arch_preset=arch_preset
        )
        del all_magnitude_samples

        # Apply merge strategy override from Conflict Editor
        # Skip when user explicitly chose additive (protects DPO/edit LoRAs)
        if merge_strategy_override and optimization_mode != "additive":
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
        model_auto_scale = 1.0
        clip_auto_scale = 1.0
        if auto_strength == "enabled":
            auto_strength_info = self._compute_auto_strengths(
                active_loras, branch_energy,
                clip_strength_multiplier=clip_strength_multiplier,
                arch_preset=arch_preset,
                detected_arch=getattr(self, '_detected_arch', None),
                auto_strength_floor=auto_strength_floor,
                is_full_rank=is_full_rank,
            )
            model_auto_scale = auto_strength_info["model_scale"]
            clip_auto_scale = auto_strength_info["clip_scale"]
            for i, stat in enumerate(lora_stats):
                stat["original_strength"] = stat["strength"]
                stat["strength"] = auto_strength_info["model_strengths"][i]

            if auto_strength_info["reasoning"]:
                logging.info(f"[LoRA Optimizer] Auto-strength: {auto_strength_info['reasoning'][0]}")
            for i in range(len(active_loras)):
                logging.info(
                    f"[LoRA Optimizer]   {active_loras[i]['name']}: "
                    f"model {active_loras[i]['strength']} -> {auto_strength_info['model_strengths'][i]:.4f}"
                )

        # Free GPU cache between passes if requested
        if free_vram_between_passes == "enabled" and use_gpu:
            torch.cuda.empty_cache()

        # Resolve patch_compression rank (sum of input LoRA ranks)
        sum_rank = sum(int(stat["avg_rank"]) for stat in lora_stats if stat["avg_rank"] > 0)
        compress_rank = 0  # 0 = disabled
        if patch_compression in ("smart", "aggressive"):
            compress_rank = max(sum_rank, 64)  # floor at 64
            logging.info(f"[LoRA Optimizer] Patch compression: {patch_compression} (rank {compress_rank} from sum of input LoRA ranks)")

        # Resolve SVD device for compression
        resolved_svd_device = None
        if compress_rank > 0 and svd_device == "gpu" and torch.cuda.is_available():
            resolved_svd_device = torch.device("cuda")
        elif compress_rank > 0 and svd_device == "cpu":
            resolved_svd_device = None  # CPU is the default in _compress_to_lowrank

        # =====================================================================
        # Pass 2 — Merge (recompute diffs per target group, merge, discard)
        # =====================================================================
        logging.info(f"[LoRA Optimizer] Pass 2: Merging {len(target_groups)} target groups "
                     f"({optimization_mode} strategy, "
                     f"{'sequential' if use_gpu else 'threaded'})...")
        t_pass2 = time.time()
        model_patches = {}
        clip_patches = {}
        processed_keys = 0
        compressed_count = 0
        strategy_counts = {"weighted_sum": 0, "weighted_average": 0, "slerp": 0, "ties": 0, "consensus": 0}
        prefix_decisions = []  # list of (prefix, mode, conflict_ratio, n_loras) for block map
        has_virtual_loras = any(item.get("_precomputed_diffs") for item in active_loras)

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

        def _merge_one_group(label_prefix, target_group):
            """Recompute diffs for one target group, merge, return patch or None."""
            nonlocal gpu_patch_bytes
            should_keep = vram_budget_bytes > 0 and gpu_patch_bytes < vram_budget_bytes
            target_key = target_group["target_key"]
            is_clip_key = target_group["is_clip"]

            # Determine strategy BEFORE computing diffs (use Pass 1 stats)
            pf_conflict = 0.0
            pf_n_loras = 0
            pf_mode = mode
            pf_density = density
            pf_sign = sign_method
            pf_orthogonal = False
            pf_opposing = False
            if optimization_mode == "additive":
                pf_mode = "weighted_sum"
                pf_n_loras = prefix_stats.get(label_prefix, {}).get("n_loras", 0)
                pf_conflict = prefix_stats.get(label_prefix, {}).get("decision_conflict",
                                                                     prefix_stats.get(label_prefix, {}).get("conflict_ratio", 0.0))
            elif optimization_mode == "per_prefix" and label_prefix in prefix_stats:
                pf = prefix_stats[label_prefix]
                pf_conflict = pf.get("decision_conflict", pf.get("conflict_ratio", 0.0))
                pf_n_loras = pf["n_loras"]
                if pf["n_loras"] <= 1:
                    pf_mode = "weighted_sum"
                    pf_density = 0.5
                    pf_sign = "frequency"
                else:
                    pf_mode, pf_density, pf_sign, _ = self._auto_select_params(
                        pf["conflict_ratio"], pf.get("decision_magnitude_ratio", pf["magnitude_ratio"]),
                        magnitude_samples=pf.get("magnitude_samples"),
                        avg_cos_sim=pf.get("decision_cosine", pf.get("avg_cos_sim", 0.0)),
                        avg_excess_conflict=pf.get("decision_conflict", pf.get("excess_conflict", pf.get("conflict_ratio", 0.0))),
                        avg_subspace_overlap=pf.get("decision_subspace_overlap", pf.get("avg_subspace_overlap", 0.0)),
                        strategy_set=strategy_set,
                        arch_preset=arch_preset,
                        precomputed_density=pf.get("precomputed_density"),
                    )
                    # Upgrade weighted_average → slerp for 2+ non-opposing LoRAs.
                    # SLERP preserves magnitude better than weighted_average's /N reduction,
                    # which is critical for video LoRAs where motion energy matters.
                    # Skip for opposing LoRAs (cos < 0): SLERP interpolates between opposing
                    # directions while preserving magnitude, amplifying artifacts.
                    pf_raw_cos = pf.get("decision_cosine", pf.get("avg_cos_sim", 0.0)) if smooth_slerp_gate else pf.get("avg_cos_sim", 0.0)
                    pf_orthogonal = abs(pf_raw_cos) < arch_preset["orthogonal_cos_sim_max"]
                    pf_opposing = pf_raw_cos < 0
                    # Full-rank gate: skip SLERP upgrade — for full-rank patches the
                    # information is spread across all dimensions, and SLERP's
                    # hypersphere interpolation loses signal from both LoRAs.
                    if (pf_mode == "weighted_average" and pf["n_loras"] >= 2
                            and strategy_set == "full"
                            and not pf_opposing
                            and not (is_full_rank and fr_preset.get("disable_slerp_upgrade", False))):
                        pf_mode = "slerp"
                    if (is_full_rank and fr_preset.get("prefer_sum_orthogonal", False)
                            and pf_mode == "weighted_average" and pf_orthogonal):
                        pf_mode = "weighted_sum"

            # Apply merge strategy override from Conflict Editor (takes priority over auto-selection)
            # Skip when user explicitly chose additive (protects DPO/edit LoRAs)
            if (merge_strategy_override and optimization_mode != "additive"
                    and merge_strategy_override in ("ties", "weighted_average", "weighted_sum", "consensus", "slerp")):
                pf_mode = merge_strategy_override

            pf = prefix_stats.get(label_prefix, {})
            raw_n = pf.get("raw_n_loras", pf_n_loras)
            if pf_n_loras <= 1 and pf_mode != "weighted_sum":
                pf_mode = "weighted_sum"

            linear_stats = None
            if (pf_mode in ("weighted_sum", "weighted_average", "normalize")
                    and sparsification == "disabled"
                    and merge_refinement == "none"
                    and not has_virtual_loras):
                linear_patch_info = self._build_exact_linear_patch(
                    target_group, active_loras, raw_n, pf_mode,
                    is_clip_key=is_clip_key, model_scale=model_auto_scale,
                )
                if linear_patch_info is not None:
                    patch = linear_patch_info["patch"]
                    weights = linear_patch_info["weights"]
                    input_norms_mean = (
                        sum(math.sqrt(max(pf.get("per_lora_norm_sq", {}).get(i, 0.0), 0.0)) * abs(w)
                            for i, w in weights.items()) / len(weights)
                    ) if weights else 0.0
                    energy_sq = 0.0
                    for i, weight in weights.items():
                        energy_sq += (weight ** 2) * pf.get("per_lora_norm_sq", {}).get(i, 0.0)
                    for (i, j), dot in pf.get("pairwise_dots", {}).items():
                        if i in weights and j in weights:
                            energy_sq += 2.0 * weights[i] * weights[j] * dot
                    merged_norm = math.sqrt(max(energy_sq, 0.0))
                    linear_stats = (input_norms_mean, merged_norm)
                    if should_keep:
                        p_bytes = self._estimate_single_patch_bytes(patch)
                        if gpu_patch_bytes + p_bytes <= vram_budget_bytes:
                            patch = self._move_patch_to_device(patch, compute_device)
                            gpu_patch_bytes += p_bytes
                    return (
                        target_key, is_clip_key, patch, pf_mode, label_prefix,
                        pf_conflict, max(pf_n_loras, 1), False,
                        linear_stats[0], linear_stats[1]
                    )

            prepared = self._prepare_group_diffs(
                target_group, active_loras, model, clip, compute_device,
                clip_strength_multiplier=clip_strength_multiplier,
                merge_refinement=merge_refinement,
                diff_cache=_diff_cache,
                auto_scale=model_auto_scale if not is_clip_key else 1.0,
            )
            if prepared is None or len(prepared["diffs"]) == 0:
                return None

            diffs_list = []
            storage_dtype = prepared["storage_dtype"]
            for i in sorted(prepared["diffs"].keys()):
                diffs_list.append((prepared["diffs"][i], prepared["eff_strengths"][i]))

            if len(diffs_list) <= 1 and pf_mode != "weighted_sum":
                pf_mode = "weighted_sum"

            # Create deterministic per-group RNG for reproducible sparsification
            sp_gen = None
            if sparsification != "disabled":
                seed = int(hashlib.sha256(label_prefix.encode()).hexdigest(), 16) % (2**63)
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
            pf_quality = merge_refinement
            if (pf_mode == "weighted_average" and pf_opposing):
                pf_quality = "none"

            merged_diff = self._merge_diffs(
                diffs_list, pf_mode,
                density=pf_density, majority_sign_method=pf_sign,
                compute_device=compute_device,
                sparsification=sparsification,
                sparsification_density=sparsification_density,
                sparsification_generator=sp_gen,
                merge_refinement=pf_quality,
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
                               (patch_compression == "aggressive" or pf_mode != "ties"))
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
            return (target_key, is_clip_key, patch, pf_mode, label_prefix, pf_conflict, max(pf_n_loras, 1), is_compressed, input_norms_mean, merged_norm)

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
            group_items = list(target_groups.items())
            n_loras = len(active_loras)
            for idx, (label_prefix, target_group) in enumerate(group_items):
                if _diff_cache is not None and idx + 1 < len(group_items):
                    next_group = group_items[idx + 1][1]
                    prefetch_keys = [
                        (alias, i)
                        for alias in next_group["aliases"]
                        for i in range(n_loras)
                    ]
                    _diff_cache.prefetch(prefetch_keys)
                _collect_merge_result(_merge_one_group(label_prefix, target_group))
        else:
            max_workers = min(4, max(1, len(target_groups)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_merge_one_group, label_prefix, target_group): label_prefix
                    for label_prefix, target_group in target_groups.items()
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
        spars_skipped = getattr(self, '_sparsification_skipped', 0)
        if spars_skipped > 0:
            logging.info(f"[LoRA Optimizer]   Conflict-aware sparsification skipped for "
                         f"{spars_skipped} groups (base-rate noise from orthogonal LoRAs)")
            self._sparsification_skipped = 0
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
        # Skip in sub-merges: virtual LoRA patches must stay unfused so the
        # outer merge can pair them with other LoRAs' unfused keys.
        if getattr(self, '_detected_arch', None) == 'zimage' and not _skip_qkv_refusion:
            if len(model_patches) > 0:
                model_patches = self._refuse_zimage_patches(model_patches)
                logging.info(f"[LoRA Optimizer] Re-fused Z-Image QKV patches ({len(model_patches)} model patches)")

        # Build reverse key map: target_key → canonical prefix metadata
        # (used by SaveMergedLoRA to reconstruct standard LoRA key names)
        reverse_key_map = {}
        for label_prefix, target_group in target_groups.items():
            target_key = target_group["target_key"]
            tkey = target_key[0] if isinstance(target_key, tuple) else target_key
            entry = {
                "canonical_prefix": label_prefix,
                "aliases": list(target_group["aliases"]),
            }
            reverse_key_map[target_key] = entry
            reverse_key_map[tkey] = entry

        # Apply patches
        new_model = model
        new_clip = clip

        auto_output_strength = False
        if output_strength < 0 and suggested_max_strength is not None:
            output_strength = suggested_max_strength
            auto_output_strength = True
            logging.info(f"[LoRA Optimizer] Auto output_strength: {output_strength:.2f} (suggested max)")
        elif output_strength < 0:
            output_strength = 1.0
            logging.info("[LoRA Optimizer] Auto output_strength: no suggestion available, using 1.0")

        all_explicit_clip = all(item["clip_strength"] is not None for item in active_loras)
        if all_explicit_clip:
            clip_strength_out = output_strength * clip_auto_scale
        else:
            clip_strength_out = output_strength * clip_strength_multiplier * clip_auto_scale

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
            merge_refinement=merge_refinement,
            compatibility_warnings=compatibility_warnings,
            strategy_set=strategy_set,
            architecture_preset=preset_key,
        )

        # Bundle LORA_DATA for optional downstream saving
        lora_data = {
            "model_patches": model_patches,
            "clip_patches": clip_patches,
            "key_map": reverse_key_map,
            "output_strength": output_strength,
            "clip_strength": clip_strength_out,
            "suggested_max_strength": suggested_max_strength,
            "sum_rank": compress_rank if compress_rank > 0 else 128,
            "merge_metadata": {
                "source_loras": [{"name": item["name"], "strength": item["strength"]} for item in active_loras],
                "mode": mode,
                "optimization_mode": optimization_mode,
                "architecture": getattr(self, '_detected_arch', None) or 'unknown',
                "architecture_preset": preset_key,
                "auto_strength": auto_strength,
                "sparsification": sparsification,
                "sparsification_density": sparsification_density,
                "merge_refinement": merge_refinement,
                "strategy_set": strategy_set,
                "bake_strength_output": output_strength,
                "bake_strength_clip": clip_strength_out,
            },
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
        # Report is returned in the UI output — no need to also save to disk

        logging.info(f"[LoRA Optimizer] Done! {processed_keys} keys processed ({time.time() - t_start:.1f}s total)")

        return (new_model, new_clip, report, None, lora_data)

    # ------------------------------------------------------------------
    #  Merge-formula tree executor
    # ------------------------------------------------------------------

    def _execute_merge_tree(self, tree, normalized_stack, model, clip, output_strength, _orig_cache_patches="enabled", **kwargs):
        """
        Execute a merge formula tree leaf-to-root.
        Each sub-group is a full optimize_merge call.
        Returns the same 5-tuple as optimize_merge.
        """
        # Save _detected_arch — sub-merges may overwrite it (e.g. all-virtual stacks)
        saved_arch = getattr(self, '_detected_arch', None)

        # Collect the final flat stack by recursively resolving groups
        final_stack, sub_reports = self._resolve_tree_to_stack(
            tree, normalized_stack, model, clip, **kwargs)

        # Restore _detected_arch for the final merge
        self._detected_arch = saved_arch

        # Final merge: restore original cache_patches setting (sub-merges use "disabled")
        final_kwargs = dict(kwargs)
        final_kwargs["cache_patches"] = _orig_cache_patches
        final_kwargs["_skip_qkv_refusion"] = False  # final merge must re-fuse QKV for Z-Image
        result = self.optimize_merge(model, final_stack, output_strength, clip=clip, **final_kwargs)

        # Prepend sub-reports to the final report
        if sub_reports:
            if len(result) == 3:
                model_out, report, lora_data = result
                clip_out, tuner_data = None, None
            else:
                model_out, clip_out, report, tuner_data, lora_data = result
            separator = "\n" + "=" * 50 + "\n"
            sub_section = separator.join(sub_reports)
            report = (
                "MERGE FORMULA SUB-MERGE REPORTS\n"
                + separator + sub_section + separator
                + "\nFINAL MERGE REPORT:\n" + report
            )
            if len(result) == 3:
                result = (model_out, report, lora_data)
            else:
                result = (model_out, clip_out, report, tuner_data, lora_data)

        return result

    def _resolve_tree_to_stack(self, tree, normalized_stack, model, clip, **kwargs):
        """
        Recursively resolve a merge tree into a flat LoRA stack.
        Groups are merged into virtual LoRAs; leaves reference the original stack.
        Returns (resolved_stack, sub_reports).
        """
        sub_reports = []

        if tree["type"] == "leaf":
            item = dict(normalized_stack[tree["index"]])
            if "metadata" in item and isinstance(item["metadata"], dict):
                item["metadata"] = dict(item["metadata"])
            if tree["weight"] is not None:
                item["strength"] = tree["weight"]
            return ([item], sub_reports)

        # Group: resolve each child
        resolved = []
        for child in tree["children"]:
            if child["type"] == "leaf":
                item = dict(normalized_stack[child["index"]])
                if "metadata" in item and isinstance(item["metadata"], dict):
                    item["metadata"] = dict(item["metadata"])
                if child["weight"] is not None:
                    item["strength"] = child["weight"]
                resolved.append(item)
            else:
                # Sub-group: resolve recursively then merge
                sub_stack, child_reports = self._resolve_tree_to_stack(
                    child, normalized_stack, model, clip, **kwargs)
                sub_reports.extend(child_reports)

                if len(sub_stack) >= 2:
                    # Merge this sub-group via full pipeline
                    try:
                        sub_result = self.optimize_merge(
                            model, sub_stack, 1.0, clip=clip, **kwargs)
                        if len(sub_result) == 3:
                            sub_model, sub_report, sub_lora_data = sub_result
                            sub_clip = None
                        else:
                            sub_model, sub_clip, sub_report, _, sub_lora_data = sub_result
                        sub_reports.append(sub_report)

                        if sub_lora_data is None:
                            # Single-LoRA fast path was hit (e.g. other LoRA had strength 0)
                            # Pass sub-stack items through directly instead of empty virtual LoRA
                            del sub_model, sub_clip, sub_result
                            for sub_item in sub_stack:
                                item = dict(sub_item)
                                if child.get("weight") is not None:
                                    item["strength"] = child["weight"]
                                resolved.append(item)
                            continue

                        # Extract patches as virtual LoRA from merge output
                        sub_model_patches = sub_lora_data.get("model_patches", {})
                        sub_clip_patches = sub_lora_data.get("clip_patches", {})
                        virtual = self._model_to_virtual_lora(
                            sub_model_patches, sub_clip_patches, child)
                        del sub_model, sub_clip, sub_result, sub_lora_data
                        if child["weight"] is not None:
                            virtual["strength"] = child["weight"]
                        resolved.append(virtual)
                    except Exception as e:
                        logging.warning(
                            f"[LoRA Optimizer] Sub-merge failed: {e} — "
                            "falling back to flat merge for this sub-group")
                        for item in sub_stack:
                            resolved.append(item)
                elif len(sub_stack) == 1:
                    item = sub_stack[0]
                    if child["weight"] is not None:
                        item["strength"] = child["weight"]
                    resolved.append(item)

        return (resolved, sub_reports)

    @staticmethod
    def _model_to_virtual_lora(model_patches, clip_patches, tree_node):
        """
        Build a virtual LoRA from pre-computed merge patches.
        Stores the actual diff tensors (keyed by target model key) so the
        merge pipeline can use them directly without LoRA decomposition.
        """
        virtual_lora = {}

        for key, patch in model_patches.items():
            # Store raw patch; expansion to dense is deferred to the merge
            # loop so compressed adapters (LoRAAdapter etc.) don't blow up
            # memory here.
            if isinstance(patch, tuple) and patch[0] == "diff":
                virtual_lora[key] = patch[1][0]  # extract the tensor
            else:
                virtual_lora[key] = patch  # tensor or adapter object

        for key, patch in clip_patches.items():
            if isinstance(patch, tuple) and patch[0] == "diff":
                virtual_lora[key] = patch[1][0]
            else:
                virtual_lora[key] = patch

        # Build label from tree
        def _tree_label(node):
            if node["type"] == "leaf":
                return str(node["index"] + 1)
            return "(" + "+".join(_tree_label(c) for c in node["children"]) + ")"

        return {
            "name": _tree_label(tree_node),
            "lora": virtual_lora,
            "_precomputed_diffs": True,
            "strength": 1.0,
            "clip_strength": None,
            "conflict_mode": "all",
            "key_filter": "all",
            "metadata": {},
        }


class LoRAMergeSettings:
    """
    Common merge settings shared between Optimizer and AutoTuner modes.
    Connect to the 'merge_settings' input of either settings node.
    """

    _DEFAULTS = {
        "normalize_keys": "enabled",
        "architecture_preset": "auto",
        "auto_strength_floor": -1.0,
        "decision_smoothing": 0.25,
        "smooth_slerp_gate": False,
        "vram_budget": 0.0,
        "cache_patches": "enabled",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalize_keys": (["disabled", "enabled"], {
                    "default": "enabled",
                    "tooltip": "Ensures LoRAs from different training tools (Kohya, PEFT, etc.) work together. Keep enabled unless you have a specific reason to disable."
                }),
                "architecture_preset": (["auto", "sd_unet", "dit", "llm"], {
                    "default": "auto",
                    "tooltip": "Tells the optimizer what type of model you're using so it can pick the best merge settings. 'auto' detects it for you. Only change if auto-detection gets it wrong."
                }),
                "auto_strength_floor": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much the weakest LoRA is allowed to be scaled down during auto-strength. Higher values keep all LoRAs more visible. -1 picks a good default based on model type."
                }),
                "decision_smoothing": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Prevents the optimizer from picking wildly different merge methods for similar layers. Higher values = more consistent choices across the model. 0.2-0.4 is usually a good range."
                }),
                "smooth_slerp_gate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Changes how the optimizer decides when to use SLERP blending. When enabled, the decision is smoother and more stable. Try enabling if you notice inconsistent results between runs."
                }),
                "vram_budget": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much GPU memory to use for storing merge results. 0 = keep everything in system RAM (safest). Increase to use GPU memory and reduce RAM usage."
                }),
                "cache_patches": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": "Keeps the merge result in memory so re-running the workflow is instant. Disable to free RAM — recommended for large video models."
                }),
            },
        }

    RETURN_TYPES = ("MERGE_SETTINGS",)
    RETURN_NAMES = ("merge_settings",)
    FUNCTION = "build_settings"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Shared settings that apply to both Optimizer and AutoTuner modes. "
        "Connect to the 'merge_settings' input on an Optimizer Settings or AutoTuner Settings node."
    )

    def build_settings(self, normalize_keys, architecture_preset,
                       auto_strength_floor, decision_smoothing,
                       smooth_slerp_gate, vram_budget, cache_patches):
        return ({
            "normalize_keys": normalize_keys,
            "architecture_preset": architecture_preset,
            "auto_strength_floor": auto_strength_floor,
            "decision_smoothing": decision_smoothing,
            "smooth_slerp_gate": smooth_slerp_gate,
            "vram_budget": vram_budget,
            "cache_patches": cache_patches,
        },)


class LoRAOptimizerSettings:
    """
    Pure data node that outputs Advanced-mode settings for the Simple optimizer.
    Connect to the 'settings' input of LoRA Optimizer to unlock all Advanced
    knobs without switching node types.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_strength": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": "Automatically turns down individual LoRA strengths when combining many LoRAs, preventing oversaturation. Recommended to keep enabled."
                }),
                "optimization_mode": (["per_prefix", "global", "additive"], {
                    "default": "per_prefix",
                    "tooltip": "How the optimizer picks merge methods. 'per_prefix' (recommended): picks the best method for each part of the model. 'global': uses one method everywhere. 'additive': simple stacking with no conflict handling — use for edit/DPO LoRAs."
                }),
                "merge_refinement": (["none", "refine", "full"], {
                    "default": "none",
                    "tooltip": "Extra processing to reduce interference between LoRAs. 'none': fastest, usually fine. 'refine': light cleanup for better quality. 'full': most thorough but slower. Try 'refine' if you see artifacts or color shifts."
                }),
                "sparsification": (["disabled", "dare", "della", "dare_conflict", "della_conflict"], {
                    "default": "disabled",
                    "tooltip": "Removes low-impact weights before merging to reduce interference between LoRAs. The 'conflict' variants (recommended if enabling) only trim where LoRAs disagree, leaving unique contributions intact."
                }),
                "sparsification_density": ("FLOAT", {
                    "default": 0.7, "min": 0.01, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of each LoRA to keep when sparsification is enabled. 0.7 = keep 70%. Lower values are more aggressive — reduces interference but may lose detail."
                }),
                "dare_dampening": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Controls how strongly surviving weights are boosted after sparsification trims some away. 0 = standard boost, higher = gentler. Only matters when sparsification is enabled."
                }),
                "patch_compression": (["smart", "aggressive", "disabled"], {
                    "default": "smart",
                    "tooltip": "Shrinks the merged result to use less memory. 'smart' (recommended): only compresses where it's lossless. 'aggressive': compresses everything, saves more memory but slightly lossy. 'disabled': no compression."
                }),
                "svd_device": (["gpu", "cpu"], {
                    "default": "gpu",
                    "tooltip": "Where to run compression math. GPU is much faster. Switch to CPU only if you get out-of-memory errors."
                }),
                "free_vram_between_passes": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Frees GPU memory between merge steps. Enable if you're running out of VRAM on large models. Barely affects speed."
                }),
                "strategy_set": (["full", "no_slerp", "basic"], {
                    "default": "full",
                    "tooltip": "Which merge methods the optimizer can choose from. 'full' (recommended): all methods available including SLERP blending. 'no_slerp': excludes SLERP. 'basic': only simple averaging."
                }),
            },
            "optional": {
                "merge_settings": ("MERGE_SETTINGS", {
                    "tooltip": "Connect a LoRA Merge Settings node here to share common settings. Uses good defaults if not connected."
                }),
                "merge_strategy_override": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Forces one specific merge method everywhere instead of letting the optimizer choose. Connect from a LoRA Conflict Editor node."
                }),
            },
        }

    RETURN_TYPES = ("OPTIMIZER_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build_settings"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Advanced settings for fine-tuning how LoRAs are merged. "
        "Connect to the 'settings' input on the LoRA Optimizer node."
    )

    def build_settings(self, auto_strength, optimization_mode, merge_refinement,
                       sparsification, sparsification_density, dare_dampening,
                       patch_compression, svd_device,
                       free_vram_between_passes, strategy_set,
                       merge_settings=None, merge_strategy_override=""):
        ms = merge_settings if merge_settings is not None else LoRAMergeSettings._DEFAULTS
        return ({
            "mode": "advanced",
            "auto_strength": auto_strength,
            "optimization_mode": optimization_mode,
            "merge_refinement": merge_refinement,
            "sparsification": sparsification,
            "sparsification_density": sparsification_density,
            "dare_dampening": dare_dampening,
            "patch_compression": patch_compression,
            "svd_device": svd_device,
            "cache_patches": ms["cache_patches"],
            "free_vram_between_passes": free_vram_between_passes,
            "normalize_keys": ms["normalize_keys"],
            "strategy_set": strategy_set,
            "architecture_preset": ms["architecture_preset"],
            "auto_strength_floor": ms["auto_strength_floor"],
            "decision_smoothing": ms["decision_smoothing"],
            "smooth_slerp_gate": ms["smooth_slerp_gate"],
            "vram_budget": ms["vram_budget"],
            "merge_strategy_override": merge_strategy_override,
        },)


class LoRAAutoTunerSettings:
    """
    Pure data node that outputs AutoTuner-mode settings for the Simple optimizer.
    Connect to the 'settings' input of LoRA Optimizer to run a full AutoTuner
    sweep without switching node types.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "top_n": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "How many of the best configurations to try with a real merge. Higher = explores more options but takes longer."
                }),
                "scoring_svd": (["disabled", "merge_quality", "lora_rank", "full"], {
                    "default": "disabled",
                    "tooltip": "SVD-based scoring for ranking configurations.\n"
                               "disabled: fast norm-only scoring (usually sufficient).\n"
                               "merge_quality: SVD on merged diff tensors — more thorough quality measurement.\n"
                               "lora_rank: effective rank of LoRA factors — experimental, changes ranking.\n"
                               "full: both merge_quality + lora_rank."
                }),
                "scoring_device": (["cpu", "gpu"], {
                    "default": "gpu",
                    "tooltip": "Where to run scoring math. GPU is much faster, especially with SVD scoring modes."
                }),
                "scoring_speed": (["full", "fast", "turbo", "turbo+"], {
                    "default": "turbo",
                    "tooltip": "How thoroughly to score each configuration. 'full': most accurate, slowest. 'turbo' (recommended): good balance of speed and accuracy. 'turbo+': fastest, may miss subtle differences."
                }),
                "scoring_formula": (["v2", "v1"], {
                    "default": "v2",
                    "tooltip": "Which scoring formula to use. v2 (recommended): smarter scoring that adapts to your model type. v1: older formula, kept for comparison."
                }),
                "diff_cache_mode": (["disabled", "auto", "ram", "disk"], {
                    "default": "auto",
                    "tooltip": "Caches intermediate data to speed up the sweep. 'auto' (recommended): uses RAM first, spills to disk if needed. 'disabled': slower but uses no extra memory."
                }),
                "diff_cache_ram_pct": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "How much of your free RAM the diff cache can use (in 'auto' mode). 0.5 = up to half your available RAM."
                }),
                "record_dataset": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Saves detailed scoring data to a file for analysis. Only useful for developers tuning the scoring system."
                }),
                "memory_mode": (["disabled", "auto", "read_only", "clear_and_run"], {
                    "default": "auto",
                    "tooltip": "Persistent memory for tuning results across sessions.\n"
                               "auto: Load cached results if available, save after tuning.\n"
                               "read_only: Use cached results but don't save new ones.\n"
                               "clear_and_run: Delete cached entry and re-tune from scratch."
                }),
                "selection": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Which ranked configuration to apply (1 = top-ranked). "
                               "Change this to try a different config without re-running the full sweep."
                }),
            },
            "optional": {
                "merge_settings": ("MERGE_SETTINGS", {
                    "tooltip": "Connect a LoRA Merge Settings node here to share common settings. Uses good defaults if not connected."
                }),
                "evaluator": ("AUTOTUNER_EVALUATOR", {
                    "tooltip": "Connect an external evaluator to influence how configurations are ranked. Optional — the built-in scoring works well on its own."
                }),
            },
        }

    RETURN_TYPES = ("OPTIMIZER_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build_settings"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Runs an automatic parameter sweep to find the best merge settings for your LoRA stack. "
        "Connect to the 'settings' input on the LoRA Optimizer node."
    )

    def build_settings(self, top_n, scoring_svd, scoring_device,
                       scoring_speed, scoring_formula,
                       diff_cache_mode, diff_cache_ram_pct, record_dataset,
                       memory_mode="disabled", selection=1,
                       merge_settings=None, evaluator=None):
        ms = merge_settings if merge_settings is not None else LoRAMergeSettings._DEFAULTS
        return ({
            "mode": "autotuner",
            "top_n": top_n,
            "scoring_svd": scoring_svd,
            "scoring_device": scoring_device,
            "scoring_speed": scoring_speed,
            "scoring_formula": scoring_formula,
            "output_mode": "merge",
            "smooth_slerp_gate": ms["smooth_slerp_gate"],
            "normalize_keys": ms["normalize_keys"],
            "architecture_preset": ms["architecture_preset"],
            "auto_strength_floor": ms["auto_strength_floor"],
            "decision_smoothing": ms["decision_smoothing"],
            "vram_budget": ms["vram_budget"],
            "cache_patches": ms["cache_patches"],
            "diff_cache_mode": diff_cache_mode,
            "diff_cache_ram_pct": diff_cache_ram_pct,
            "record_dataset": record_dataset,
            "evaluator": evaluator,
            "memory_mode": memory_mode,
            "selection": selection,
        },)


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
                    "tooltip": "Your base model (e.g. SDXL, Flux). The merged LoRAs will be applied to it."
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "Connect your LoRA Stack node here — this is the list of LoRAs you want to merge together."
                }),
                "output_strength": ("FLOAT", {
                    "default": 1.0, "min": -1.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Master volume for the merged result. 1.0 = full effect, 0.5 = half. Set to -1 for auto: the optimizer picks a good strength for you."
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Connect your text encoder so LoRAs can also affect how prompts are understood. Leave empty for video models."
                }),
                "clip_strength_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "How strongly LoRAs affect text understanding. At 1.0, same strength as the model. Lower values reduce LoRA influence on prompts while keeping the visual effect."
                }),
                "tuner_data": ("TUNER_DATA", {
                    "tooltip": "Connect results from a LoRA AutoTuner or Load Tuner Data node. The optimizer will use the best settings found by the tuner instead of defaults."
                }),
                "settings": ("OPTIMIZER_SETTINGS", {
                    "tooltip": "Connect a Settings node (Optimizer Settings or AutoTuner Settings) for full control. Takes priority over tuner_data and defaults."
                }),
            },
        }

    FUNCTION = "execute_simple"
    DESCRIPTION = (
        "Simplified LoRA Optimizer — merges a LoRA stack with good defaults. "
        "Connect tuner_data to apply AutoTuner results, or connect settings "
        "from an Optimizer/AutoTuner Settings node for full control."
    )

    _SIMPLE_DEFAULTS = dict(
        auto_strength="enabled",
        auto_strength_floor=-1.0,
        free_vram_between_passes="disabled",
        vram_budget=0.0,
        optimization_mode="per_prefix",
        cache_patches="enabled",
        patch_compression="smart",
        svd_device="gpu",
        normalize_keys="enabled",
        sparsification="disabled",
        sparsification_density=0.7,
        dare_dampening=0.0,
        merge_strategy_override="",
        merge_refinement="none",
        strategy_set="full",
        architecture_preset="auto",
        decision_smoothing=0.25,
        smooth_slerp_gate=False,
    )

    def execute_simple(self, model, lora_stack, output_strength,
                       clip=None, clip_strength_multiplier=1.0, tuner_data=None,
                       settings=None):
        # Priority 1: settings node connected
        if settings is not None:
            mode = settings.get("mode")
            if mode == "advanced":
                # Free delegate caches when switching away from autotuner mode
                delegate = getattr(self, '_autotuner_delegate', None)
                if delegate is not None:
                    if getattr(delegate, '_autotuner_cache', None):
                        delegate._autotuner_cache.clear()
                    if getattr(delegate, '_merge_cache', None):
                        delegate._merge_cache.clear()
                return super().optimize_merge(
                    model, lora_stack, output_strength,
                    clip=clip, clip_strength_multiplier=clip_strength_multiplier,
                    auto_strength=settings["auto_strength"],
                    auto_strength_floor=settings["auto_strength_floor"],
                    optimization_mode=settings["optimization_mode"],
                    sparsification=settings["sparsification"],
                    sparsification_density=settings["sparsification_density"],
                    dare_dampening=settings["dare_dampening"],
                    merge_strategy_override=settings.get("merge_strategy_override", ""),
                    merge_refinement=settings["merge_refinement"],
                    strategy_set=settings["strategy_set"],
                    normalize_keys=settings["normalize_keys"],
                    architecture_preset=settings["architecture_preset"],
                    decision_smoothing=settings["decision_smoothing"],
                    smooth_slerp_gate=settings["smooth_slerp_gate"],
                    cache_patches=settings["cache_patches"],
                    patch_compression=settings["patch_compression"],
                    svd_device=settings["svd_device"],
                    free_vram_between_passes=settings["free_vram_between_passes"],
                    vram_budget=settings["vram_budget"],
                )
            elif mode == "autotuner":
                # Free optimizer cache when switching away from advanced mode
                if self._merge_cache:
                    self._merge_cache.clear()
                if not hasattr(self, '_autotuner_delegate'):
                    self._autotuner_delegate = LoRAAutoTuner()
                result = self._autotuner_delegate.auto_tune(
                    model, lora_stack, output_strength,
                    clip=clip, clip_strength_multiplier=clip_strength_multiplier,
                    top_n=settings["top_n"],
                    scoring_svd=settings["scoring_svd"],
                    scoring_device=settings["scoring_device"],
                    scoring_speed=settings["scoring_speed"],
                    scoring_formula=settings.get("scoring_formula", "v2"),
                    output_mode=settings["output_mode"],
                    smooth_slerp_gate=settings["smooth_slerp_gate"],
                    normalize_keys=settings["normalize_keys"],
                    architecture_preset=settings["architecture_preset"],
                    auto_strength_floor=settings["auto_strength_floor"],
                    decision_smoothing=settings["decision_smoothing"],
                    vram_budget=settings["vram_budget"],
                    cache_patches=settings["cache_patches"],
                    diff_cache_mode=settings["diff_cache_mode"],
                    diff_cache_ram_pct=settings["diff_cache_ram_pct"],
                    record_dataset=settings["record_dataset"],
                    evaluator=settings.get("evaluator"),
                    memory_mode=settings.get("memory_mode", "disabled"),
                    selection=settings.get("selection", 1),
                )
                # Map 6-value AutoTuner return to 5-value Simple return
                # (model, clip, report, analysis_report, tuner_data, lora_data)
                # → (model, clip, combined_report, tuner_data, lora_data)
                at_model, at_clip, report, analysis_report, at_tuner_data, at_lora_data = result
                combined_report = report
                if analysis_report:
                    combined_report = f"{report}\n\n{'=' * 50}\nANALYSIS REPORT\n{'=' * 50}\n{analysis_report}"
                return (at_model, at_clip, combined_report, at_tuner_data, at_lora_data)
            else:
                raise ValueError(
                    f"[LoRA Optimizer] Unknown settings mode: {mode!r}. "
                    f"Expected 'advanced' or 'autotuner'."
                )

        # Priority 2: tuner_data connected (no settings)
        if tuner_data is not None and "top_n" in tuner_data and len(tuner_data["top_n"]) > 0:
            entry = tuner_data["top_n"][0]
            config = entry["config"]
            strategy_override = config["merge_mode"] if config["optimization_mode"] == "global" else ""
            return super().optimize_merge(
                model, lora_stack, output_strength,
                clip=clip, clip_strength_multiplier=clip_strength_multiplier,
                auto_strength=config["auto_strength"],
                auto_strength_floor=tuner_data.get("auto_strength_floor", -1.0),
                optimization_mode=config["optimization_mode"],
                sparsification=config["sparsification"],
                sparsification_density=config["sparsification_density"],
                dare_dampening=config["dare_dampening"],
                merge_strategy_override=strategy_override,
                merge_refinement=config["merge_refinement"],
                strategy_set=config.get("strategy_set", "full"),
                normalize_keys=tuner_data.get("normalize_keys", "enabled"),
                architecture_preset=tuner_data.get("architecture_preset", "auto"),
                decision_smoothing=tuner_data.get("decision_smoothing", 0.25),
                cache_patches="enabled",
                patch_compression="smart",
                svd_device="gpu",
                vram_budget=0.0,
            )

        # Priority 3: simple defaults
        return super().optimize_merge(
            model, lora_stack, output_strength,
            clip=clip, clip_strength_multiplier=clip_strength_multiplier,
            **self._SIMPLE_DEFAULTS,
        )

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength,
                   clip=None, clip_strength_multiplier=1.0, tuner_data=None,
                   settings=None):
        base = LoRAOptimizer.IS_CHANGED(
            model, lora_stack, output_strength,
            clip=clip, clip_strength_multiplier=clip_strength_multiplier,
            **cls._SIMPLE_DEFAULTS,
        )
        if settings is not None:
            import hashlib, json
            settings_hash = hashlib.md5(
                json.dumps(settings, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
            return f"{base}|settings={settings_hash}"
        if tuner_data is not None:
            return f"{base}|td={id(tuner_data)}"
        return base


class LoRAAutoTuner(LoRAOptimizer):
    """
    Automatic parameter sweep that ranks merge configurations for a given
    LoRA stack. Runs Pass 1 analysis once, scores all parameter combinations
    via heuristic, then merges top-N candidates and measures output quality.
    Outputs the top-ranked result as MODEL/CLIP, plus a ranked report and
    TUNER_DATA for optional override via a Merge Selector node.
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
                "scoring_svd": (["disabled", "merge_quality", "lora_rank", "full"], {
                    "default": "disabled",
                    "tooltip": "SVD-based scoring for ranking configurations.\n"
                               "disabled: fast norm-only scoring (usually sufficient).\n"
                               "merge_quality: SVD on merged diff tensors — more thorough quality measurement.\n"
                               "lora_rank: effective rank of LoRA factors — experimental, changes ranking.\n"
                               "full: both merge_quality + lora_rank."
                }),
                "scoring_device": (["cpu", "gpu"], {
                    "default": "gpu",
                    "tooltip": "Device for scoring computations. GPU is much faster with SVD scoring modes."
                }),
                "architecture_preset": (["auto", "sd_unet", "dit", "llm"], {
                    "default": "auto",
                    "tooltip": "Architecture-aware threshold tuning. 'auto' detects from LoRA keys."
                }),
                "auto_strength_floor": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum auto-strength scale factor for orthogonal LoRAs. -1 = architecture-aware default. Increase toward 1.0 to preserve more independent LoRA energy."
                }),
                "evaluator": ("AUTOTUNER_EVALUATOR", {
                    "tooltip": "Optional external evaluator spec. Use this to blend prompt/reference scoring from your own generation code with the built-in merge metrics."
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
                               "• fast — Every 2nd prefix (~50%% faster). Good default for most merges.\n"
                               "• turbo — Every 3rd prefix (~67%% faster). Works well when your LoRAs have similar conflict across blocks (e.g. multiple characters from the same trainer).\n"
                               "• turbo+ — Every 4th prefix (~75%% faster). Best for large models (DiT/Flux/WAN) or when iterating quickly. May miss subtle block-level differences on SD/SDXL."
                }),
                "scoring_formula": (["v2", "v1"], {
                    "default": "v2",
                    "tooltip": "Phase 2 scoring formula. v2: arch-aware sparsity + energy metrics (recommended). v1: legacy scoring with fixed 40% sparsity target."
                }),
                "output_mode": (["merge", "tuning_only"], {
                    "default": "merge",
                    "tooltip": "merge: output the top-ranked merged model. tuning_only: skip the final merge and pass the base model through so a downstream optimizer can apply the selected AutoTuner settings."
                }),
                "decision_smoothing": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Smooth per-group decision metrics toward each block average before candidate ranking and final merge. 0 disables smoothing."
                }),
                "smooth_slerp_gate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, uses smoothed cosine (decision_cosine) for SLERP gate instead of raw avg_cos_sim. Can affect SLERP/weighted_average ratio."
                }),
                "memory_mode": (["disabled", "auto", "read_only", "clear_and_run"], {
                    "default": "auto",
                    "tooltip": "Persistent memory for tuning results across sessions.\n"
                               "auto: Load cached results if available, save after tuning.\n"
                               "read_only: Use cached results but don't save new ones.\n"
                               "clear_and_run: Delete cached entry and re-tune from scratch.\n\n"
                               "Cache key uses LoRA names + strengths (order-independent) and tuning settings. "
                               "Does not track LoRA file contents — if you retrain a LoRA with the same filename, use clear_and_run."
                }),
                "selection": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Which ranked configuration to apply (1 = top-ranked). "
                               "Change this to try a different config without re-running the full sweep."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "TUNER_DATA", "LORA_DATA")
    RETURN_NAMES = ("model", "clip", "report", "analysis_report", "tuner_data", "lora_data")
    FUNCTION = "auto_tune"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Automatically sweeps all merge parameters and ranks "
        "configurations for your LoRA stack. Outputs the top-ranked merge "
        "directly. Connect TUNER_DATA to a Merge Selector node to try alternatives."
    )

    # --- Persistent memory helpers ---

    @staticmethod
    def _memory_settings_hash(settings):
        """SHA256[:16] of JSON-serialized tuning-relevant settings."""
        raw = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _memory_file_path(lora_hash, settings_hash):
        """Build full path for a memory file."""
        return os.path.join(AUTOTUNER_MEMORY_DIR,
                            f"{lora_hash}_{settings_hash}.memory.json")

    @staticmethod
    def _memory_load(lora_hash, settings_hash, requested_top_n):
        """Load and validate a memory entry. Returns tuner_data or None."""
        path = LoRAAutoTuner._memory_file_path(lora_hash, settings_hash)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            logging.warning(f"[AutoTuner Memory] Corrupt memory file, ignoring: {path}")
            return None

        # Version check
        if data.get("algo_version") != AUTOTUNER_ALGO_VERSION:
            logging.info(f"[AutoTuner Memory] Stale algo version "
                         f"({data.get('algo_version')} != {AUTOTUNER_ALGO_VERSION}), ignoring")
            return None
        if data.get("memory_version") != AUTOTUNER_MEMORY_VERSION:
            logging.info("[AutoTuner Memory] Stale memory version, ignoring")
            return None

        tuner_data = data.get("tuner_data")
        if not tuner_data or "top_n" not in tuner_data:
            return None

        # top_n count check — stored must have enough entries
        if len(tuner_data["top_n"]) < requested_top_n:
            logging.info(f"[AutoTuner Memory] Stored top_n={len(tuner_data['top_n'])} "
                         f"< requested={requested_top_n}, ignoring")
            return None

        return tuner_data

    @staticmethod
    def _memory_save(lora_hash, settings_hash, settings, source_loras, tuner_data):
        """Atomic write of memory entry to disk."""
        from datetime import datetime
        path = LoRAAutoTuner._memory_file_path(lora_hash, settings_hash)
        entry = {
            "memory_version": AUTOTUNER_MEMORY_VERSION,
            "algo_version": AUTOTUNER_ALGO_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "lora_hash": lora_hash,
            "settings_hash": settings_hash,
            "settings": settings,
            "source_loras": [{"name": l["name"], "strength": l["strength"]}
                             for l in source_loras],
            "tuner_data": tuner_data,
        }
        try:
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(entry, f, indent=2)
            os.replace(tmp_path, path)
            logging.info(f"[AutoTuner Memory] Saved: {path}")
        except Exception as e:
            logging.warning(f"[AutoTuner Memory] Failed to save: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _memory_clear(lora_hash=None, settings_hash=None):
        """Delete memory files. Both None = clear all. lora_hash only = all for that combo."""
        if lora_hash and settings_hash:
            path = LoRAAutoTuner._memory_file_path(lora_hash, settings_hash)
            if os.path.exists(path):
                os.unlink(path)
                logging.info(f"[AutoTuner Memory] Deleted: {path}")
        elif lora_hash:
            pattern = os.path.join(AUTOTUNER_MEMORY_DIR, f"{lora_hash}_*.memory.json")
            for path in glob.glob(pattern):
                os.unlink(path)
                logging.info(f"[AutoTuner Memory] Deleted: {path}")
        else:
            pattern = os.path.join(AUTOTUNER_MEMORY_DIR, "*.memory.json")
            for path in glob.glob(pattern):
                os.unlink(path)
            logging.info("[AutoTuner Memory] Cleared all memory files")

    def _build_memory_hit_report(self, lora_hash, tuner_data, output_strength,
                                 scoring_speed="full", applied_rank=1):
        """Build report for a memory cache hit."""
        banner = (
            "=" * 54 + "\n"
            "  MEMORY HIT — Results loaded from persistent cache\n"
            f"  LoRA hash: {lora_hash}\n"
            "  Set memory_mode='clear_and_run' to re-tune.\n"
            "=" * 54 + "\n\n"
        )
        results = tuner_data["top_n"]
        report = self._build_autotuner_report(
            results, tuner_data["analysis_summary"], output_strength,
            scoring_speed=scoring_speed, applied_rank=applied_rank)
        return banner + report

    def auto_tune(self, model, lora_stack, output_strength, clip=None,
                  clip_strength_multiplier=1.0, top_n=3, normalize_keys="disabled",
                  scoring_svd="disabled",
                  scoring_device="gpu",
                  architecture_preset="auto", auto_strength_floor=-1.0, evaluator=None,
                  record_dataset="disabled",
                  cache_patches="enabled",
                  diff_cache_mode="disabled", diff_cache_ram_pct=0.5, vram_budget=0.0,
                  scoring_speed="full", scoring_formula="v2", output_mode="merge",
                  decision_smoothing=0.25, smooth_slerp_gate=False,
                  memory_mode="disabled", selection=1,
                  _is_sub_merge=False, _suppress_pbar=False):
        import hashlib, json

        # Free stale cached models when the input model changes
        current_mid = id(model) if model is not None else None
        prev_mid = getattr(self, '_cached_model_id', None)
        if prev_mid is not None and current_mid != prev_mid:
            if hasattr(self, '_merge_cache') and self._merge_cache:
                self._merge_cache.clear()
            if hasattr(self, '_autotuner_cache') and getattr(self, '_autotuner_cache', None):
                self._autotuner_cache.clear()
            gc.collect()
        self._cached_model_id = current_mid

        # --- Extract merge formula before normalization ---
        merge_formula = None
        clean_stack = []
        for item in lora_stack:
            if isinstance(item, dict) and "_merge_formula" in item:
                merge_formula = item["_merge_formula"]
            else:
                clean_stack.append(item)
        if merge_formula:
            lora_stack = clean_stack

        # --- Normalize & validate stack ---
        normalized_stack = self._normalize_stack(lora_stack, normalize_keys=normalize_keys)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]
        if not active_loras:
            return (model, clip, "No active LoRAs in stack.", "", None, None)

        # --- Formula-based hierarchical auto-tune ---
        if merge_formula and len(active_loras) >= 2:
            try:
                tree = _parse_merge_formula(merge_formula, len(normalized_stack))
            except ValueError as e:
                logging.warning(f"[LoRA AutoTuner] Invalid merge formula: {e} — using flat auto-tune")
                tree = None

            if tree is not None and tree["type"] == "group":
                logging.info(f"[LoRA AutoTuner] Using merge formula: {merge_formula}")

                # Resolve architecture preset before sub-merges
                preset_key, _ = _resolve_arch_preset(
                    architecture_preset, getattr(self, '_detected_arch', None) or 'unknown')

                at_kwargs = {
                    "clip_strength_multiplier": clip_strength_multiplier,
                    "top_n": top_n,
                    "normalize_keys": normalize_keys,
                    "scoring_svd": scoring_svd,
                    "scoring_device": scoring_device,
                    "architecture_preset": preset_key,
                    "auto_strength_floor": auto_strength_floor,
                    "decision_smoothing": decision_smoothing,
                    "smooth_slerp_gate": smooth_slerp_gate,
                    "vram_budget": vram_budget,
                    "scoring_speed": scoring_speed,
                    "scoring_formula": scoring_formula,
                    "diff_cache_mode": diff_cache_mode,
                    "diff_cache_ram_pct": diff_cache_ram_pct,
                }

                # Save _detected_arch — sub-merges may overwrite it
                # when all resolved items are virtual (no arch detection possible)
                saved_arch = getattr(self, '_detected_arch', None)

                resolved_stack, sub_reports = self._autotune_resolve_tree(
                    tree, normalized_stack, model, clip, **at_kwargs)

                if len(resolved_stack) >= 2:

                    # Run outer auto_tune on the resolved flat stack (no formula).
                    # normalize_keys="disabled": stack is already normalized.
                    outer_result = self.auto_tune(
                        model, resolved_stack, output_strength,
                        clip=clip,
                        clip_strength_multiplier=clip_strength_multiplier,
                        top_n=top_n,
                        normalize_keys="disabled",
                        scoring_svd=scoring_svd,
                        scoring_device=scoring_device,
                        architecture_preset=preset_key,
                        auto_strength_floor=auto_strength_floor,
                        evaluator=evaluator,
                        record_dataset=record_dataset,
                        cache_patches=cache_patches,
                        diff_cache_mode=diff_cache_mode,
                        diff_cache_ram_pct=diff_cache_ram_pct,
                        vram_budget=vram_budget,
                        scoring_speed=scoring_speed,
                        scoring_formula=scoring_formula,
                        output_mode=output_mode,
                        decision_smoothing=decision_smoothing,
                        smooth_slerp_gate=smooth_slerp_gate,
                        memory_mode=memory_mode,
                        selection=selection,
                        _is_sub_merge=_is_sub_merge,
                        _suppress_pbar=_suppress_pbar,
                    )

                    # Restore _detected_arch
                    self._detected_arch = saved_arch

                    # Prepend sub-reports to the outer report
                    if sub_reports:
                        # outer_result is 6-tuple
                        ret_model, ret_clip, report, analysis_report, tuner_data, lora_data = outer_result
                        separator = "\n" + "=" * 50 + "\n"
                        sub_section = separator.join(sub_reports)
                        report = (
                            "AUTOTUNER FORMULA SUB-MERGE REPORTS\n"
                            + separator + sub_section + separator
                            + "\nFINAL AUTOTUNER REPORT:\n" + report
                        )
                        outer_result = (ret_model, ret_clip, report, analysis_report, tuner_data, lora_data)

                    return outer_result
                elif len(resolved_stack) == 1:
                    # All sub-merges collapsed to one — update state and fall through
                    logging.info("[LoRA AutoTuner] Formula resolved to single LoRA — skipping outer tune")
                    self._detected_arch = saved_arch
                    normalized_stack = resolved_stack
                    active_loras = [item for item in normalized_stack if item["strength"] != 0]
                    # Fall through to single-LoRA or normal path below

        if len(active_loras) == 1:
            # Single LoRA: nothing to tune, delegate directly
            if output_mode == "tuning_only":
                return (model, clip, "Single LoRA detected -- tuning_only passthrough.", "", None, None)
            merged_model, merged_clip, report, _, lora_data = super().optimize_merge(
                model, normalized_stack, output_strength,
                clip=clip, clip_strength_multiplier=clip_strength_multiplier,
                normalize_keys=normalize_keys, strategy_set="full",
                architecture_preset=preset_key if merge_formula else architecture_preset, vram_budget=vram_budget,
                auto_strength_floor=auto_strength_floor,
                decision_smoothing=decision_smoothing,
                smooth_slerp_gate=smooth_slerp_gate,
                _skip_qkv_refusion=_is_sub_merge,
            )
            return (merged_model, merged_clip,
                    "Single LoRA detected -- no parameters to tune.\n\n" + report, report, None, lora_data)

        # Compute lora_hash for cache validation
        hash_input = json.dumps([(l["name"], l["strength"]) for l in active_loras],
                                sort_keys=True)
        lora_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Order-independent hash for persistent memory (sorted pairs)
        if memory_mode != "disabled" and not _is_sub_merge:
            memory_lora_hash = hashlib.sha256(
                json.dumps(sorted([(l["name"], l["strength"]) for l in active_loras]),
                           separators=(",", ":")).encode()
            ).hexdigest()[:16]

        evaluator_hash = self._stable_data_hash(evaluator) if evaluator is not None else ""

        # Check AutoTuner cache
        at_cache_key = hashlib.sha256(
            f"{lora_hash}|os={output_strength}|csm={clip_strength_multiplier}"
            f"|top_n={top_n}|nk={normalize_keys}|ss={scoring_svd}"
            f"|ap={architecture_preset}|vb={vram_budget}"
            f"|spd={scoring_speed}|mid={id(model)}|asf={auto_strength_floor}|ds={decision_smoothing}|eh={evaluator_hash}"
            f"|mm={memory_mode}|sel={selection}".encode()
        ).hexdigest()[:16]
        if cache_patches == "enabled" and hasattr(self, '_autotuner_cache') and at_cache_key in self._autotuner_cache:
            cached_result, cached_mode = self._autotuner_cache[at_cache_key]
            if output_mode == "tuning_only":
                logging.info("[LoRA AutoTuner] Using cached result (tuning_only passthrough)")
                return (model, clip, cached_result[2], "", cached_result[4], None)
            if cached_mode == "merge":
                logging.info("[LoRA AutoTuner] Using cached result")
                return cached_result

        # --- Persistent memory lookup ---
        if memory_mode != "disabled" and not _is_sub_merge:
            memory_settings = {
                "normalize_keys": normalize_keys,
                "scoring_svd": scoring_svd,
                "scoring_device": scoring_device,
                "architecture_preset": architecture_preset,
                "auto_strength_floor": auto_strength_floor,
                "scoring_speed": scoring_speed,
                "scoring_formula": scoring_formula,
                "decision_smoothing": decision_smoothing,
                "smooth_slerp_gate": smooth_slerp_gate,
                "evaluator_hash": evaluator_hash,
            }
            settings_hash = self._memory_settings_hash(memory_settings)

            if memory_mode == "clear_and_run":
                self._memory_clear(memory_lora_hash, settings_hash)

            if memory_mode in ("auto", "read_only"):
                cached_tuner_data = self._memory_load(
                    memory_lora_hash, settings_hash, top_n)
                if cached_tuner_data is not None:
                    logging.info("[AutoTuner Memory] HIT — loading cached tuning results")
                    # Truncate top_n if needed
                    if len(cached_tuner_data["top_n"]) > top_n:
                        cached_tuner_data["top_n"] = cached_tuner_data["top_n"][:top_n]

                    sel_idx = min(selection, len(cached_tuner_data["top_n"])) - 1
                    report = self._build_memory_hit_report(
                        memory_lora_hash, cached_tuner_data, output_strength,
                        scoring_speed=scoring_speed, applied_rank=sel_idx + 1)

                    if output_mode == "tuning_only":
                        result = (model, clip, report, "", cached_tuner_data, None)
                        if cache_patches == "enabled":
                            if not hasattr(self, '_autotuner_cache'):
                                self._autotuner_cache = {}
                            self._autotuner_cache[at_cache_key] = (result, "tuning_only")
                        return result

                    # Replay the selected config via optimize_merge
                    config = cached_tuner_data["top_n"][sel_idx]["config"]
                    strategy_override = (config["merge_mode"]
                                         if config["optimization_mode"] == "global" else "")
                    merged_model, merged_clip, _replay_report, _, lora_data = super().optimize_merge(
                        model, lora_stack, output_strength,
                        clip=clip,
                        clip_strength_multiplier=clip_strength_multiplier,
                        auto_strength=config["auto_strength"],
                        auto_strength_floor=cached_tuner_data.get(
                            "auto_strength_floor", auto_strength_floor),
                        optimization_mode=config["optimization_mode"],
                        sparsification=config["sparsification"],
                        sparsification_density=config["sparsification_density"],
                        dare_dampening=config["dare_dampening"],
                        merge_refinement=config["merge_refinement"],
                        merge_strategy_override=strategy_override,
                        strategy_set=config.get("strategy_set", "full"),
                        normalize_keys=cached_tuner_data.get(
                            "normalize_keys", normalize_keys),
                        architecture_preset=cached_tuner_data.get(
                            "architecture_preset", architecture_preset),
                        decision_smoothing=cached_tuner_data.get(
                            "decision_smoothing", decision_smoothing),
                        smooth_slerp_gate=smooth_slerp_gate,
                        cache_patches=cache_patches,
                        vram_budget=vram_budget,
                    )

                    result = (merged_model, merged_clip, report,
                              _replay_report, cached_tuner_data, lora_data)
                    if cache_patches == "enabled":
                        if not hasattr(self, '_autotuner_cache'):
                            self._autotuner_cache = {}
                        self._autotuner_cache[at_cache_key] = (result, "merge")
                    return result

        # --- Pass 1: Analysis (run once, reuse for all configs) ---
        model_keys = self._get_model_keys(model)
        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})

        all_lora_prefixes = self._collect_lora_prefixes(active_loras)
        target_groups = self._build_target_groups(all_lora_prefixes, model_keys, clip_keys)
        if not target_groups:
            return (model, clip, "No compatible LoRA target groups found.", "", None, None)

        # Determine compute device
        compute_device = self._get_compute_device()
        use_gpu = compute_device.type != "cpu"

        logging.info(f"[LoRA AutoTuner] Pass 1: Analyzing {len(target_groups)} target groups...")
        t_start = time.time()
        # Progress bar: analysis groups + top_n merges (+ 1 final merge when subsampling)
        n_pbar_merges = top_n + (1 if scoring_speed != "full" and top_n > 1 and output_mode != "tuning_only" else 0)
        if _suppress_pbar:
            class _NullPbar:
                def update(self, n): pass
            pbar = _NullPbar()
        else:
            pbar = comfy.utils.ProgressBar(len(target_groups) + n_pbar_merges)
        analysis_data = self._run_group_analysis(
            target_groups, active_loras, model, clip, compute_device,
            clip_strength_multiplier=clip_strength_multiplier,
            merge_refinement="none",
            decision_smoothing=decision_smoothing,
            progress_cb=lambda: pbar.update(1),
        )
        all_key_targets = analysis_data["all_key_targets"]
        target_groups = analysis_data["target_groups"]
        prefix_stats = analysis_data["prefix_stats"]
        per_lora_stats = analysis_data["per_lora_stats"]
        pair_accum = analysis_data["pair_accum"]
        branch_energy = analysis_data["branch_energy"]
        all_magnitude_samples = analysis_data["all_magnitude_samples"]
        prefix_count = analysis_data["prefix_count"]
        skipped_keys = analysis_data["skipped_keys"]
        pairs = analysis_data["pairs"]

        if prefix_count == 0:
            return (model, clip, "No compatible LoRA keys found.", "", None, None)

        t_analysis = time.time() - t_start
        logging.info(f"[LoRA AutoTuner] Analysis complete: {prefix_count} target groups ({t_analysis:.1f}s)")

        # Finalize global stats
        l2_means = []
        for i, stat in enumerate(per_lora_stats):
            l2_mean = sum(stat["l2_norms"]) / len(stat["l2_norms"]) if stat["l2_norms"] else 0
            l2_means.append(l2_mean)

        total_overlap = sum(pair_accum[p]["overlap"] for p in pairs)
        total_conflict = sum(pair_accum[p]["conflict"] for p in pairs)
        total_weighted_total = sum(pair_accum[p]["weighted_total"] for p in pairs)
        total_weighted_conflict = sum(pair_accum[p]["weighted_conflict"] for p in pairs)
        total_expected_conflict_weighted = sum(pair_accum[p]["expected_conflict_weighted"] for p in pairs)
        total_excess_conflict_weighted = sum(pair_accum[p]["excess_conflict_weighted"] for p in pairs)
        total_subspace_num = sum(pair_accum[p]["subspace_num"] for p in pairs)
        total_subspace_den = sum(pair_accum[p]["subspace_den"] for p in pairs)
        avg_conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0
        avg_excess_conflict = total_excess_conflict_weighted / total_weighted_total if total_weighted_total > 0 else 0.0
        avg_subspace_overlap = total_subspace_num / total_subspace_den if total_subspace_den > 0 else 0.0

        pairwise_similarities = {}
        for i, j in pairs:
            dot = pair_accum[(i, j)]["dot"]
            na_sq = pair_accum[(i, j)]["norm_a_sq"]
            nb_sq = pair_accum[(i, j)]["norm_b_sq"]
            denom = math.sqrt(na_sq) * math.sqrt(nb_sq)
            pairwise_similarities[(i, j)] = dot / denom if denom > 0 else 0.0

        branch_measure = branch_energy["model"]["norm_sq"]
        model_effective = [
            abs(active_loras[i]["strength"]) * math.sqrt(max(branch_measure[i], 0.0))
            for i in range(len(active_loras))
        ]
        valid_l2 = [m for m in model_effective if m > 0]
        magnitude_ratio = max(valid_l2) / min(valid_l2) if len(valid_l2) >= 2 else 1.0

        avg_cos_sim = (sum(pairwise_similarities.values())
                       / len(pairwise_similarities)) if pairwise_similarities else 0.0
        analysis_summary = {
            "n_loras": len(active_loras),
            "prefix_count": prefix_count,
            "avg_conflict_ratio": avg_conflict_ratio,
            "avg_excess_conflict": avg_excess_conflict,
            "avg_subspace_overlap": avg_subspace_overlap,
            "avg_cosine_sim": avg_cos_sim,
            "magnitude_ratio": magnitude_ratio,
            "decision_smoothing": decision_smoothing,
        }

        _analysis_cache = {
            "all_key_targets": all_key_targets,
            "target_groups": target_groups,
            "prefix_stats": prefix_stats,
            "per_lora_stats": per_lora_stats,
            "pair_accum": pair_accum,
            "branch_energy": branch_energy,
            "all_magnitude_samples": all_magnitude_samples,
            "prefix_count": prefix_count,
            "skipped_keys": skipped_keys,
            "pairs": pairs,
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
                magnitude_ratio, prefix_stats, arch_preset=tuner_arch_preset,
                avg_excess_conflict=avg_excess_conflict,
                avg_subspace_overlap=avg_subspace_overlap)
            scored.append((h_score, config))
        scored.sort(key=lambda x: x[0], reverse=True)
        logging.info(f"[LoRA AutoTuner] Scored {len(grid)} combos in {time.time() - t_start:.1f}s")
        for i in range(min(5, len(scored))):
            c = scored[i][1]
            strat_info = f" [{c.get('strategy_set', 'full')}]" if c['optimization_mode'] == 'per_prefix' else ""
            logging.info(f"[LoRA AutoTuner]   #{i+1} heuristic={scored[i][0]:.3f}: "
                         f"{c['merge_mode']}/{c['merge_refinement']}"
                         f"{' +' + c['sparsification'] if c['sparsification'] != 'disabled' else ''}"
                         f" auto_str={c['auto_strength']} {c['optimization_mode']}{strat_info}")

        # Pre-compute per-group density from magnitude samples before freeing them.
        # Phase 2's per-group auto_select_params needs density but magnitude_samples
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
            # Separate single-LoRA groups (always included, identical across candidates)
            # from multi-LoRA groups (where merge strategy matters)
            single_lora_targets = {}
            multi_lora_targets = {}
            for pfx, tgt in all_key_targets.items():
                if prefix_stats.get(pfx, {}).get("n_loras", 0) <= 1:
                    single_lora_targets[pfx] = tgt
                else:
                    multi_lora_targets[pfx] = tgt
            # Sort multi-LoRA groups by conflict_ratio descending (highest conflict first)
            sorted_multi = sorted(
                multi_lora_targets.keys(),
                key=lambda p: prefix_stats.get(p, {}).get("decision_conflict", prefix_stats.get(p, {}).get("conflict_ratio", 0.0)),
                reverse=True,
            )
            # Keep every Nth group from the sorted list
            subsampled_multi = {p: multi_lora_targets[p] for p in sorted_multi[::step]}
            scoring_targets = {**single_lora_targets, **subsampled_multi}
            scoring_target_groups = {
                label: target_groups[label]
                for label in scoring_targets.keys()
                if label in target_groups
            }
            scoring_cache = {
                **_analysis_cache,
                "all_key_targets": scoring_targets,
                "target_groups": scoring_target_groups,
                "prefix_count": len(scoring_target_groups),
            }
            logging.info(f"[LoRA AutoTuner] Scoring speed '{scoring_speed}': "
                         f"scoring {len(scoring_targets)}/{len(all_key_targets)} groups "
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
                         f"{config['merge_mode']}, {config['merge_refinement']}"
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
                merge_refinement=config["merge_refinement"],
                merge_strategy_override=strategy_override,
                free_vram_between_passes="disabled",
                vram_budget=vram_budget,
                cache_patches="disabled",
                patch_compression="disabled",
                svd_device="gpu",
                normalize_keys=normalize_keys,
                strategy_set=config.get("strategy_set", "full"),
                architecture_preset=architecture_preset,
                decision_smoothing=decision_smoothing,
                smooth_slerp_gate=smooth_slerp_gate,
                _analysis_cache=scoring_cache,
                _diff_cache=_diff_cache,
                _skip_report=True,
                _skip_qkv_refusion=_is_sub_merge,
            )

            # Measure output quality (single-LoRA prefixes may still produce
            # LoRAAdapter patches; _score_merge_result handles both formats)
            m_patches = lora_data["model_patches"] if lora_data else {}
            c_patches = lora_data["clip_patches"] if lora_data else {}
            is_ortho_score = (
                abs(avg_cos_sim) < tuner_arch_preset["orthogonal_cos_sim_max"]
                and avg_subspace_overlap < 0.35
            )
            # Skip SVD for orthogonal LoRAs — SLERP vs average produce different
            # rank profiles that don't correlate with generation quality
            compute_svd = scoring_svd in ("merge_quality", "full") and not is_ortho_score
            compute_lora_svd = scoring_svd in ("lora_rank", "full")
            score_dev = torch.device("cuda") if scoring_device == "gpu" and torch.cuda.is_available() else None
            t_score = time.time()
            score_arch = tuner_arch_preset if scoring_formula == "v2" else None
            measured = _score_merge_result(
                m_patches, c_patches, compute_svd=compute_svd,
                score_device=score_dev, arch_preset=score_arch,
                lora_svd=compute_lora_svd
            )
            t_score_elapsed = time.time() - t_score
            # --- Post-scoring adjustments ---
            if scoring_formula == "v1":
                # Legacy scoring: conditional sparsity discount, old composite weights
                if config["sparsification"] != "disabled":
                    measured["sparsity_fit"] *= 0.5
                cv_s = max(0.0, 1.0 - measured["norm_cv"])
                if measured.get("effective_rank_mean", 0) > 0:
                    rank_s = min(measured["effective_rank_mean"] / 40.0, 1.0)
                    measured["composite_score"] = (
                        rank_s * 0.4 + cv_s * 0.3 + measured["sparsity_fit"] * 0.3
                    )
                else:
                    measured["composite_score"] = (
                        cv_s * 0.5 + measured["sparsity_fit"] * 0.5
                    )
            else:
                # v2: energy-aware scoring with arch-aware sparsity baseline
                # Compute energy_preservation from branch_energy (model + clip).
                # Baseline = weighted_average expected energy (not weighted_sum).
                # Auto-strength cancels in WA normalization, so same formula for all.
                measured_energy_sq = measured.get("norm_energy_sq", 0.0)
                measured_energy = math.sqrt(max(measured_energy_sq, 0.0))
                model_strengths = [item["strength"] for item in active_loras]
                clip_strengths = [
                    item["clip_strength"] if item["clip_strength"] is not None else item["strength"]
                    for item in active_loras
                ]
                total_model_w = sum(abs(s) for s in model_strengths)
                total_clip_w = sum(abs(s) for s in clip_strengths)
                if is_ortho_score:
                    # Orthogonal: Pythagorean — no cross-terms
                    expected_model_sq = sum(
                        model_strengths[i] ** 2 * branch_energy["model"]["norm_sq"][i]
                        for i in range(len(active_loras))
                    )
                    expected_clip_sq = sum(
                        clip_strengths[i] ** 2 * branch_energy["clip"]["norm_sq"][i]
                        for i in range(len(active_loras))
                    )
                else:
                    # Non-orthogonal: include cross-terms
                    expected_model_sq = sum(
                        model_strengths[i] ** 2 * branch_energy["model"]["norm_sq"][i]
                        for i in range(len(active_loras))
                    )
                    for (i, j), dot in branch_energy["model"]["dot"].items():
                        expected_model_sq += 2.0 * model_strengths[i] * model_strengths[j] * dot
                    expected_clip_sq = sum(
                        clip_strengths[i] ** 2 * branch_energy["clip"]["norm_sq"][i]
                        for i in range(len(active_loras))
                    )
                    for (i, j), dot in branch_energy["clip"]["dot"].items():
                        expected_clip_sq += 2.0 * clip_strengths[i] * clip_strengths[j] * dot
                # Divide by total_weight^2 per branch for weighted_average baseline
                if total_model_w > 0:
                    expected_model_sq = max(expected_model_sq, 0.0) / (total_model_w ** 2)
                if total_clip_w > 0:
                    expected_clip_sq = max(expected_clip_sq, 0.0) / (total_clip_w ** 2)
                expected_energy = math.sqrt(expected_model_sq + expected_clip_sq)
                # Scale expected energy by prefix fraction when subsampling
                if use_subsampling and prefix_count > 0:
                    prefix_fraction = len(scoring_cache.get("all_key_targets", all_key_targets)) / len(all_key_targets)
                    expected_energy *= math.sqrt(prefix_fraction)
                energy_ratio = measured_energy / expected_energy if expected_energy > 0 else 1.0
                # One-sided: only penalize energy loss below WA baseline (ratio < 1)
                # SLERP legitimately boosts energy above WA — don't penalize that
                energy_preservation = min(energy_ratio, 1.0)
                measured["energy_ratio"] = energy_ratio
                measured["energy_preservation"] = energy_preservation
                # Discount sparsity_fit when sparsification artificially inflates it
                if config["sparsification"] != "disabled":
                    measured["sparsity_fit"] *= 0.5
                # Recompute composite score
                cv_s = max(0.0, 1.0 - measured["norm_cv"])
                if measured.get("effective_rank_mean", 0) > 0:
                    # Non-orthogonal (with SVD): energy helps detect destructive merges
                    rank_s = min(measured["effective_rank_mean"] / 40.0, 1.0)
                    measured["composite_score"] = (
                        rank_s * 0.30 + cv_s * 0.25
                        + energy_preservation * 0.20 + measured["sparsity_fit"] * 0.25
                    )
                else:
                    # Orthogonal (without SVD): energy doesn't predict quality here
                    # (SLERP vs WA energy ≠ quality for orthogonal LoRAs).
                    # Arch-aware sparsity_fit is the primary discriminator.
                    measured["composite_score"] = (
                        cv_s * 0.50 + measured["sparsity_fit"] * 0.50
                    )
            external_eval = _run_autotuner_evaluator(
                evaluator, merged_model, merged_clip, lora_data, config, analysis_summary
            ) if evaluator else None
            external_score = external_eval.get("score") if external_eval else None
            combine_mode = evaluator.get("combine_mode", "blend") if evaluator else "blend"
            eval_weight = max(0.0, min(1.0, float(evaluator.get("weight", 0.5)))) if evaluator else 0.5
            final_score = measured["composite_score"]
            if external_score is not None:
                if combine_mode == "external_only":
                    final_score = external_score
                elif combine_mode == "multiply":
                    final_score = final_score * external_score
                else:
                    final_score = (1.0 - eval_weight) * final_score + eval_weight * external_score

            t_elapsed = time.time() - t_merge
            energy_log = f", energy={measured['energy_ratio']:.2f}x" if "energy_ratio" in measured else ""
            logging.info(f"[LoRA AutoTuner]   Candidate #{rank_idx + 1}: "
                         f"measured={measured['composite_score']:.3f}"
                         f"{energy_log}"
                         f"{f', external={external_score:.3f}' if external_score is not None else ''}"
                         f", final={final_score:.3f} "
                         f"(merge {t_elapsed - t_score_elapsed:.1f}s + score {t_score_elapsed:.1f}s)")
            pbar.update(1)

            if use_subsampling or output_mode == "tuning_only":
                # When subsampling, discard all candidates — final full merge comes after
                del merged_model, merged_clip, lora_data
            elif final_score > best_score:
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
            if final_score > best_score:
                best_score = final_score
                best_config = config
            del m_patches, c_patches  # Drop patch-dict references so tensors can free
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()

            results.append({
                "rank": rank_idx + 1,
                "score_heuristic": h_score,
                "score_measured": measured["composite_score"],
                "score_external": external_score,
                "score_final": final_score,
                "config": config,
                "metrics": {
                    "norm_preservation": measured.get("norm_mean", 0.0),
                    "effective_rank_mean": measured.get("effective_rank_mean", 0.0),
                    "sparsity_mean": measured.get("sparsity_mean", 0.0),
                    "norm_cv": measured.get("norm_cv", 0.0),
                    **({"energy_ratio": measured.get("energy_ratio", 1.0),
                        "energy_preservation": measured.get("energy_preservation", 0.5)}
                       if scoring_formula == "v2" else
                       {"importance_cv": measured.get("importance_cv", measured.get("norm_cv", 0.0))}),
                },
                "external_details": external_eval.get("details") if external_eval else None,
            })

        # Final full merge when subsampling was used
        # Skip if selection != 1 — the replay merge below will handle it
        if use_subsampling and best_config is not None and output_mode != "tuning_only" and selection == 1:
            logging.info(f"[LoRA AutoTuner] Final merge with winning config "
                         f"({best_config['merge_mode']}, {best_config['merge_refinement']})...")
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
                merge_refinement=best_config["merge_refinement"],
                merge_strategy_override=strategy_override,
                free_vram_between_passes="disabled",
                vram_budget=vram_budget,
                cache_patches="disabled",
                patch_compression="disabled",
                svd_device="gpu",
                normalize_keys=normalize_keys,
                strategy_set=best_config.get("strategy_set", "full"),
                architecture_preset=architecture_preset,
                decision_smoothing=decision_smoothing,
                smooth_slerp_gate=smooth_slerp_gate,
                _analysis_cache=_analysis_cache,
                _diff_cache=_diff_cache,
                _skip_report=True,
                _skip_qkv_refusion=_is_sub_merge,
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

        # Sort by final score
        results.sort(key=lambda x: x["score_final"], reverse=True)
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
            "source_loras": [{"name": l["name"], "strength": l["strength"]} for l in active_loras],
            "normalize_keys": normalize_keys,
            "architecture_preset": architecture_preset,
            "auto_strength_floor": auto_strength_floor,
            "decision_smoothing": decision_smoothing,
            "analysis_summary": analysis_summary,
            "top_n": [{
                "rank": r["rank"],
                "score_heuristic": r["score_heuristic"],
                "score_measured": r["score_measured"],
                "score_external": r.get("score_external"),
                "score_final": r["score_final"],
                "config": r["config"],
                "metrics": r["metrics"],
                "external_details": r.get("external_details"),
            } for r in results],
        }

        # Save dataset entry for threshold tuning (opt-in)
        if record_dataset == "enabled":
            self._save_tuner_dataset_entry(
                tuner_data, active_loras, prefix_stats,
                getattr(self, '_detected_arch', None))
        prefix_stats.clear()

        # Save to persistent memory
        if memory_mode in ("auto", "clear_and_run") and not _is_sub_merge:
            memory_settings = {
                "normalize_keys": normalize_keys,
                "scoring_svd": scoring_svd,
                "scoring_device": scoring_device,
                "architecture_preset": architecture_preset,
                "auto_strength_floor": auto_strength_floor,
                "scoring_speed": scoring_speed,
                "scoring_formula": scoring_formula,
                "decision_smoothing": decision_smoothing,
                "smooth_slerp_gate": smooth_slerp_gate,
                "evaluator_hash": evaluator_hash,
            }
            settings_hash = self._memory_settings_hash(memory_settings)
            self._memory_save(memory_lora_hash, settings_hash,
                              memory_settings, active_loras, tuner_data)

        # Clamp selection to available results
        sel_idx = min(selection, len(results)) - 1

        # Build report
        suggested_max = best_lora_data.get("suggested_max_strength") if best_lora_data else None
        report = self._build_autotuner_report(
            results, tuner_data["analysis_summary"], output_strength,
            suggested_max_strength=suggested_max, scoring_speed=scoring_speed,
            applied_rank=sel_idx + 1)

        if output_mode == "tuning_only":
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
            if cache_patches == "enabled" and not _is_sub_merge:
                self._autotuner_cache = {at_cache_key: (result, "tuning_only")}
            elif not _is_sub_merge:
                self._autotuner_cache = {}
            return result

        # If selection != 1, replay-merge the selected config (rank-1 model is
        # the only one kept in memory during the sweep)
        if sel_idx > 0:
            sel_config = results[sel_idx]["config"]
            strategy_override = (sel_config["merge_mode"]
                                 if sel_config["optimization_mode"] == "global" else "")
            # Free the rank-1 model before replaying
            del best_model, best_clip, best_lora_data, best_analysis_report
            for r in results:
                r.pop("merged_model", None)
                r.pop("merged_clip", None)
                r.pop("lora_data", None)
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()
            logging.info(f"[LoRA AutoTuner] Replaying selected config #{sel_idx + 1}")
            ret_model, ret_clip, ret_analysis_report, _, ret_lora_data = super().optimize_merge(
                model, lora_stack, output_strength,
                clip=clip,
                clip_strength_multiplier=clip_strength_multiplier,
                auto_strength=sel_config["auto_strength"],
                auto_strength_floor=tuner_data.get("auto_strength_floor", auto_strength_floor),
                optimization_mode=sel_config["optimization_mode"],
                sparsification=sel_config["sparsification"],
                sparsification_density=sel_config["sparsification_density"],
                dare_dampening=sel_config["dare_dampening"],
                merge_refinement=sel_config["merge_refinement"],
                merge_strategy_override=strategy_override,
                strategy_set=sel_config.get("strategy_set", "full"),
                normalize_keys=tuner_data.get("normalize_keys", normalize_keys),
                architecture_preset=tuner_data.get("architecture_preset", architecture_preset),
                decision_smoothing=tuner_data.get("decision_smoothing", decision_smoothing),
                smooth_slerp_gate=smooth_slerp_gate,
                cache_patches=cache_patches,
                vram_budget=vram_budget,
            )
            del results, best, active_loras
        else:
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
        if cache_patches == "enabled" and not _is_sub_merge:
            self._autotuner_cache = {at_cache_key: (result, "merge")}
        elif not _is_sub_merge:
            self._autotuner_cache = {}
            logging.info("[LoRA AutoTuner] Patch cache disabled — RAM freed after merge")

        return result

    def _autotune_resolve_tree(self, tree, normalized_stack, model, clip, **at_kwargs):
        """
        Recursively resolve a merge formula tree, running auto_tune for each
        sub-group with 2+ items.  Returns (resolved_stack, sub_reports).
        Same tree format as _resolve_tree_to_stack but uses AutoTuner for sub-merges.
        """
        sub_reports = []

        if tree["type"] == "leaf":
            item = dict(normalized_stack[tree["index"]])
            if "metadata" in item and isinstance(item["metadata"], dict):
                item["metadata"] = dict(item["metadata"])
            if tree["weight"] is not None:
                item["strength"] = tree["weight"]
            return ([item], sub_reports)

        # Group: resolve each child
        resolved = []
        for child in tree["children"]:
            if child["type"] == "leaf":
                item = dict(normalized_stack[child["index"]])
                if "metadata" in item and isinstance(item["metadata"], dict):
                    item["metadata"] = dict(item["metadata"])
                if child["weight"] is not None:
                    item["strength"] = child["weight"]
                resolved.append(item)
            else:
                # Sub-group: resolve recursively then auto-tune
                sub_stack, child_reports = self._autotune_resolve_tree(
                    child, normalized_stack, model, clip, **at_kwargs)
                sub_reports.extend(child_reports)

                if len(sub_stack) >= 2:
                    try:
                        # Override settings for sub-merge
                        sub_kwargs = dict(at_kwargs)
                        sub_kwargs["cache_patches"] = "disabled"
                        sub_kwargs["record_dataset"] = "disabled"
                        sub_kwargs["output_mode"] = "merge"
                        sub_kwargs["_is_sub_merge"] = True
                        sub_kwargs["_suppress_pbar"] = True
                        # Evaluator is excluded: it may be prompt-specific and
                        # inappropriate for sub-groups (character-only merge etc.)

                        sub_result = self.auto_tune(
                            model, sub_stack, 1.0, clip=clip, **sub_kwargs)

                        # auto_tune returns 6-tuple
                        sub_model, sub_clip, sub_report, _, _, sub_lora_data = sub_result

                        sub_reports.append(sub_report)

                        if sub_lora_data is None:
                            # Fallback: pass items through
                            for sub_item in sub_stack:
                                item = dict(sub_item)
                                if child.get("weight") is not None:
                                    item["strength"] = child["weight"]
                                resolved.append(item)
                            continue

                        # Build virtual LoRA from sub-merge result
                        sub_model_patches = sub_lora_data.get("model_patches", {})
                        sub_clip_patches = sub_lora_data.get("clip_patches", {})
                        virtual = self._model_to_virtual_lora(
                            sub_model_patches, sub_clip_patches, child)
                        del sub_model, sub_clip, sub_result, sub_lora_data
                        if child["weight"] is not None:
                            virtual["strength"] = child["weight"]
                        resolved.append(virtual)
                    except Exception as e:
                        logging.warning(
                            f"[LoRA AutoTuner] Sub-merge auto_tune failed: {e} — "
                            "falling back to flat merge for this sub-group")
                        for item in sub_stack:
                            resolved.append(item)
                elif len(sub_stack) == 1:
                    item = sub_stack[0]
                    if child["weight"] is not None:
                        item["strength"] = child["weight"]
                    resolved.append(item)

        return (resolved, sub_reports)

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
            excess_conflicts = []
            cos_sims = []
            mag_ratios = []
            subspace_overlaps = []
            for pf in prefix_stats.values():
                if pf.get("n_loras", 0) > 1:
                    conflict_ratios.append(pf.get("decision_conflict", pf["conflict_ratio"]))
                    excess_conflicts.append(pf.get("excess_conflict", 0.0))
                    cos_sims.append(pf.get("avg_cos_sim", 0.0))
                    mag_ratios.append(pf.get("magnitude_ratio", 1.0))
                    subspace_overlaps.append(pf.get("avg_subspace_overlap", 0.0))

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
                    "excess_conflicts": {
                        "min": min(excess_conflicts) if excess_conflicts else 0,
                        "max": max(excess_conflicts) if excess_conflicts else 0,
                        "mean": sum(excess_conflicts) / len(excess_conflicts) if excess_conflicts else 0,
                        "n": len(excess_conflicts),
                    },
                    "cos_sims": {
                        "min": min(cos_sims) if cos_sims else 0,
                        "max": max(cos_sims) if cos_sims else 0,
                        "mean": sum(cos_sims) / len(cos_sims) if cos_sims else 0,
                        "n": len(cos_sims),
                    },
                    "subspace_overlaps": {
                        "min": min(subspace_overlaps) if subspace_overlaps else 0,
                        "max": max(subspace_overlaps) if subspace_overlaps else 0,
                        "mean": sum(subspace_overlaps) / len(subspace_overlaps) if subspace_overlaps else 0,
                        "n": len(subspace_overlaps),
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
                               suggested_max_strength=None, scoring_speed="full",
                               applied_rank=1):
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
        lines.append(f"    Excess conflict: {s.get('avg_excess_conflict', 0.0):.1%} "
                     f"| Subspace overlap: {s.get('avg_subspace_overlap', 0.0):.2f}")
        lines.append(f"    Avg cosine similarity: {s['avg_cosine_sim']:.2f} "
                     f"| Importance ratio: {s['magnitude_ratio']:.1f}x")
        lines.append(f"    Output strength: {output_strength}")
        if suggested_max_strength is not None:
            lines.append(f"    Suggested max output_strength: {suggested_max_strength:.2f}")
        if scoring_speed != "full":
            lines.append(f"    Scoring speed: {scoring_speed} (subsampled prefix scoring)")
        if s.get("decision_smoothing", 0.0) > 0:
            lines.append(f"    Decision smoothing: {s['decision_smoothing']:.2f}")
        lines.append("")
        lines.append("  " + "-" * 38)
        lines.append(f"  Top {len(results)} Configurations")
        lines.append("  " + "-" * 38)

        for r in results:
            lines.append("")
            c = r["config"]
            m = r["metrics"]
            marker = " (applied to output)" if r["rank"] == applied_rank else ""
            star = " \u2605" if r["rank"] == applied_rank else ""
            score_line = f"  #{r['rank']}{star}{marker}          Score: {r['score_measured']:.2f}"
            if r.get("score_external") is not None:
                score_line += f" (internal {r['score_measured']:.2f}, external {r['score_external']:.2f})"
            lines.append(score_line)
            mode_display = "per-prefix (auto)" if c["optimization_mode"] == "per_prefix" else c["merge_mode"]
            lines.append(f"    Mode: {mode_display} | Refinement: {c['merge_refinement']}")
            if c["sparsification"] != "disabled":
                spars_info = f"{c['sparsification']} (density={c['sparsification_density']}"
                if c["dare_dampening"] > 0:
                    spars_info += f", dampening={c['dare_dampening']}"
                spars_info += ")"
                lines.append(f"    Sparsification: {spars_info}")
            else:
                lines.append(f"    Sparsification: disabled")
            strat_set = c.get('strategy_set', 'full')
            strat_label = f" | Strategy: {strat_set}" if c['optimization_mode'] == 'per_prefix' else ""
            lines.append(f"    Auto-strength: {c['auto_strength']} "
                         f"| Optimization: {c['optimization_mode']}{strat_label}")
            energy_label = f" | Energy: {m['energy_ratio']:.2f}x" if "energy_ratio" in m else ""
            if m.get("effective_rank_mean", 0) > 0:
                lines.append(f"    Effective rank: {m['effective_rank_mean']:.1f} "
                             f"| Sparsity: {m.get('sparsity_mean', 0):.1%}{energy_label}")
            elif energy_label:
                lines.append(f"    Sparsity: {m.get('sparsity_mean', 0):.1%}{energy_label}")

        lines.append("")
        lines.append("  To use a different config: change selection=N")
        lines.append("  or connect TUNER_DATA to a Merge Selector node")
        lines.append("=" * 54)
        return "\n".join(lines)

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, clip=None,
                   clip_strength_multiplier=1.0, top_n=3, normalize_keys="disabled",
                   scoring_svd="disabled",
                   scoring_device="gpu",
                   architecture_preset="auto", auto_strength_floor=-1.0, evaluator=None, record_dataset="disabled",
                   cache_patches="enabled",
                   diff_cache_mode="disabled", diff_cache_ram_pct=0.5,
                   vram_budget=0.0, scoring_speed="full", scoring_formula="v2",
                   output_mode="merge", decision_smoothing=0.25,
                   smooth_slerp_gate=False, memory_mode="disabled", selection=1):
        evaluator_hash = cls._stable_data_hash(evaluator) if evaluator is not None else ""
        return (id(model), id(lora_stack), output_strength, clip_strength_multiplier, top_n,
                normalize_keys, scoring_svd, scoring_device,
                architecture_preset,
                vram_budget, record_dataset, scoring_speed, scoring_formula, output_mode,
                auto_strength_floor, decision_smoothing, evaluator_hash, memory_mode,
                selection)


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
                    "tooltip": "Which ranked configuration to apply (1 = top-ranked, 2 = next-ranked, etc.)."
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
                "auto_strength_floor": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum auto-strength scale factor for orthogonal LoRAs. Leave at -1 to use the AutoTuner's stored setting when available."
                }),
                "decision_smoothing": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Must match the AutoTuner run when per-group decisions depend on smoothing. 0 disables smoothing."
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
                     auto_strength_floor=-1.0, decision_smoothing=0.25,
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
        resolved_smoothing = tuner_data.get("decision_smoothing", decision_smoothing)
        resolved_auto_strength_floor = tuner_data.get("auto_strength_floor", auto_strength_floor)

        logging.info(f"[Merge Selector] Applying config #{selection}: "
                     f"{config['merge_mode']}, {config['merge_refinement']}")

        # Run merge with selected config
        # In per_prefix mode, let the optimizer auto-select strategy per prefix.
        strategy_override = config["merge_mode"] if config["optimization_mode"] == "global" else ""

        merged_model, merged_clip, _report, _, _lora_data = super().optimize_merge(
            model, lora_stack, output_strength,
            clip=clip,
            clip_strength_multiplier=clip_strength_multiplier,
            auto_strength=config["auto_strength"],
            auto_strength_floor=resolved_auto_strength_floor,
            optimization_mode=config["optimization_mode"],
            sparsification=config["sparsification"],
            sparsification_density=config["sparsification_density"],
            dare_dampening=config["dare_dampening"],
            merge_refinement=config["merge_refinement"],
            merge_strategy_override=strategy_override,
            free_vram_between_passes="disabled",
            vram_budget=vram_budget,
            cache_patches="enabled",
            patch_compression="smart",
            svd_device="gpu",
            normalize_keys=tuner_data.get("normalize_keys", "disabled"),
            strategy_set="full",
            architecture_preset=tuner_data.get("architecture_preset", "auto"),
            decision_smoothing=resolved_smoothing,
        )

        # Build report for this selection
        lines = []
        lines.append(f"Merge Selector \u2014 Applied config #{selection}")
        mode_display = "per-prefix (auto)" if config["optimization_mode"] == "per_prefix" else config["merge_mode"]
        lines.append(f"  Mode: {mode_display} | Refinement: {config['merge_refinement']}")
        if config["sparsification"] != "disabled":
            lines.append(f"  Sparsification: {config['sparsification']} "
                         f"(density={config['sparsification_density']})")
        lines.append(f"  Auto-strength: {config['auto_strength']} "
                     f"| Optimization: {config['optimization_mode']}")
        lines.append(f"  Heuristic score: {entry['score_heuristic']:.3f} "
                     f"| Final score: {entry.get('score_final', entry['score_measured']):.3f}")
        if entry.get("score_external") is not None:
            lines.append(f"  Internal score: {entry['score_measured']:.3f} "
                         f"| External score: {entry['score_external']:.3f}")
        report = "\n".join(lines)

        return (merged_model, merged_clip, report, _lora_data)

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, tuner_data, selection,
                   output_strength, clip=None, clip_strength_multiplier=1.0,
                   auto_strength_floor=-1.0, decision_smoothing=0.25,
                   vram_budget=0.0):
        return (id(tuner_data), selection, output_strength, clip_strength_multiplier,
                auto_strength_floor, decision_smoothing)


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
                "save_folder": (folder_choices, {"tooltip": "Which loras folder to save into. Lists configured lora paths from ComfyUI and extra_model_paths.yaml."}),
                "filename": ("STRING", {"default": "merged_lora", "tooltip": "Name for the saved file. Subdirectories are allowed (e.g. 'merged/my_lora'). Extension .safetensors is added automatically."}),
                "save_rank": ("INT", {
                    "default": 0, "min": 0, "max": 2048, "step": 4,
                    "tooltip": "0 = auto (adaptively finds the rank needed for <5% reconstruction error — recommended). Non-zero = force this rank for any layers that need compression. Higher values = more accurate but larger file."
                }),
                "bake_strength": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, the saved LoRA reproduces your exact merge when loaded at strength 1.0. When disabled, strengths are not baked in — you'll need to set the strength manually when loading."
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Example prompt or trigger words to embed in the file metadata. Useful for sharing — some UIs display this automatically."
                }),
                "description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional description or notes about this merged LoRA. Stored in file metadata."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_lora"
    CATEGORY = "LoRA Optimizer"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves merged LoRA data as a standalone .safetensors file that can be loaded by any standard LoRA loader."

    def save_lora(self, lora_data, save_folder, filename, save_rank=0, bake_strength=True, prompt="", description=""):
        if lora_data is None:
            logging.warning("[Save Merged LoRA] No lora_data received (optimizer may have returned early). Nothing to save.")
            return ("",)

        save_path = _resolve_safe_output_path(save_folder, filename, ".safetensors", "Save Merged LoRA")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model_patches = lora_data["model_patches"]
        clip_patches = lora_data["clip_patches"]
        key_map = lora_data["key_map"]
        output_strength = lora_data["output_strength"]
        clip_strength = lora_data["clip_strength"]

        auto_rank = save_rank == 0

        # Auto mode: adapt rank for any full-rank diffs that need compression.
        if auto_rank:
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
                fallback_rank = lora_data.get("sum_rank", 128)
                logging.info(f"[Save Merged LoRA] Auto rank: {fallback_rank} (no full-rank diffs to compress)")

        save_dtype = None
        for patch in list(model_patches.values()) + list(clip_patches.values()):
            if isinstance(patch, tuple) and patch[0] == "diff":
                dtype = patch[1][0].dtype
            elif hasattr(patch, "weights") and patch.weights[0] is not None:
                dtype = patch.weights[0].dtype
            else:
                continue
            if dtype not in (torch.float32, torch.float64):
                save_dtype = dtype
                break
        if save_dtype is None:
            save_dtype = torch.float16
        logging.info(f"[Save Merged LoRA] Output dtype: {save_dtype}")

        state_dict = {}

        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            for target_key, patch in patches.items():
                tkey = target_key[0] if isinstance(target_key, tuple) else target_key
                key_info = key_map.get(target_key)
                if key_info is None:
                    key_info = key_map.get(tkey, tkey)
                if isinstance(key_info, dict):
                    lora_prefix = key_info.get("canonical_prefix", tkey)
                else:
                    lora_prefix = key_info

                if isinstance(patch, (LoKrAdapter, LoHaAdapter)):
                    diff_tensor = _LoRAMergeBase._expand_patch_to_diff(patch)
                    rank = fallback_rank if auto_rank else save_rank
                    compressed = LoRAOptimizer._compress_to_lowrank(diff_tensor, rank, output_dtype=save_dtype)
                    mat_up, mat_down, alpha, mid, _, _ = compressed.weights
                    alpha = float(alpha)
                elif isinstance(patch, LoRAAdapter):
                    mat_up, mat_down, alpha, mid, _, _ = patch.weights
                    alpha = float(alpha) if alpha is not None else float(mat_down.shape[0])
                elif isinstance(patch, tuple) and len(patch) == 2 and patch[0] == "diff":
                    diff_tensor = patch[1][0]
                    rank = fallback_rank if auto_rank else save_rank
                    compressed = LoRAOptimizer._compress_to_lowrank(diff_tensor, rank, output_dtype=save_dtype)
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

        unmapped_keys = []
        nan_keys = []
        zero_keys = []
        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            for target_key, _patch in patches.items():
                direct = key_map.get(target_key)
                if direct is None:
                    tkey = target_key[0] if isinstance(target_key, tuple) else target_key
                    fallback = key_map.get(tkey)
                    label = f"{'CLIP' if is_clip else 'MODEL'} {tkey}"
                    if fallback is None:
                        unmapped_keys.append(f"{label} (using raw key as prefix)")
                    else:
                        unmapped_keys.append(f"{label} (fallback to base key)")

        for state_key, tensor in state_dict.items():
            if state_key.endswith(".alpha"):
                continue
            if torch.isnan(tensor).any():
                nan_keys.append(state_key)
            if tensor.abs().max().item() == 0:
                zero_keys.append(state_key)

        if unmapped_keys:
            logging.warning(f"[Save Merged LoRA] {len(unmapped_keys)} keys fell through to fallback mapping:")
            for item in unmapped_keys[:5]:
                logging.warning(f"  {item}")
        if nan_keys:
            logging.error(f"[Save Merged LoRA] {len(nan_keys)} tensors contain NaN")
            for item in nan_keys[:5]:
                logging.error(f"  {item}")
        if zero_keys:
            logging.warning(f"[Save Merged LoRA] {len(zero_keys)} tensors are all zeros")

        prefixes = sorted(set(key.rsplit(".lora_", 1)[0] for key in state_dict if ".lora_" in key))
        if prefixes:
            logging.info(f"[Save Merged LoRA] Sample prefixes: {prefixes[:3]} ... ({len(prefixes)} total)")

        svd_errors = []
        for is_clip, patches in [(False, model_patches), (True, clip_patches)]:
            for target_key, patch in patches.items():
                key_info = key_map.get(target_key)
                if key_info is None:
                    continue
                if isinstance(key_info, dict):
                    lora_prefix = key_info.get("canonical_prefix")
                else:
                    lora_prefix = key_info
                if not lora_prefix:
                    continue
                up_key = f"{lora_prefix}.lora_up.weight"
                down_key = f"{lora_prefix}.lora_down.weight"
                alpha_key = f"{lora_prefix}.alpha"
                if up_key not in state_dict:
                    continue
                saved_up = state_dict[up_key].float()
                saved_down = state_dict[down_key].float()
                saved_alpha = state_dict[alpha_key].item()
                rank = saved_down.shape[0]
                scale = saved_alpha / rank
                reconstructed = torch.mm(saved_up, saved_down) * scale
                if isinstance(patch, tuple) and patch[0] == "diff":
                    original_diff = patch[1][0].float()
                    strength = clip_strength if is_clip else output_strength
                    reference = original_diff * strength if bake_strength else original_diff
                    original_norm = reference.norm().item()
                    if original_norm > 0:
                        error = (reconstructed - reference).norm().item() / original_norm
                        svd_errors.append(error)
        if svd_errors:
            avg_error = sum(svd_errors) / len(svd_errors)
            max_error = max(svd_errors)
            logging.info(f"[Save Merged LoRA] SVD reconstruction error: "
                         f"avg={avg_error:.4f}, max={max_error:.4f} "
                         f"({len(svd_errors)} diffs checked)")

        # Build safetensors metadata header
        metadata = {"tool": "ComfyUI-ZImage-LoRA-Merger"}
        merge_meta = lora_data.get("merge_metadata", {})
        if merge_meta:
            source_loras = merge_meta.get("source_loras", [])
            if source_loras:
                metadata["source_loras"] = ", ".join(
                    f"{s['name']} @ {s['strength']}" for s in source_loras
                )
            for key in ("mode", "optimization_mode", "architecture", "architecture_preset",
                        "auto_strength", "sparsification", "merge_refinement", "strategy_set"):
                val = merge_meta.get(key)
                if val is not None:
                    metadata[f"merge_{key}"] = str(val)
            if merge_meta.get("sparsification_density") is not None:
                metadata["merge_sparsification_density"] = str(merge_meta["sparsification_density"])
            metadata["merge_output_strength"] = str(merge_meta.get("bake_strength_output", output_strength))
            metadata["merge_clip_strength"] = str(merge_meta.get("bake_strength_clip", clip_strength))
            metadata["merge_bake_strength"] = str(bake_strength)
        if prompt.strip():
            metadata["prompt"] = prompt.strip()
        if description.strip():
            metadata["description"] = description.strip()

        save_file(state_dict, save_path, metadata=metadata)
        logging.info(f"[Save Merged LoRA] Saved {len(state_dict) // 3} LoRA keys to {save_path}")

        return (save_path,)


class BuildAutoTunerPythonEvaluator:
    """Builds an external AutoTuner evaluator spec from a Python callable."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "module_path": ("STRING", {
                    "default": "",
                    "tooltip": "Python file path or importable module name that contains the evaluator callable."
                }),
                "callable_name": ("STRING", {
                    "default": "evaluate_candidate",
                    "tooltip": "Callable to import. Signature: fn(model=..., clip=..., lora_data=..., config=..., context=..., analysis_summary=...) -> float or {'score': float, 'details': ...}."
                }),
            },
            "optional": {
                "combine_mode": (["blend", "external_only", "multiply"], {
                    "default": "blend",
                    "tooltip": "How AutoTuner combines the built-in score with the external evaluator score."
                }),
                "weight": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend weight for the external evaluator when combine_mode=blend."
                }),
                "context_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "Optional JSON object passed through to the evaluator as context."
                }),
            },
        }

    RETURN_TYPES = ("AUTOTUNER_EVALUATOR",)
    RETURN_NAMES = ("evaluator",)
    FUNCTION = "build"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Builds a Python evaluator spec for LoRA AutoTuner so candidate ranking can incorporate prompt/reference scoring from external code."

    def build(self, module_path, callable_name, combine_mode="blend", weight=0.5, context_json="{}"):
        context = {}
        if context_json and context_json.strip():
            try:
                context = json.loads(context_json)
            except Exception as exc:
                raise ValueError(f"Invalid evaluator context JSON: {exc}")
        return ({
            "module_path": module_path.strip(),
            "callable_name": callable_name.strip(),
            "combine_mode": combine_mode,
            "weight": float(weight),
            "context": context,
        },)

    @classmethod
    def IS_CHANGED(cls, module_path, callable_name, combine_mode="blend", weight=0.5, context_json="{}"):
        return _LoRAMergeBase._stable_data_hash({
            "module_path": module_path,
            "callable_name": callable_name,
            "combine_mode": combine_mode,
            "weight": weight,
            "context_json": context_json,
        })


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
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Example prompt or trigger words to embed in the tuner file. Shown when loading."
                }),
                "description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional description or notes about this tuner run. Stored in the file."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_tuner_data"
    CATEGORY = "LoRA Optimizer"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves AutoTuner results to a .tuner file so they can be reloaded later without re-running the tuner."

    def save_tuner_data(self, tuner_data, save_folder, filename, overwrite=True, prompt="", description=""):
        if tuner_data is None:
            return ("",)

        # Embed user metadata into tuner_data before saving
        save_data = dict(tuner_data)
        if prompt.strip():
            save_data["prompt"] = prompt.strip()
        if description.strip():
            save_data["description"] = description.strip()

        base_dir = os.path.realpath(save_folder)
        base_name = filename if filename.endswith((".json", ".tuner")) else f"{filename}.tuner"
        save_path = os.path.realpath(os.path.join(base_dir, base_name))
        try:
            if os.path.commonpath([base_dir, save_path]) != base_dir:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"[Save Tuner Data] Path escapes tuner_data directory: {filename}") from exc
        if not overwrite and os.path.exists(save_path):
            stem, ext = os.path.splitext(save_path)
            counter = 1
            while os.path.exists(save_path):
                save_path = f"{stem}_{counter:03d}{ext}"
                counter += 1
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
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

    RETURN_TYPES = ("TUNER_DATA", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("tuner_data", "prompt", "description", "metadata_info")
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

        prompt = tuner_data.get("prompt", "")
        description = tuner_data.get("description", "")

        # Build metadata info string
        info_lines = []
        for key in ("source_loras", "architecture_preset", "normalize_keys",
                     "auto_strength_floor", "decision_smoothing", "lora_hash"):
            val = tuner_data.get(key)
            if val is not None:
                info_lines.append(f"{key}: {val}")
        top_n = tuner_data.get("top_n", [])
        if top_n:
            info_lines.append(f"configs: {len(top_n)}")
            best = top_n[0]
            cfg = best.get("config", {})
            info_lines.append(f"best score: {best.get('score_final', 'N/A')}")
            for k in ("mode", "optimization_mode", "auto_strength", "sparsification",
                       "merge_refinement", "strategy_set"):
                v = cfg.get(k)
                if v is not None:
                    info_lines.append(f"best {k}: {v}")
        if prompt:
            info_lines.append(f"prompt: {prompt}")
        if description:
            info_lines.append(f"description: {description}")
        metadata_info = "\n".join(info_lines) if info_lines else "No metadata found."

        return (tuner_data, prompt, description, metadata_info)


class LoRACompatibilityAnalyzer(LoRAOptimizer):
    """
    Standalone pre-merge planning tool. Analyzes pairwise LoRA interactions
    and recommends which LoRAs are safe to merge together.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Toggle on to run compatibility analysis. Off by default so normal queue execution stays cheap."
                }),
                "create_nodes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically create LoRA Stack and Load LoRA nodes from results."
                }),
                "model": ("MODEL", {"tooltip": "Base model used for key mapping and target grouping."}),
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRA stack to analyze."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional CLIP model for text-encoder LoRA keys."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("report", "compatibility_map")
    FUNCTION = "analyze"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Analyzes LoRA compatibility and suggests merge groups. No merge is performed."

    @classmethod
    def IS_CHANGED(cls, enabled, **kwargs):
        if not enabled:
            return "disabled"
        return float("NaN")

    def analyze(self, enabled, create_nodes, model, lora_stack, clip=None):
        has_clip = [clip is not None]
        if not enabled:
            msg = "Analysis disabled. Toggle 'enabled' to run."
            return {"ui": {"groups": [], "has_clip": has_clip}, "result": (msg, self._empty_image())}
        if not lora_stack or len(lora_stack) == 0:
            msg = "No LoRAs in stack."
            return {"ui": {"groups": [], "has_clip": has_clip}, "result": (msg, self._empty_image())}

        normalized_stack = self._normalize_stack(lora_stack)
        active_loras = [item for item in normalized_stack if item["strength"] != 0]
        if len(active_loras) == 0:
            msg = "No active LoRAs in stack (all zero strength)."
            return {"ui": {"groups": [], "has_clip": has_clip}, "result": (msg, self._empty_image())}

        n_loras = len(active_loras)
        detected_arch = getattr(self, "_detected_arch", None) or "unknown"
        preset_key, arch_preset = _resolve_arch_preset("auto", detected_arch)
        logging.info(f"[Compatibility Analyzer] Architecture preset: {preset_key} ({arch_preset['display_name']})")

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
            return {"ui": {"groups": [], "has_clip": has_clip}, "result": (report, self._empty_image())}

        logging.info(f"[Compatibility Analyzer] Analyzing {n_loras} LoRAs...")
        t_start = time.time()

        model_keys = self._get_model_keys(model)
        clip_keys = {}
        if clip is not None:
            clip_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})

        all_lora_prefixes = self._collect_lora_prefixes(active_loras)
        target_groups = self._build_target_groups(all_lora_prefixes, model_keys, clip_keys)
        if not target_groups:
            msg = "No compatible LoRA target groups found. LoRAs may be incompatible with this model architecture."
            return {"ui": {"groups": [], "has_clip": has_clip}, "result": (msg, self._empty_image())}

        compute_device = self._get_compute_device()
        analysis = self._run_group_analysis(
            target_groups, active_loras, model, clip, compute_device,
            clip_strength_multiplier=1.0,
            merge_refinement="none",
            decision_smoothing=0.0,
        )
        prefix_count = analysis["prefix_count"]
        if prefix_count == 0:
            msg = "No compatible LoRA target groups found. LoRAs may be incompatible with this model architecture."
            return {"ui": {"groups": [], "has_clip": has_clip}, "result": (msg, self._empty_image())}

        per_lora_stats = analysis["per_lora_stats"]
        pair_accum = analysis["pair_accum"]
        pairs = analysis["pairs"]
        branch_energy = analysis["branch_energy"]

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

        pairwise_conflicts = []
        pairwise_similarities = {}
        total_overlap = 0
        total_conflict = 0
        for i, j in pairs:
            pair_metrics = pair_accum[(i, j)]
            pair_overlap = pair_metrics["overlap"]
            pair_conflict = pair_metrics["conflict"]
            pair_dot = pair_metrics["dot"]
            pair_na_sq = pair_metrics["norm_a_sq"]
            pair_nb_sq = pair_metrics["norm_b_sq"]
            total_overlap += pair_overlap
            total_conflict += pair_conflict
            ratio = pair_conflict / pair_overlap if pair_overlap > 0 else 0.0
            denom = math.sqrt(pair_na_sq) * math.sqrt(pair_nb_sq)
            cos_sim = pair_dot / denom if denom > 0 else 0.0
            pairwise_similarities[(i, j)] = cos_sim
            pairwise_conflicts.append({
                "i": i,
                "j": j,
                "pair": f"{active_loras[i]['name']} vs {active_loras[j]['name']}",
                "overlap": pair_overlap,
                "conflicts": pair_conflict,
                "ratio": ratio,
                "cosine_sim": cos_sim,
            })

        avg_conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0.0
        avg_cos_sim = (sum(pairwise_similarities.values()) / len(pairwise_similarities)) if pairwise_similarities else 0.0
        valid_l2 = [m for m in l2_means if m > 0]
        magnitude_ratio = max(valid_l2) / min(valid_l2) if len(valid_l2) >= 2 else 1.0

        collection_stats = {
            "n_loras": n_loras,
            "total_keys": prefix_count,
            "avg_conflict": avg_conflict_ratio,
            "avg_cos_sim": avg_cos_sim,
            "magnitude_ratio": magnitude_ratio,
        }

        compat_matrix = self._compute_compat_matrix(pairwise_similarities, pairwise_conflicts, n_loras)
        groups = self._cluster_loras(compat_matrix, threshold=0.05)

        group_info = []
        for group in groups:
            info = {"indices": group}
            if len(group) == 1:
                idx = group[0]
                info["type"] = "solo"
                info["strength"] = active_loras[idx]["strength"]
                opposing = []
                for other_idx in range(n_loras):
                    if other_idx == idx:
                        continue
                    key = (min(idx, other_idx), max(idx, other_idx))
                    cos_sim = pairwise_similarities.get(key, 0.0)
                    if cos_sim < -0.05:
                        opposing.append((active_loras[other_idx]["name"], cos_sim))
                opposing.sort(key=lambda item: item[1])
                info["opposing"] = opposing
            else:
                info["type"] = "merge"
                intra_cos = []
                intra_conflict = []
                intra_excess = []
                intra_subspace = []
                model_norm_sq = []
                clip_norm_sq = []
                model_dots = {}
                clip_dots = {}
                group_loras = [active_loras[idx] for idx in group]

                for pos, orig_idx in enumerate(group):
                    model_norm_sq.append(branch_energy["model"]["norm_sq"][orig_idx])
                    clip_norm_sq.append(branch_energy["clip"]["norm_sq"][orig_idx])

                for a_pos in range(len(group)):
                    for b_pos in range(a_pos + 1, len(group)):
                        left = group[a_pos]
                        right = group[b_pos]
                        pair_key = (min(left, right), max(left, right))
                        metrics = pair_accum.get(pair_key, {})
                        cos_sim = pairwise_similarities.get(pair_key, 0.0)
                        ratio = (metrics.get("conflict", 0) / metrics.get("overlap", 1)) if metrics.get("overlap", 0) > 0 else 0.0
                        weighted_total = metrics.get("weighted_total", 0.0)
                        excess_conflict = (metrics.get("excess_conflict_weighted", 0.0) / weighted_total) if weighted_total > 0 else 0.0
                        subspace_overlap = (metrics.get("subspace_num", 0.0) / metrics.get("subspace_den", 1.0)) if metrics.get("subspace_den", 0.0) > 0 else 0.0
                        intra_cos.append(cos_sim)
                        intra_conflict.append(ratio)
                        intra_excess.append(excess_conflict)
                        intra_subspace.append(subspace_overlap)
                        model_dots[(a_pos, b_pos)] = branch_energy["model"]["dot"].get(pair_key, 0.0)
                        clip_dots[(a_pos, b_pos)] = branch_energy["clip"]["dot"].get(pair_key, 0.0)

                info["avg_cos_sim"] = sum(intra_cos) / len(intra_cos) if intra_cos else 0.0
                info["avg_conflict"] = sum(intra_conflict) / len(intra_conflict) if intra_conflict else 0.0
                info["avg_excess_conflict"] = sum(intra_excess) / len(intra_excess) if intra_excess else 0.0
                info["avg_subspace_overlap"] = sum(intra_subspace) / len(intra_subspace) if intra_subspace else 0.0

                avg_compat = sum(cs * (1.0 - cr) for cs, cr in zip(intra_cos, intra_conflict)) / len(intra_cos) if intra_cos else 0.0
                if avg_compat > 0.2:
                    info["confidence"] = "High"
                elif avg_compat > 0.05:
                    info["confidence"] = "Moderate"
                else:
                    info["confidence"] = "Low"

                group_branch_energy = {
                    "model": {"norm_sq": model_norm_sq, "dot": model_dots},
                    "clip": {"norm_sq": clip_norm_sq, "dot": clip_dots},
                }
                auto_strength_info = self._compute_auto_strengths(
                    group_loras,
                    group_branch_energy,
                    arch_preset=arch_preset,
                    detected_arch=detected_arch,
                )
                info["strengths"] = {
                    group[idx]: auto_strength_info["model_strengths"][idx]
                    for idx in range(len(group))
                }

                mode, _, _, _ = self._auto_select_params(
                    info["avg_conflict"],
                    magnitude_ratio,
                    magnitude_samples=analysis["all_magnitude_samples"],
                    avg_cos_sim=info["avg_cos_sim"],
                    avg_excess_conflict=info["avg_excess_conflict"],
                    avg_subspace_overlap=info["avg_subspace_overlap"],
                    arch_preset=arch_preset,
                )
                info["suggested_merge"] = mode

            group_info.append(info)

        warnings = []
        for pc in pairwise_conflicts:
            if pc["cosine_sim"] < -0.1:
                warnings.append({
                    "type": "opposing",
                    "name_i": active_loras[pc["i"]]["name"],
                    "name_j": active_loras[pc["j"]]["name"],
                    "cos_sim": pc["cosine_sim"],
                })

        strong_loras = {}
        if len(l2_means) >= 2:
            valid_l2 = [m for m in l2_means if m > 0]
            mean_l2 = sum(valid_l2) / len(valid_l2) if valid_l2 else 1.0
            lora_avg_ratios = {
                idx: (l2 / mean_l2)
                for idx, l2 in enumerate(l2_means)
                if l2 > 0 and mean_l2 > 0
            }
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
                        strong_loras.setdefault(strong_idx, [])
                        if weak_name not in [name for name, _ in strong_loras[strong_idx]]:
                            strong_loras[strong_idx].append((weak_name, ratio))
            for strong_idx in sorted(strong_loras.keys()):
                warnings.append({
                    "type": "magnitude_group",
                    "stronger": active_loras[strong_idx]["name"],
                    "avg_ratio": lora_avg_ratios.get(strong_idx, 0.0),
                    "weaker_list": [name for name, _ in strong_loras[strong_idx]],
                })

        logging.info(f"[Compatibility Analyzer] Analysis complete ({time.time() - t_start:.1f}s)")
        report = self._build_compatibility_report(
            active_loras, groups, group_info, lora_stats, pairwise_conflicts,
            collection_stats, warnings, detected_arch, prefix_count,
        )
        raw_matrix = self._compute_raw_compat_matrix(pairwise_similarities, pairwise_conflicts, n_loras)
        display_names = [os.path.splitext(os.path.basename(item["name"]))[0] for item in active_loras]
        heatmap = self._generate_heatmap(raw_matrix, display_names, groups)

        groups_for_ui = []
        if create_nodes:
            for info in group_info:
                if info["type"] == "merge":
                    groups_for_ui.append({
                        "node_type": "stack",
                        "loras": [
                            {"name": active_loras[idx]["name"], "strength": active_loras[idx]["strength"]}
                            for idx in info["indices"]
                        ],
                        "suggested_merge": info.get("suggested_merge", "weighted_average"),
                        "confidence": info.get("confidence", "Low"),
                    })
                elif info["type"] == "solo":
                    idx = info["indices"][0]
                    groups_for_ui.append({
                        "node_type": "loader",
                        "lora_name": active_loras[idx]["name"],
                        "strength": active_loras[idx]["strength"],
                    })

        return {"ui": {"groups": groups_for_ui, "has_clip": has_clip}, "result": (report, heatmap)}

    @staticmethod
    def _empty_image():
        return torch.zeros(1, 1, 1, 3, dtype=torch.float32)

    @staticmethod
    def _compute_compat_matrix(pairwise_similarities, pairwise_conflicts, n_loras):
        conflict_by_pair = {(pc["i"], pc["j"]): pc["ratio"] for pc in pairwise_conflicts}
        overlap_by_pair = {(pc["i"], pc["j"]): pc["overlap"] for pc in pairwise_conflicts}
        matrix = [[0.0] * n_loras for _ in range(n_loras)]
        for (i, j), cos_sim in pairwise_similarities.items():
            conflict_ratio = conflict_by_pair.get((i, j), 0.0)
            overlap = overlap_by_pair.get((i, j), 0)
            compat = cos_sim * (1.0 - conflict_ratio)
            # Only boost non-opposing pairs that actually share weights
            if cos_sim > -0.1 and overlap > 0:
                compat += 0.1
            matrix[i][j] = compat
            matrix[j][i] = compat
        return matrix

    @staticmethod
    def _compute_raw_compat_matrix(pairwise_similarities, pairwise_conflicts, n_loras):
        conflict_by_pair = {(pc["i"], pc["j"]): pc["ratio"] for pc in pairwise_conflicts}
        matrix = [[0.0] * n_loras for _ in range(n_loras)]
        for (i, j), cos_sim in pairwise_similarities.items():
            conflict_ratio = conflict_by_pair.get((i, j), 0.0)
            compat = cos_sim * (1.0 - conflict_ratio)
            matrix[i][j] = compat
            matrix[j][i] = compat
        return matrix

    @staticmethod
    def _cluster_loras(compat_matrix, threshold=0.05):
        n = len(compat_matrix)
        groups = [[i] for i in range(n)]
        while len(groups) > 1:
            best_score = -float("inf")
            best_pair = None
            for left_idx in range(len(groups)):
                for right_idx in range(left_idx + 1, len(groups)):
                    total = 0.0
                    count = 0
                    for a in groups[left_idx]:
                        for b in groups[right_idx]:
                            total += compat_matrix[a][b]
                            count += 1
                    avg_compat = total / count if count > 0 else 0.0
                    if avg_compat > best_score:
                        best_score = avg_compat
                        best_pair = (left_idx, right_idx)
            if best_score < threshold or best_pair is None:
                break
            left_idx, right_idx = best_pair
            merged = groups[left_idx] + groups[right_idx]
            groups = [group for idx, group in enumerate(groups) if idx not in best_pair]
            groups.append(merged)
        groups.sort(key=lambda group: (-len(group), group[0]))
        return groups

    @staticmethod
    def _generate_heatmap(compat_matrix, lora_names, groups):
        from PIL import Image, ImageDraw
        import numpy as np

        n = len(compat_matrix)
        if n == 0:
            return torch.zeros(1, 1, 1, 3, dtype=torch.float32)

        display_order = []
        for group in groups:
            display_order.extend(group)

        cell_size = 80
        tmp_img = Image.new("RGB", (1, 1))
        tmp_draw = ImageDraw.Draw(tmp_img)
        max_label_w = max(tmp_draw.textlength(name) for name in lora_names) if lora_names else 100
        margin = int(max_label_w) + 20
        col_header_h = margin
        img_w = margin + n * cell_size
        img_h = col_header_h + n * cell_size

        img = Image.new("RGB", (img_w, img_h), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        def compat_to_color(value):
            t = max(0.0, min(1.0, (value + 0.4) / 0.9))
            if t < 0.5:
                s = t / 0.5
                return (int(200 + 20 * s), int(180 * s), 0)
            s = (t - 0.5) / 0.5
            return (int(220 * (1 - s)), int(180 + 20 * s), 0)

        for row in range(n):
            for col in range(n):
                orig_i = display_order[row]
                orig_j = display_order[col]
                x = margin + col * cell_size
                y = col_header_h + row * cell_size
                if orig_i == orig_j:
                    color = (80, 80, 80)
                    value_text = "-"
                else:
                    value = compat_matrix[orig_i][orig_j]
                    color = compat_to_color(value)
                    value_text = f"{value:.2f}"
                draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=color)
                bbox = draw.textbbox((0, 0), value_text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                tx = x + (cell_size - text_w) // 2
                ty = y + (cell_size - text_h) // 2
                brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                draw.text((tx, ty), value_text, fill=text_color)

        for pos in range(n):
            label = lora_names[display_order[pos]]
            draw.text((5, col_header_h + pos * cell_size + cell_size // 2 - 5), label, fill=(220, 220, 220))

        for pos in range(n):
            label = lora_names[display_order[pos]]
            bbox = tmp_draw.textbbox((0, 0), label)
            text_w = bbox[2] - bbox[0] + 4
            text_h = bbox[3] - bbox[1] + 4
            txt_img = Image.new("RGBA", (text_w, text_h), (30, 30, 30, 255))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((2, 2), label, fill=(220, 220, 220, 255))
            rotated = txt_img.rotate(90, expand=True)
            paste_x = margin + pos * cell_size + (cell_size - rotated.width) // 2
            paste_y = col_header_h - rotated.height - 5
            img.paste(rotated.convert("RGB"), (paste_x, max(0, paste_y)))

        pos = 0
        for group in groups:
            if len(group) > 1:
                x0 = margin + pos * cell_size - 1
                y0 = col_header_h + pos * cell_size - 1
                x1 = margin + (pos + len(group)) * cell_size
                y1 = col_header_h + (pos + len(group)) * cell_size
                for offset in range(3):
                    draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=(255, 255, 255))
            pos += len(group)

        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _build_compatibility_report(self, active_loras, groups, group_info,
                                    lora_stats, pairwise_conflicts,
                                    collection_stats, warnings,
                                    detected_arch, prefix_count):
        def _display_name(name):
            return name.replace(".safetensors", "")

        lines = []
        lines.append("=" * 55)
        lines.append("  LoRA Compatibility Analysis")
        lines.append("=" * 55)
        lines.append("")
        lines.append(f"Analyzed: {len(active_loras)} LoRAs across {prefix_count} target groups")
        arch_display = detected_arch.upper() if detected_arch and detected_arch != "unknown" else "Unknown"
        lines.append(f"Architecture: {arch_display}")
        lines.append(f"Avg conflict: {collection_stats['avg_conflict']:.0%} | Avg cosine similarity: {collection_stats['avg_cos_sim']:.2f}")
        lines.append("")
        lines.append("-" * 30 + " Recommended Groups " + "-" * 30)
        lines.append("")

        group_num = 0
        solo_entries = []
        for info in group_info:
            if info["type"] == "solo":
                solo_entries.append(info)
                continue
            group_num += 1
            label = "Merge together" if info["confidence"] != "Low" else "Safe to combine"
            lines.append(f"Group {group_num} -- {label} (compatibility: {info['confidence']})")
            for idx in info["indices"]:
                strength = info["strengths"].get(idx, active_loras[idx]["strength"])
                lines.append(f"  * {_display_name(active_loras[idx]['name']):<28s} strength {strength:.2f}")
            lines.append(f"  Avg cosine sim: {info['avg_cos_sim']:.2f} | Avg conflict: {info['avg_conflict']:.0%}")
            lines.append(f"  Suggested merge: {info['suggested_merge']}")
            lines.append("")

        if solo_entries:
            lines.append("Solo -- Use independently")
            for info in solo_entries:
                idx = info["indices"][0]
                lines.append(f"  * {_display_name(active_loras[idx]['name']):<28s} strength {info['strength']:.2f}")
                if info["opposing"]:
                    opposing = [f"{_display_name(name)} (cos: {cos:.2f})" for name, cos in info["opposing"][:3]]
                    lines.append(f"    Opposes: {', '.join(opposing)}")
            lines.append("")

        if warnings:
            lines.append("-" * 30 + " Warnings " + "-" * 30)
            lines.append("")
            for warning in warnings:
                if warning["type"] == "opposing":
                    lines.append(f"! {_display_name(warning['name_i'])} vs {_display_name(warning['name_j'])}: Opposing (cos_sim: {warning['cos_sim']:.2f})")
                    lines.append("  These cancel each other out and usually degrade quality when merged together.")
                    lines.append("")
                elif warning["type"] == "magnitude_group":
                    lines.append(f"! {_display_name(warning['stronger'])} ({warning['avg_ratio']:.1f}x avg) overshadows:")
                    lines.append(f"  {', '.join(_display_name(name) for name in warning['weaker_list'])}")
                    lines.append("  Consider lowering its strength or raising the weaker LoRAs.")
                    lines.append("")

        lines.append("-" * 30 + " Pairwise Compatibility " + "-" * 30)
        lines.append("")
        sorted_pairs = sorted(pairwise_conflicts, key=lambda item: item["cosine_sim"] * (1.0 - item["ratio"]), reverse=True)
        pair_labels = [_display_name(item["pair"]) for item in sorted_pairs]
        max_pair_w = max((len(label) for label in pair_labels), default=45)
        for item, label in zip(sorted_pairs, pair_labels):
            compat = round(item["cosine_sim"] * (1.0 - item["ratio"]), 2) + 0.0
            cos_sim = round(item["cosine_sim"], 2) + 0.0
            indicator = " [OK]" if compat > 0.2 else (" [!!]" if compat < -0.05 else "")
            lines.append(
                f"  {label:<{max_pair_w}s}  cos:{cos_sim:6.2f}  "
                f"conflict:{item['ratio']:5.0%}  compat:{compat:6.2f}{indicator}"
            )
        lines.append("")
        lines.append("=" * 55)
        return "\n".join(lines)


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
                enriched = [(
                    item["name"], item["strength"], item["clip_strength"],
                    resolved, item.get("key_filter", "all")
                )]
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
            # LoKr (Kronecker)
            ".lokr_w1", ".lokr_w2",
            ".lokr_w1_a", ".lokr_w1_b",
            ".lokr_w2_a", ".lokr_w2_b",
            ".lokr_t2",
            # LoHa (Hadamard)
            ".hada_w1_a", ".hada_w1_b",
            ".hada_w2_a", ".hada_w2_b",
            ".hada_t1", ".hada_t2",
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
                if info is not None:
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
                else:
                    # Try LoKr / LoHa formats
                    use_gpu = compute_device.type != "cpu"
                    alt = self._get_lokr_diff(
                        item["lora"], prefix,
                        device=compute_device if use_gpu else None, to_cpu=False,
                    )
                    if alt is None:
                        alt = self._get_loha_diff(
                            item["lora"], prefix,
                            device=compute_device if use_gpu else None, to_cpu=False,
                        )
                    if alt is not None:
                        diff, _, _ = alt
                        diff = diff.float() * item["strength"]
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
                                     item["clip_strength"], item.get("conflict_mode", "all"),
                                     item.get("key_filter", "all")))
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
                        item.get("key_filter", "all"),
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


class LoRAMetadataReader:
    """
    Passthrough node that reads embedded metadata from LoRAs in a stack.
    Outputs the stack unchanged plus extracted prompt and metadata strings.
    Works with any safetensors LoRA — not limited to LoRAs saved by this pack.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Connect a LoRA Stack here. The stack passes through unchanged."}),
            },
        }

    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_stack", "prompt", "description", "metadata_info")
    FUNCTION = "read_metadata"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = "Reads embedded metadata (prompt, description, merge info) from LoRAs in the stack. The stack passes through unchanged."

    def read_metadata(self, lora_stack):
        if not lora_stack:
            return (lora_stack, "", "", "No LoRAs in stack.")

        all_prompts = []
        all_descriptions = []
        info_lines = []

        for item in lora_stack:
            # Get metadata — either already attached by LoRAStack or read from disk
            if isinstance(item, dict):
                name = item.get("name", "unknown")
                meta = item.get("metadata", {})
                if not meta:
                    # Try reading from disk if not already attached
                    try:
                        lora_path = folder_paths.get_full_path_or_raise("loras", name)
                        meta = _read_safetensors_metadata(lora_path)
                    except Exception:
                        meta = {}
            elif isinstance(item, (tuple, list)):
                name = item[0] if len(item) > 0 else "unknown"
                try:
                    lora_path = folder_paths.get_full_path_or_raise("loras", name)
                    meta = _read_safetensors_metadata(lora_path)
                except Exception:
                    meta = {}
            else:
                continue

            if not meta:
                info_lines.append(f"[{name}] No metadata found")
                continue

            prompt = meta.get("prompt", "")
            desc = meta.get("description", "")
            if prompt:
                all_prompts.append((name, prompt))
            if desc:
                all_descriptions.append((name, desc))

            # Collect interesting metadata fields (exclude prompt/description — they have dedicated outputs)
            entry_lines = [f"[{name}]"]
            for key in ("source_loras", "tool",
                         "merge_mode", "merge_optimization_mode", "merge_architecture",
                         "merge_auto_strength", "merge_sparsification",
                         "merge_merge_refinement", "merge_strategy_set",
                         "merge_output_strength", "merge_clip_strength",
                         "merge_bake_strength"):
                val = meta.get(key)
                if val:
                    display_key = key.replace("merge_", "") if key.startswith("merge_") else key
                    entry_lines.append(f"  {display_key}: {val}")
            # Also show any other non-tensor metadata
            shown = {"prompt", "description", "source_loras", "tool",
                     "merge_mode", "merge_optimization_mode", "merge_architecture",
                     "merge_auto_strength", "merge_sparsification",
                     "merge_merge_refinement", "merge_strategy_set",
                     "merge_output_strength", "merge_clip_strength",
                     "merge_bake_strength"}
            for key, val in sorted(meta.items()):
                if key not in shown and not key.startswith("__"):
                    entry_lines.append(f"  {key}: {val[:200]}")
            info_lines.append("\n".join(entry_lines))

        if len(all_prompts) == 1:
            combined_prompt = all_prompts[0][1]
        else:
            combined_prompt = "\n\n".join(f"[{n}]: {p}" for n, p in all_prompts)
        if len(all_descriptions) == 1:
            combined_desc = all_descriptions[0][1]
        else:
            combined_desc = "\n\n".join(f"[{n}]: {d}" for n, d in all_descriptions)
        metadata_info = "\n\n".join(info_lines)

        return (lora_stack, combined_prompt, combined_desc, metadata_info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoRAStack": LoRAStack,
    "LoRAStackDynamic": LoRAStackDynamic,
    "LoRAOptimizer": LoRAOptimizer,
    "LoRAOptimizerSimple": LoRAOptimizerSimple,
    "SaveMergedLoRA": SaveMergedLoRA,
    "BuildAutoTunerPythonEvaluator": BuildAutoTunerPythonEvaluator,
    "LoRAConflictEditor": LoRAConflictEditor,
    "MergedLoRAToHook": MergedLoRAToHook,
    "MergedLoRAToWanVideo": MergedLoRAToWanVideo,
    "WanVideoLoRAOptimizer": WanVideoLoRAOptimizer,
    "LoRAAutoTuner": LoRAAutoTuner,
    "LoRAMergeSelector": LoRAMergeSelector,
    "SaveTunerData": SaveTunerData,
    "LoadTunerData": LoadTunerData,
    "LoRACompatibilityAnalyzer": LoRACompatibilityAnalyzer,
    "LoRAMergeSettings": LoRAMergeSettings,
    "LoRAOptimizerSettings": LoRAOptimizerSettings,
    "LoRAAutoTunerSettings": LoRAAutoTunerSettings,
    "LoRAMetadataReader": LoRAMetadataReader,
    "LoRAMergeFormula": LoRAMergeFormula,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAStack": "LoRA Stack",
    "LoRAStackDynamic": "LoRA Stack (Dynamic)",
    "LoRAOptimizer": "LoRA Optimizer (Legacy)",
    "LoRAOptimizerSimple": "LoRA Optimizer",
    "SaveMergedLoRA": "Save Merged LoRA",
    "BuildAutoTunerPythonEvaluator": "Build AutoTuner Python Evaluator",
    "LoRAConflictEditor": "LoRA Conflict Editor",
    "MergedLoRAToHook": "Merged LoRA to Hook",
    "MergedLoRAToWanVideo": "(WIP) Merged LoRA → WanVideo",
    "WanVideoLoRAOptimizer": "(WIP) WanVideo LoRA Optimizer",
    "LoRAAutoTuner": "LoRA AutoTuner",
    "LoRAMergeSelector": "Merge Selector",
    "SaveTunerData": "Save Tuner Data",
    "LoadTunerData": "Load Tuner Data",
    "LoRACompatibilityAnalyzer": "LoRA Compatibility Analyzer",
    "LoRAMergeSettings": "LoRA Merge Settings",
    "LoRAOptimizerSettings": "LoRA Optimizer Settings",
    "LoRAAutoTunerSettings": "LoRA AutoTuner Settings",
    "LoRAMetadataReader": "LoRA Metadata Reader",
    "LoRAMergeFormula": "LoRA Merge Formula",
}
