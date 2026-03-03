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
import concurrent.futures
import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora


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
    DESCRIPTION = "Adds a LoRA to the stack for use with LoRA Optimizer"
    
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
    def _compress_to_lowrank(diff, rank):
        """
        Re-compress a full-rank diff tensor to low-rank via truncated SVD.
        Returns ("lora", (mat_up, mat_down, alpha=rank, None)) so ComfyUI
        computes up @ down * (rank/rank) = up @ down (no extra scaling).

        For a [4096, 4096] diff at rank 128: 64MB → 2MB (~32x reduction).
        """
        original_shape = diff.shape
        # Reshape to 2D for SVD: [out_features, in_features]
        mat = diff.reshape(original_shape[0], -1).float()
        rank = min(rank, min(mat.shape))

        # Truncated SVD on CPU (more stable than GPU for large matrices)
        device = mat.device
        if device.type != "cpu":
            mat = mat.cpu()
        U, S, V = torch.svd_lowrank(mat, q=rank)
        # U: [out, rank], S: [rank], V: [in, rank]
        # Reconstruct as: mat_up = U * sqrt(S), mat_down = sqrt(S) * V^T
        sqrt_S = S.sqrt()
        mat_up = (U * sqrt_S.unsqueeze(0))    # [out, rank]
        mat_down = (V * sqrt_S.unsqueeze(0)).T  # [rank, in]
        # alpha=rank so ComfyUI computes: up @ down * (rank/rank) = up @ down
        return ("lora", (mat_up, mat_down, float(rank), None))

    @torch.no_grad()
    def _merge_diffs(self, diffs_with_weights, mode, density=0.5, majority_sign_method="frequency",
                     compute_device=None):
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
            for idx in range(len(diffs_with_weights)):
                d, w = diffs_with_weights[idx]
                diffs_with_weights[idx] = None  # Free input diff early
                d_f = d.to(device=dev, dtype=torch.float32)
                del d
                if w < 0:
                    d_f = -d_f
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

    def _normalize_stack(self, lora_stack):
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

        if isinstance(first, (tuple, list)):
            # Standard format: (lora_name, model_strength, clip_strength)
            normalized = []
            for entry in lora_stack:
                if not isinstance(entry, (tuple, list)) or len(entry) < 3:
                    logging.warning("[LoRA Optimizer] Skipping malformed tuple entry (expected 3 elements)")
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
                        logging.warning(f"[LoRA Optimizer] Failed to load LoRA '{lora_name}': {e}")
                        continue

                normalized.append({
                    "name": lora_name,
                    "lora": lora_dict,
                    "strength": model_str,
                    "clip_strength": clip_str,
                })
            return normalized

        elif isinstance(first, dict):
            # LoRAStack format: already loaded dicts
            normalized = []
            for item in lora_stack:
                if not isinstance(item, dict) or "lora" not in item or "strength" not in item or "name" not in item:
                    logging.warning("[LoRA Optimizer] Skipping malformed dict entry (expected keys: name, lora, strength)")
                    continue
                normalized.append({
                    "name": item["name"],
                    "lora": item["lora"],
                    "strength": item["strength"],
                    "clip_strength": None,  # use global multiplier
                })
            return normalized

        logging.warning("[LoRA Optimizer] Unrecognized stack format")
        return []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to apply LoRA to"}),
                "lora_stack": ("LORA_STACK", {"tooltip": "LoRA stack - accepts standard (name, model_str, clip_str) tuples or LoRAStack dicts"}),
                "output_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                                              "tooltip": "Strength of the merged effect"}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "The CLIP model (optional — omit for video/latent-only workflows)"}),
                "clip_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                                                       "tooltip": "Strength multiplier for CLIP"}),
                "auto_strength": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Auto-reduce LoRA strengths to prevent overexposure from stacking"
                }),
                "free_vram_between_passes": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Release GPU cache between analysis and merge passes. Lowers peak VRAM at negligible speed cost."
                }),
                "optimization_mode": (["per_prefix", "global", "weighted_sum_only"], {
                    "default": "per_prefix",
                    "tooltip": "per_prefix: each weight group picks its own strategy. global: single strategy for all. weighted_sum_only: force simple weighted sum everywhere (no TIES/averaging, fully compressible)."
                }),
                "cache_patches": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": "Cache merged patches in RAM for faster re-execution. Disable to free RAM after merge (recommended for video models)."
                }),
                "compress_patches": (["non_ties", "all", "disabled"], {
                    "default": "non_ties",
                    "tooltip": "Re-compress full-rank merged patches to low-rank via SVD. 'non_ties' compresses only weighted_sum/weighted_average prefixes (lossless, TIES stays full-rank). 'all' compresses everything (lossy on TIES prefixes). Recommended for video models."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "analysis_report")
    FUNCTION = "optimize_merge"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Auto-analyzes LoRA stack and selects optimal merge strategy per weight group. Outputs merged model + analysis report."

    @staticmethod
    def _compute_cache_key(lora_stack, output_strength, clip_strength_multiplier, auto_strength, optimization_mode="per_prefix", compress_patches="non_ties"):
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
        h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}|om={optimization_mode}|cp={compress_patches}".encode())
        return h.hexdigest()[:16]

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, clip=None,
                   clip_strength_multiplier=1.0, auto_strength="disabled",
                   free_vram_between_passes="disabled", optimization_mode="per_prefix",
                   cache_patches="enabled", compress_patches="non_ties"):
        return cls._compute_cache_key(lora_stack, output_strength,
                                      clip_strength_multiplier, auto_strength,
                                      optimization_mode, compress_patches)

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
        Compute sign conflict ratio between two diff tensors.
        Samples up to 100k positions for large tensors.
        Returns (n_overlap, n_conflict).
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
        pair_conflicts: dict mapping (i, j) -> (overlap, conflict)
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
                ov, conf = self._sample_conflict(diff_i, diff_j, device=device)
                pair_conflicts[(i, j)] = (ov, conf)

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
                      prefix_decisions=None):
        """Format analysis as a multi-line report string."""
        lines = []
        lines.append("=" * 50)
        lines.append("LORA OPTIMIZER - ANALYSIS REPORT")
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
            block_summary = []
            for block_name, entries in block_data.items():
                modes = [e[0] for e in entries]
                conflicts = [e[1] for e in entries]
                n_loras_max = max(e[2] for e in entries)
                # Dominant mode = most frequent
                mode_counts = {}
                for m in modes:
                    mode_counts[m] = mode_counts.get(m, 0) + 1
                dominant = max(mode_counts, key=mode_counts.get)
                avg_conflict = sum(conflicts) / len(conflicts) if conflicts else 0
                n_prefixes = len(entries)
                block_summary.append((block_name, dominant, avg_conflict, n_loras_max, n_prefixes))

            # Sort by block name for consistent ordering
            block_summary.sort(key=lambda x: x[0])

            lines.append("")
            lines.append("--- Block Strategy Map ---")
            symbols = {"weighted_sum": "====", "slerp": "~~~~", "weighted_average": "----", "ties": "####"}
            labels = {"weighted_sum": "sum", "slerp": "slrp", "weighted_average": "avg", "ties": "TIES"}
            # Find max block name length for alignment
            max_name = max(len(b[0]) for b in block_summary) if block_summary else 10
            for block_name, dominant, avg_conflict, n_loras_max, n_prefixes in block_summary:
                sym = symbols.get(dominant, "????")
                lbl = labels.get(dominant, dominant)
                if dominant == "weighted_sum":
                    detail = f"1 LoRA"
                elif dominant == "ties":
                    detail = f"{avg_conflict:.0%} conflict"
                else:
                    detail = f"{avg_conflict:.0%} conflict"
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

    def optimize_merge(self, model, lora_stack, output_strength, clip=None, clip_strength_multiplier=1.0, auto_strength="disabled", free_vram_between_passes="disabled", optimization_mode="per_prefix", cache_patches="enabled", compress_patches="non_ties"):
        """
        Main entry point. Two-pass streaming architecture:
        Pass 1: Compute diffs per-prefix, sample conflicts + magnitudes, discard diffs
        Decision: Finalize stats, auto-select params from lightweight accumulators
        Pass 2: Recompute diffs per-prefix, merge immediately, discard
        Peak memory: ~260MB (one prefix's diffs at a time) vs ~50GB (all diffs).
        """
        # Normalize stack format (standard tuples or LoRAStack dicts)
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
                "LORA OPTIMIZER - ANALYSIS REPORT\n"
                "=" * 50 + "\n\n"
                "Single LoRA detected — bypassing analysis.\n"
                f"  Name: {item['name']}\n"
                f"  Strength: {strength}\n"
                f"  Applied directly with output_strength={output_strength}\n"
                "\n" + "=" * 50
            )
            return (new_model, new_clip, report)

        # Check instance-level patch cache (survives ComfyUI re-execution
        # triggered by downstream seed changes or similar non-LoRA changes)
        cache_key = self._compute_cache_key(lora_stack, output_strength,
                                            clip_strength_multiplier, auto_strength,
                                            optimization_mode, compress_patches)
        if cache_patches == "enabled" and cache_key in self._merge_cache:
            model_patches, clip_patches, report, clip_strength_out = self._merge_cache[cache_key]
            new_model = model
            new_clip = clip
            if model is not None and len(model_patches) > 0:
                new_model = model.clone()
                new_model.add_patches(model_patches, output_strength)
            if clip is not None and len(clip_patches) > 0:
                new_clip = clip.clone()
                new_clip.add_patches(clip_patches, clip_strength_out)
            logging.info(f"[LoRA Optimizer] Using cached merge result ({len(model_patches)} model + {len(clip_patches)} CLIP patches)")
            return (new_model, new_clip, report)

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
        pair_accum = {(i, j): [0, 0] for i, j in pairs}  # [overlap, conflict]
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
            for (i, j), (ov, conf) in pair_conflicts.items():
                pair_accum[(i, j)][0] += ov
                pair_accum[(i, j)][1] += conf
            all_magnitude_samples.extend(mag_samples)

            # Store per-prefix stats for per_prefix optimization mode
            if len(partial_stats) > 0:
                # Number of LoRAs contributing to this prefix
                n_contributing = len(partial_stats)

                # Per-prefix conflict ratio
                pf_overlap = sum(ov for ov, _ in pair_conflicts.values())
                pf_conflict = sum(conf for _, conf in pair_conflicts.values())
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
                    "LoRAs may be incompatible with this model architecture.")

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

        # Pairwise conflict stats from accumulated counts
        total_overlap = 0
        total_conflict = 0
        pairwise_conflicts = []
        for i, j in pairs:
            pair_overlap, pair_conflict = pair_accum[(i, j)]
            total_overlap += pair_overlap
            total_conflict += pair_conflict
            ratio = pair_conflict / pair_overlap if pair_overlap > 0 else 0
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
            })
            logging.info(f"[LoRA Optimizer]   {pair_label} -> {ratio:.1%} conflict")

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

        logging.info(f"[LoRA Optimizer] Decision: {mode} (conflict {avg_conflict_ratio:.1%} "
                     f"{'>' if avg_conflict_ratio > 0.25 else '<='} 25% threshold)")
        if mode == "ties":
            logging.info(f"[LoRA Optimizer]   density={density:.2f}, sign_method={sign_method}")

        # Auto-strength adjustment
        auto_strength_info = None
        scale_ratios = {}
        if auto_strength == "enabled":
            new_strengths, strength_reasoning = self._compute_auto_strengths(active_loras, lora_stats)

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
                    patch = ("lora", (mat_up, mat_down, alpha_scaled, mid))
                    return (target_key, is_clip_key, patch, pf_mode, lora_prefix, pf_conflict, max(pf_n_loras, 1), False)
                return None

            # FULL-RANK PATH: compute diffs on GPU, merge
            diffs_list = []
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

            if len(diffs_list) == 0:
                return None

            # Re-check single-LoRA case (diff computation may have failed for some)
            if pf_mode == "weighted_sum" and len(diffs_list) <= 1:
                pass  # weighted_sum with 1 diff is fine
            elif len(diffs_list) <= 1 and pf_mode != "weighted_sum":
                pf_mode = "weighted_sum"

            merged_diff = self._merge_diffs(
                diffs_list, pf_mode,
                density=pf_density, majority_sign_method=pf_sign,
                compute_device=compute_device
            )
            diffs_list.clear()  # Free input diffs from GPU
            if merged_diff is None:
                return None
            # Compress full-rank diff to low-rank via SVD if requested
            # non_ties: skip compression on TIES prefixes (lossy); all: compress everything
            should_compress = (compress_rank > 0 and
                               (compress_patches == "all" or pf_mode != "ties"))
            if should_compress:
                patch = self._compress_to_lowrank(merged_diff, compress_rank)
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
            if patch[0] == "lora":
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
            prefix_decisions=prefix_decisions if optimization_mode == "per_prefix" else None
        )

        # Cache patches for re-use (single entry to limit memory)
        if cache_patches == "enabled":
            self._merge_cache = {cache_key: (model_patches, clip_patches, report, clip_strength_out)}
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

        return (new_model, new_clip, report)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoRAStack": LoRAStack,
    "LoRAOptimizer": LoRAOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAStack": "LoRA Stack",
    "LoRAOptimizer": "LoRA Optimizer",
}
