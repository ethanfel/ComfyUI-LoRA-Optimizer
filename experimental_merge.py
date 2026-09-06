"""Opt-in, weight-only merge experiments. No ComfyUI or sampling dependency.

NP-LoRA: https://arxiv.org/html/2511.11051v3 (Eq. 12).
CT-Merging: https://arxiv.org/html/2607.20561v1 (Algorithm 1).
See docs/experimental-merging.md for numerical extensions and limitations.
"""

import math

import torch


VERSION = 1
MODES = ("np_lora", "ct_merge")
DEFAULTS = dict(enabled=True, np_lora=True, ct_merge=True,
                np_subject_slot=1, np_style_slot=2, np_strength=0.5,
                np_rank=0, np_energy=1.0, ct_common_rank=4,
                ct_residual_rank=16, ct_scale=1.0, trials_per_method=1)


class UnsupportedMerge(ValueError):
    """An experimental candidate cannot faithfully process these inputs."""


def options(value):
    """Canonical, JSON-safe options; no-op inputs have exactly one identity."""
    if value is None or value == {}:
        return None
    if not isinstance(value, dict):
        raise ValueError("Experimental options must come from LoRA Experimental Options.")
    unknown = set(value) - set(DEFAULTS) - {"version"}
    if unknown or value.get("version", VERSION) != VERSION:
        raise ValueError("Unknown experimental options/schema; recreate the options node.")
    cfg = {**DEFAULTS, **value, "version": VERSION}
    for key in ("enabled", "np_lora", "ct_merge"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"Experimental {key} must be a boolean.")
    if not cfg["enabled"] or not (cfg["np_lora"] or cfg["ct_merge"]):
        return None
    ranges = {"np_subject_slot": (1, 2), "np_style_slot": (1, 2),
              "np_rank": (0, 4096), "ct_common_rank": (0, 256),
              "ct_residual_rank": (1, 256), "trials_per_method": (1, 3)}
    for key, (low, high) in ranges.items():
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"Experimental {key} must be an integer in [{low}, {high}].")
    for key, low, high in (("np_strength", 0, 1000), ("np_energy", 0.01, 1),
                           ("ct_scale", 0.01, 10)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)):
            raise ValueError(f"Experimental {key} must be a finite number.")
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or not low <= cfg[key] <= high:
            raise ValueError(f"Experimental {key} must be in [{low}, {high}].")
    if cfg["np_lora"] and cfg["np_subject_slot"] == cfg["np_style_slot"]:
        raise ValueError("NP-LoRA subject and style must be different participants.")
    return cfg


def validate_method(mode, cfg):
    if mode not in MODES or not isinstance(cfg, dict) or cfg.get("version") != VERSION:
        raise ValueError("Missing/unsupported experimental method settings; re-run AutoTuner.")
    expected = ({"version", "subject_slot", "style_slot", "strength", "rank", "energy"}
                if mode == "np_lora" else {"version", "common_rank", "residual_rank", "scale"})
    if set(cfg) != expected:
        raise ValueError(f"Invalid settings for experimental {mode}; re-run AutoTuner.")
    translated = ({"np_" + k: v for k, v in cfg.items() if k != "version"}
                  if mode == "np_lora" else
                  {"ct_" + k: v for k, v in cfg.items() if k != "version"})
    options({**translated, "np_lora": mode == "np_lora", "ct_merge": mode == "ct_merge"})
    return dict(cfg)


def participants(active_loras):
    return [i for i, item in enumerate(active_loras) if not item.get("preserve", False)]


def validate_stack(mode, cfg, active_loras):
    validate_method(mode, cfg)
    ids = participants(active_loras)
    if mode == "np_lora" and len(ids) != 2:
        raise UnsupportedMerge("NP-LoRA needs exactly two active, non-preserved adapters; "
                               "subject/style slots refer to that pair in stack order.")
    if mode == "ct_merge" and len(ids) < 2:
        raise UnsupportedMerge("CT-Merging needs at least two active, non-preserved adapters.")
    return ids


def candidates(value, active_loras):
    """Bounded trials, independent of the stable grid and its heuristic."""
    cfg = options(value)
    if cfg is None:
        return [], []
    result, skipped = [], []
    for mode in MODES:
        if not cfg[mode]:
            continue
        if mode == "np_lora":
            params = dict(version=VERSION, subject_slot=cfg["np_subject_slot"],
                          style_slot=cfg["np_style_slot"], strength=cfg["np_strength"],
                          rank=cfg["np_rank"], energy=cfg["np_energy"])
        else:
            params = dict(version=VERSION, common_rank=cfg["ct_common_rank"],
                          residual_rank=cfg["ct_residual_rank"], scale=cfg["ct_scale"])
        try:
            validate_stack(mode, params, active_loras)
        except UnsupportedMerge as exc:
            skipped.append(f"{mode}: {exc}")
            continue
        seen = set()
        for multiplier in (1.0, 0.5, 2.0)[:cfg["trials_per_method"]]:
            p = dict(params)
            key, ceiling = ("strength", 1000) if mode == "np_lora" else ("scale", 10)
            p[key] = min(ceiling, max(0.01 if key == "scale" else 0, p[key] * multiplier))
            if p[key] in seen:
                continue
            seen.add(p[key])
            result.append(dict(merge_mode=mode, experimental=p, sparsification="disabled",
                               sparsification_density=0.7, dare_dampening=0.0,
                               merge_refinement="none", auto_strength="disabled",
                               optimization_mode="global", strategy_set="full"))
    if result:
        # Explicit additive comparison, without changing the stable grid.
        baseline = {k: v for k, v in result[0].items() if k != "experimental"}
        baseline.update(merge_mode="weighted_sum", experimental_baseline=True)
        result.insert(0, baseline)
    return result, skipped


def _svd(matrix, rank=0):
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except torch.linalg.LinAlgError as exc:
        raise UnsupportedMerge("Experimental SVD did not converge; try CPU or stable merging.") from exc
    # Do not invent directions in a zero/rank-deficient update.
    tol = max(matrix.shape) * torch.finfo(matrix.dtype).eps * s[0] if s.numel() else 0
    n = int((s > tol).sum().item())
    if rank:
        n = min(n, rank)
    return u[:, :n], s[:n], vh[:n]


def _polar(matrix):
    # The partial polar factor is unique on the supported subspace. Completing
    # zero singular directions would inject arbitrary signal on cancelling or
    # overlapping adapters. Also handles rectangular/overcomplete inputs.
    u, _, vh = _svd(matrix)
    return u @ vh


def factor_svd(up, down, rank=0):
    """Exact QR + small-core SVD of up @ down (no dense SVD or randomness)."""
    qb, rb = torch.linalg.qr(up, mode="reduced")
    qa, ra = torch.linalg.qr(down.T, mode="reduced")
    u, s, vh = _svd(rb @ ra.T)
    tol = max(up.shape[0], down.shape[1]) * torch.finfo(up.dtype).eps * s[0] if s.numel() else 0
    n = int((s > tol).sum().item())
    if rank:
        n = min(n, rank)
    return qb @ u[:, :n], s[:n], vh[:n] @ qa.T


def merge(diffs_with_weights, mode, cfg, source_indices=None, role_indices=None, svd_factors=None):
    """Merge weighted 2D updates; bias/norm vectors remain strictly additive."""
    validate_method(mode, cfg)
    ref = diffs_with_weights[0][0]
    dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
    weighted = []
    for diff, weight in diffs_with_weights:
        if diff.shape != ref.shape or not math.isfinite(float(weight)):
            raise UnsupportedMerge(f"{mode}: mismatched shapes or non-finite strength.")
        d = diff.to(device=ref.device, dtype=dtype) * weight
        if not torch.isfinite(d).all():
            raise UnsupportedMerge(f"{mode}: non-finite input delta.")
        weighted.append(d)
    indices = list(range(len(weighted))) if source_indices is None else source_indices
    def task_svd(position, rank=0):
        pair = (svd_factors or {}).get(indices[position])
        if pair is None:
            return _svd(weighted[position], rank)
        return factor_svd(*(f.to(device=ref.device, dtype=dtype) for f in pair), rank=rank)
    if ref.ndim <= 1 or len(weighted) == 1:
        return sum(weighted, torch.zeros_like(weighted[0]))
    if ref.ndim != 2:
        raise UnsupportedMerge(f"{mode}: shared spatial convolution tensors are unsupported; "
                               "use a stable merge mode.")
    if mode == "np_lora":
        roles = [0, 1] if role_indices is None else role_indices
        by_id = dict(zip(indices, weighted))
        content = by_id.get(roles[cfg["subject_slot"] - 1])
        style = by_id.get(roles[cfg["style_slot"] - 1])
        if content is None or style is None:
            return sum(weighted, torch.zeros_like(weighted[0]))
        if len(weighted) != 2:
            raise UnsupportedMerge("NP-LoRA requires a subject/style pair.")
        if cfg["strength"] == 0:
            return style + content
        _, s, vh = task_svd(indices.index(roles[cfg["style_slot"] - 1]), cfg["rank"])
        if s.numel() and cfg["energy"] < 1:
            n = int(torch.searchsorted(s.square().cumsum(0), s.square().sum() * cfg["energy"]).item()) + 1
            vh = vh[:n]
        return style + content - (cfg["strength"] / (1 + cfg["strength"])) * ((content @ vh.T) @ vh)

    # Signed strengths are applied BEFORE SVD. Magnitudes enter the RMS
    # coefficients and signs remain in paired left/right directions.
    tasks = [task_svd(i, cfg["residual_rank"]) for i in range(len(weighted))]
    tasks = [(d, u, s, vh) for d, (u, s, vh) in zip(weighted, tasks) if s.numel()]
    if not tasks:
        return torch.zeros_like(weighted[0])
    if len(tasks) == 1:
        return tasks[0][0]  # unique/nonzero contributor: additive side policy
    # SVD of concatenated thin bases equals eigendecomposition of the mean
    # projector, without allocating an output_width x output_width projector.
    uc, support, _ = _svd(torch.cat([t[1] for t in tasks], dim=1))
    k = min(cfg["common_rank"], support.numel())
    # Do not split a tied consensus eigenspace: its arbitrary SVD orientation
    # would otherwise make an unordered stack depend on adapter order/device.
    if 0 < k < support.numel():
        tol = 8 * torch.finfo(dtype).eps * support[0]
        while k > 0 and abs(support[k - 1] - support[k]) <= tol:
            k -= 1
    uc = uc[:, :k]
    left, right, scales = [], [], []
    if uc.shape[1]:
        response = sum((uc.T @ t[0] for t in tasks)) / len(tasks)
        p, sc, vhc = _svd(response)
        if sc.numel():
            left.append(uc @ p)
            right.append(vhc.T)
            scales.append(sc.square().mean().sqrt().expand(sc.numel()))
    for _, u, s, vh in tasks:
        residual = u - uc @ (uc.T @ u)
        # Numerical cancellation must not become a unit direction in polar().
        residual = torch.where(residual.abs() < 8 * torch.finfo(dtype).eps,
                               torch.zeros_like(residual), residual)
        live = residual.norm(dim=0) > 8 * torch.finfo(dtype).eps
        # A fully common direction has no residual contribution; discard both
        # sides rather than letting its unused right vector distort polar().
        if not live.any():
            continue
        residual = residual[:, live]
        left.append(residual)
        right.append(vh.T[:, live])
        scales.append(s.square().mean().sqrt().expand(int(live.sum().item())))
    if not left:
        return torch.zeros_like(weighted[0])
    ul = _polar(torch.cat(left, dim=1))
    vr = _polar(torch.cat(right, dim=1))
    return cfg["scale"] * ((ul * torch.cat(scales)) @ vr.T)


class LoRAExperimentalOptions:
    RETURN_TYPES = ("LORA_EXPERIMENTAL_OPTIONS",)
    RETURN_NAMES = ("experimental_options",)
    FUNCTION = "build_options"
    CATEGORY = "LoRA Optimizer/Experimental"
    DESCRIPTION = ("Opt-in NP-LoRA and CT-Merging trials for AutoTuner. Stable choices remain "
                   "available. H3 video/audio quality is unvalidated. No sampling or training.")

    @classmethod
    def INPUT_TYPES(cls):
        fields = {k: ("BOOLEAN", {"default": DEFAULTS[k]})
                  for k in ("enabled", "np_lora", "ct_merge")}
        for key, low, high in (("np_subject_slot", 1, 2), ("np_style_slot", 1, 2),
                               ("np_rank", 0, 4096), ("ct_common_rank", 0, 256),
                               ("ct_residual_rank", 1, 256), ("trials_per_method", 1, 3)):
            fields[key] = ("INT", {"default": DEFAULTS[key], "min": low, "max": high})
        for key, low, high in (("np_strength", 0, 1000), ("np_energy", 0.01, 1), ("ct_scale", 0.01, 10)):
            fields[key] = ("FLOAT", {"default": DEFAULTS[key], "min": low, "max": high, "step": 0.01})
        for key in ("np_subject_slot", "np_style_slot"):
            fields[key][1]["tooltip"] = "Position among active, non-preserved adapters, in stack order. NP requires exactly two."
        fields["np_rank"][1]["tooltip"] = "Maximum style-subspace rank; 0 uses all numerically supported directions."
        fields["np_energy"][1]["tooltip"] = "Energy retained within the rank-capped style subspace. 1 keeps all."
        fields["ct_scale"][1]["tooltip"] = "Scale for shared matrix merges only. Unique targets, bias/norm vectors and preserve overlays remain additive."
        fields["ct_common_rank"][1]["tooltip"] = "Maximum common rank per layer; reduced for numerical rank limits or tied consensus boundaries."
        fields["trials_per_method"][1]["tooltip"] = "Extra trials per enabled method (1–3), plus one additive baseline. Does not consume stable top_n slots."
        return {"required": fields}

    def build_options(self, **kwargs):
        return (options(kwargs),)
