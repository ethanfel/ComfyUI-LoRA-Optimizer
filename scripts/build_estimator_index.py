#!/usr/bin/env python3
"""Build the estimator k-NN index from the HF community cache dataset.

Offline pipeline: snapshot_download config/pair/lora JSONs, featurize,
z-score, fit NearestNeighbors, pickle to disk with a meta.json sidecar.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Allow running as a script without PYTHONPATH tweaks.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lora_estimator import EstimatorFeatureExtractor, ESTIMATOR_INDEX_VERSION  # noqa: E402


# Architectures that are single tokens in HF filenames.
_SIMPLE_ARCHS = {"dit", "llm", "unet"}
# Compound (two-token) arch names. Order matters: longer first.
_COMPOUND_ARCHS = ("acestep_dit", "sd_unet")


def parse_config_name(fname: str):
    """Extract (sorted_hashes, arch_preset) from 'h1_h2_..._{arch}.config.json'.

    arch is the trailing token (or trailing two tokens for compound names
    like `sd_unet`).
    """
    stem = fname
    if stem.endswith(".config.json"):
        stem = stem[: -len(".config.json")]
    for compound in _COMPOUND_ARCHS:
        suffix = "_" + compound
        if stem.endswith(suffix):
            return sorted(stem[: -len(suffix)].split("_")), compound
    parts = stem.split("_")
    arch = parts[-1]
    if arch in _SIMPLE_ARCHS:
        return sorted(parts[:-1]), arch
    # Unknown arch token — treat whole trailing token as arch anyway.
    return sorted(parts[:-1]), arch


def _load_json(path: Path):
    return json.loads(path.read_text())


def load_phase1_for_config(dataset_dir: Path, sorted_hashes) -> tuple[dict, dict]:
    """Assemble phase1 {pair_stats, lora_stats} by reading cache JSONs on disk."""
    pair_stats = {}
    lora_stats = {}
    for i, h in enumerate(sorted_hashes):
        lora_path = dataset_dir / "lora" / f"{h}.lora.json"
        if lora_path.exists():
            raw = _load_json(lora_path)
            # The cache format stores stats under `per_prefix`.
            lora_stats[i] = {"per_prefix": raw.get("per_prefix", {})}
    for i, ha in enumerate(sorted_hashes):
        for j, hb in enumerate(sorted_hashes):
            if i >= j:
                continue
            key_a, key_b = sorted([ha, hb])
            pair_path = dataset_dir / "pair" / f"{key_a}_{key_b}.pair.json"
            if pair_path.exists():
                raw = _load_json(pair_path)
                pair_stats[(i, j)] = {"per_prefix": raw.get("per_prefix", {})}
    return pair_stats, lora_stats


def build_index(local_dataset_dir, out_dir, hf_commit_sha: str) -> dict:
    """Build index.pkl + meta.json in out_dir from a local HF dataset snapshot."""
    dataset_dir = Path(local_dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = EstimatorFeatureExtractor()

    feature_rows = []
    labels = []
    for cfg_path in sorted((dataset_dir / "config").glob("*.config.json")):
        cfg = _load_json(cfg_path)
        sorted_hashes, _arch = parse_config_name(cfg_path.name)
        pair_stats, lora_stats = load_phase1_for_config(dataset_dir, sorted_hashes)
        if len(sorted_hashes) >= 2 and not pair_stats:
            # No pair caches available — skip; featurisation would be all zeros.
            continue
        phase1 = {
            "pair_stats": pair_stats,
            "lora_stats": lora_stats,
            "combo_size": len(sorted_hashes),
            "base_model_family": cfg.get("base_model_family", "unknown"),
        }
        vec = extractor.featurize(phase1)
        feature_rows.append(vec)
        labels.append({
            "arch_preset": cfg.get("arch_preset"),
            "base_model_family": cfg.get("base_model_family", "unknown"),
            "combo_size": len(sorted_hashes),
            "config": cfg.get("config"),
            "candidates": cfg.get("candidates", []),
            "score_final": cfg.get("score", 0.0),
            "lora_content_hashes": sorted_hashes,
        })

    if not feature_rows:
        raise RuntimeError("No valid configs found to build index")

    X = np.stack(feature_rows).astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_norm = (X - mean) / std

    nn = NearestNeighbors(metric="cosine", n_neighbors=min(10, len(X)))
    nn.fit(X_norm)

    with open(out_dir / "index.pkl", "wb") as f:
        pickle.dump({
            "index": nn,
            "features": X_norm,
            "raw_features": X,
            "labels": labels,
            "zscore": (mean.tolist(), std.tolist()),
        }, f)

    meta = {
        "hf_commit_sha": hf_commit_sha,
        "estimator_index_version": ESTIMATOR_INDEX_VERSION,
        "n_samples": len(feature_rows),
        "feature_dim": int(X.shape[1]),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser(description="Build estimator k-NN index from HF community cache")
    p.add_argument("--out", default="./models/estimator", help="output directory for index.pkl + meta.json")
    p.add_argument("--local-dataset-dir", default=None,
                   help="Use a local snapshot (skip HF download)")
    p.add_argument("--hf-repo", default="ethanfel/lora-optimizer-community-cache")
    args = p.parse_args()

    if args.local_dataset_dir:
        sha = "local"
        ds_dir = args.local_dataset_dir
    else:
        from huggingface_hub import HfApi, snapshot_download
        info = HfApi().dataset_info(args.hf_repo)
        sha = info.sha
        ds_dir = snapshot_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            allow_patterns=["config/*", "pair/*", "lora/*"],
        )

    meta = build_index(ds_dir, args.out, sha)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
