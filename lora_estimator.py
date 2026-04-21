"""LoRA merge estimator — case-based retrieval over Phase 1 stats.

Predicts optimal merge configs from the HF community cache dataset via
k-NN retrieval on combo-level feature vectors aggregated from per-prefix
conflict / magnitude / subspace stats.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

ESTIMATOR_INDEX_VERSION = "1.0.0"
HF_REPO_ID = "ethanfel/lora-optimizer-community-cache"

_log = logging.getLogger(__name__)


class EstimatorFeatureExtractor:
    """Build a fixed-dim combo feature vector from Phase 1 analysis output."""

    PAIR_STATS = [
        "overlap", "conflict", "dot", "norm_a_sq", "norm_b_sq",
        "weighted_total", "weighted_conflict", "expected_conflict",
        "excess_conflict", "subspace_overlap", "subspace_weight",
    ]
    LORA_STATS = ["norm_sq", "rank", "strength_sign"]
    FAMILY_LABELS = [
        "zimage", "wan", "flux", "qwen_image", "acestep", "ltx",
        "sdxl", "sd15", "unknown",
    ]
    AGG_FUNCS = ["mean", "p90", "max", "std"]
    # Bucket 0 = pair (2), 1 = triple (3), 2 = quad (4), 3 = 5+ LoRAs.
    COMBO_SIZE_BUCKETS = [2, 3, 4, 5]

    PAIR_DIM = len(PAIR_STATS) * len(AGG_FUNCS)   # 44
    LORA_DIM = len(LORA_STATS) * len(AGG_FUNCS)   # 12
    WORST_DIM = len(PAIR_STATS)                   # 11
    SIZE_DIM = len(COMBO_SIZE_BUCKETS)            # 4
    FAM_DIM = len(FAMILY_LABELS)                  # 9
    DIM = PAIR_DIM + LORA_DIM + WORST_DIM + SIZE_DIM + FAM_DIM  # 80

    def featurize(self, phase1: dict) -> np.ndarray:
        parts = [
            self._aggregate_pairs(phase1.get("pair_stats", {})),
            self._aggregate_loras(phase1.get("lora_stats", {})),
            self._worst_pair(phase1.get("pair_stats", {})),
            self._size_onehot(phase1.get("combo_size", 2)),
            self._family_onehot(phase1.get("base_model_family", "unknown")),
        ]
        vec = np.concatenate(parts).astype(np.float32)
        assert vec.shape == (self.DIM,), f"expected {self.DIM}, got {vec.shape}"
        return vec

    def _aggregate_pairs(self, pair_stats: dict) -> np.ndarray:
        buckets = {s: [] for s in self.PAIR_STATS}
        for pair_data in pair_stats.values():
            per_prefix = pair_data.get("per_prefix", {})
            for stats in per_prefix.values():
                for s in self.PAIR_STATS:
                    v = stats.get(s, 0.0)
                    if isinstance(v, (int, float)):
                        buckets[s].append(float(v))
        return self._stack_aggs(buckets, self.PAIR_STATS, self.PAIR_DIM)

    def _aggregate_loras(self, lora_stats: dict) -> np.ndarray:
        buckets = {s: [] for s in self.LORA_STATS}
        for lora_data in lora_stats.values():
            per_prefix = lora_data.get("per_prefix", {})
            for stats in per_prefix.values():
                for s in self.LORA_STATS:
                    v = stats.get(s)
                    if isinstance(v, (int, float)):
                        buckets[s].append(float(v))
        return self._stack_aggs(buckets, self.LORA_STATS, self.LORA_DIM)

    def _stack_aggs(self, buckets: dict, stat_names: list, out_dim: int) -> np.ndarray:
        out = np.zeros(out_dim, dtype=np.float32)
        for i, s in enumerate(stat_names):
            vals = np.asarray(buckets[s], dtype=np.float32) if buckets[s] else np.zeros(1, dtype=np.float32)
            out[i * 4 + 0] = vals.mean()
            out[i * 4 + 1] = np.quantile(vals, 0.9)
            out[i * 4 + 2] = vals.max()
            out[i * 4 + 3] = vals.std()
        return out

    def _worst_pair(self, pair_stats: dict) -> np.ndarray:
        if not pair_stats:
            return np.zeros(self.WORST_DIM, dtype=np.float32)
        worst_key = None
        worst_ec = -np.inf
        for key, pair_data in pair_stats.items():
            per_prefix = pair_data.get("per_prefix", {})
            ecs = [stats.get("excess_conflict", 0.0) for stats in per_prefix.values()]
            mean_ec = float(np.mean(ecs)) if ecs else 0.0
            if mean_ec > worst_ec:
                worst_ec = mean_ec
                worst_key = key
        if worst_key is None:
            return np.zeros(self.WORST_DIM, dtype=np.float32)
        per_prefix = pair_stats[worst_key].get("per_prefix", {})
        out = np.zeros(self.WORST_DIM, dtype=np.float32)
        for i, s in enumerate(self.PAIR_STATS):
            vals = [stats.get(s, 0.0) for stats in per_prefix.values()]
            out[i] = float(np.mean(vals)) if vals else 0.0
        return out

    def _size_onehot(self, size: int) -> np.ndarray:
        out = np.zeros(self.SIZE_DIM, dtype=np.float32)
        idx = min(max(0, int(size) - 2), self.SIZE_DIM - 1)
        out[idx] = 1.0
        return out

    def _family_onehot(self, family: str) -> np.ndarray:
        out = np.zeros(self.FAM_DIM, dtype=np.float32)
        if family in self.FAMILY_LABELS:
            out[self.FAMILY_LABELS.index(family)] = 1.0
        else:
            out[self.FAMILY_LABELS.index("unknown")] = 1.0
        return out

    # Slice helpers — used by tests and by downstream analytics.
    def worst_pair_slice(self, vec: np.ndarray) -> np.ndarray:
        start = self.PAIR_DIM + self.LORA_DIM
        return vec[start:start + self.WORST_DIM]

    def size_slice(self, vec: np.ndarray) -> np.ndarray:
        start = self.PAIR_DIM + self.LORA_DIM + self.WORST_DIM
        return vec[start:start + self.SIZE_DIM]

    def family_slice(self, vec: np.ndarray) -> np.ndarray:
        start = self.PAIR_DIM + self.LORA_DIM + self.WORST_DIM + self.SIZE_DIM
        return vec[start:start + self.FAM_DIM]


def _fetch_hf_head_sha(repo_id: str = HF_REPO_ID) -> str:
    """Return the current HEAD commit SHA of the dataset repo. Raises on network failure."""
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo_id).sha


def ensure_index_fresh(
    index_dir: Union[str, Path],
    rebuild_fn: Callable[[], None],
    mode: str = "auto",
    repo_id: str = HF_REPO_ID,
) -> None:
    """Rebuild the index if stale.

    mode:
      - "auto":  compare cached hf_commit_sha to HF HEAD; rebuild on mismatch.
                 On network failure, fall back to cached index with a warning.
      - "force": always call rebuild_fn().
      - "skip":  never call rebuild_fn().
    """
    index_dir = Path(index_dir)
    if mode == "skip":
        return
    if mode == "force":
        rebuild_fn()
        return
    if mode != "auto":
        raise ValueError(f"unknown freshness mode: {mode!r}")

    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        rebuild_fn()
        return
    try:
        cached = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        _log.warning("[LoRA Estimator] meta.json unreadable (%s) — rebuilding", e)
        rebuild_fn()
        return
    # Schema-version mismatch trumps SHA: a new feature layout can't read an old pickle.
    cached_version = cached.get("estimator_index_version")
    if cached_version != ESTIMATOR_INDEX_VERSION:
        _log.info(
            "[LoRA Estimator] Index schema changed (%s → %s), rebuilding…",
            cached_version or "?", ESTIMATOR_INDEX_VERSION,
        )
        rebuild_fn()
        return
    try:
        current_sha = _fetch_hf_head_sha(repo_id)
    except Exception as e:
        _log.warning("[LoRA Estimator] HF SHA check failed — using cached index: %s", e)
        return
    cached_sha = cached.get("hf_commit_sha")
    if cached_sha != current_sha:
        _log.info(
            "[LoRA Estimator] Dataset updated (%s → %s), rebuilding index…",
            (cached_sha or "?")[:8], (current_sha or "?")[:8],
        )
        rebuild_fn()


def _config_key(config: dict) -> str:
    """Stable hash for a merge config dict — used to dedupe across neighbors."""
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()


class LoRAEstimator:
    """Retrieve k-NN combos, aggregate their candidates, emit a tuner_data dict."""

    def __init__(self, features: np.ndarray, labels: list,
                 zscore: tuple[np.ndarray, np.ndarray]):
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = labels
        mean, std = zscore
        self.zscore_mean = np.asarray(mean, dtype=np.float32)
        # Guard against constant dims (std==0) so downstream divides never NaN.
        std_arr = np.asarray(std, dtype=np.float32)
        self.zscore_std = np.where(std_arr < 1e-8, 1.0, std_arr)

    @classmethod
    def from_disk(cls, index_path: Union[str, Path]) -> "LoRAEstimator":
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        return cls(
            features=data["features"],
            labels=data["labels"],
            zscore=data["zscore"],
        )

    @classmethod
    def from_memory(cls, features, labels, zscore) -> "LoRAEstimator":
        """Test helper — normalise features the same way from_disk does."""
        features = np.asarray(features, dtype=np.float32)
        mean, std = zscore
        std_arr = np.asarray(std, dtype=np.float32)
        safe_std = np.where(std_arr < 1e-8, 1.0, std_arr)
        X_norm = (features - np.asarray(mean, dtype=np.float32)) / safe_std
        return cls(features=X_norm, labels=labels, zscore=(mean, std))

    def retrieve(self, feature_vec: np.ndarray, base_model_family: str,
                 combo_size: int, k: int = 5) -> list[dict]:
        """Return up to k nearest neighbors matching the family+size filter."""
        from sklearn.neighbors import NearestNeighbors

        mask = np.array([
            lbl.get("base_model_family") == base_model_family
            and lbl.get("combo_size") == combo_size
            for lbl in self.labels
        ])
        if not mask.any():
            return []
        subset_features = self.features[mask]
        subset_labels = [lbl for lbl, keep in zip(self.labels, mask) if keep]

        q = (np.asarray(feature_vec, dtype=np.float32) - self.zscore_mean) / self.zscore_std
        q = q.reshape(1, -1)

        n_neighbors = min(k, len(subset_features))
        sub_nn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
        sub_nn.fit(subset_features)
        distances, indices = sub_nn.kneighbors(q)
        return [
            {"distance": float(d), "label": subset_labels[int(i)]}
            for d, i in zip(distances[0], indices[0])
        ]

    @staticmethod
    def aggregate_candidates(neighbors: list[dict], top_n: int = 3) -> list[dict]:
        """Inverse-distance-weighted pooling of candidate scores across neighbors."""
        if not neighbors:
            return []
        pool: dict[str, dict[str, Any]] = {}
        for n in neighbors:
            w = 1.0 / (1.0 + float(n.get("distance", 0.0)))
            for cand in n.get("label", {}).get("candidates", []):
                cfg = cand.get("config")
                if not cfg:
                    continue
                key = _config_key(cfg)
                entry = pool.setdefault(key, {
                    "config": cfg,
                    "score_final": 0.0,
                    "score_heuristic": 0.0,
                    "score_measured": 0.0,
                    "weight_sum": 0.0,
                })
                entry["score_final"] += w * float(cand.get("score_final", 0.0))
                entry["score_heuristic"] += w * float(cand.get("score_heuristic", 0.0))
                entry["score_measured"] += w * float(cand.get("score_measured", 0.0))
                entry["weight_sum"] += w
        # Normalise into weighted averages so scores stay on the [0, 1] source scale.
        normalised = []
        for e in pool.values():
            w = e["weight_sum"] if e["weight_sum"] > 0 else 1.0
            normalised.append({
                "config": e["config"],
                "score_final": e["score_final"] / w,
                "score_heuristic": e["score_heuristic"] / w,
                "score_measured": e["score_measured"] / w,
            })
        ranked = sorted(normalised, key=lambda e: e["score_final"], reverse=True)
        out = []
        for rank, e in enumerate(ranked[:top_n], start=1):
            out.append({
                "rank": rank,
                "config": e["config"],
                "score_final": e["score_final"],
                "score_heuristic": e["score_heuristic"],
                "score_measured": e["score_measured"],
                "score_external": None,
                "metrics": {},
                "external_details": None,
            })
        return out

    @staticmethod
    def emit_tuner_data(top_n: list[dict], source_loras: list[dict],
                        analysis_summary: dict,
                        normalize_keys: str = "disabled",
                        architecture_preset: str = "auto",
                        auto_strength_floor: float = -1.0,
                        decision_smoothing: float = 0.25,
                        lora_hash: Optional[str] = None) -> dict:
        """Produce a tuner_data dict for settings_source=from_tuner_data."""
        return {
            "version": 1,
            "lora_hash": lora_hash or "estimator",
            "source_loras": source_loras,
            "normalize_keys": normalize_keys,
            "architecture_preset": architecture_preset,
            "auto_strength_floor": auto_strength_floor,
            "decision_smoothing": decision_smoothing,
            "analysis_summary": analysis_summary,
            "top_n": top_n,
            "_estimator": True,
        }
