"""LoRA merge estimator — case-based retrieval over Phase 1 stats.

Predicts optimal merge configs from the HF community cache dataset via
k-NN retrieval on combo-level feature vectors aggregated from per-prefix
conflict / magnitude / subspace stats.
"""
from __future__ import annotations

import numpy as np

ESTIMATOR_INDEX_VERSION = "1.0.0"
HF_REPO_ID = "ethanfel/lora-optimizer-community-cache"


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
