"""Curriculum learning scheduler with sampling probability annealing."""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SUB_BIN_ORDER: list[str] = [
    '0-19_easy', '0-19_hard',
    '20-39_easy', '20-39_hard',
    '40-59_easy', '40-59_hard',
    '60-79_easy', '60-79_hard',
    '80-123_easy', '80-123_hard',
]


class CurriculumScheduler:
    """Per-epoch sampling count scheduler using temperature-based annealing.

    All 700 train cases are always available. Each epoch, per-bin sampling
    counts are determined by difficulty-weighted probabilities that anneal
    from uniform (high temperature) toward hard-biased (low temperature).
    """

    def __init__(
        self,
        sub_bin_names_sorted: list[str],
        cases_per_rank_per_bin: dict[str, list[int]],
        T_start: float = 3.0,
        T_end: float = 1.0,
        total_epochs: int = 400,
        floor_ratio: float = 0.3,
        floor_min_abs: int = 3,
    ) -> None:
        self.sub_bin_names = list(sub_bin_names_sorted)
        self.cases_per_bin = {
            k: list(v) for k, v in cases_per_rank_per_bin.items()
        }
        self.T_start = T_start
        self.T_end = T_end
        self.total_epochs = total_epochs
        self.floor_ratio = floor_ratio
        self.floor_min_abs = floor_min_abs
        self.n_bins = len(self.sub_bin_names)

    def temperature(self, epoch: int) -> float:
        return self.T_start - (self.T_start - self.T_end) * epoch / self.total_epochs

    def get_epoch_samples(self, epoch: int, rng: np.random.Generator) -> list[int]:
        T = self.temperature(epoch)
        sampled: list[int] = []
        per_bin_counts: list[int] = []

        for i, sb in enumerate(self.sub_bin_names):
            pool = self.cases_per_bin[sb]
            n_available = len(pool)
            if n_available == 0:
                per_bin_counts.append(0)
                continue

            difficulty = i / max(self.n_bins - 1, 1)
            weight = math.exp(-difficulty * T)

            floor_min = max(self.floor_min_abs, math.ceil(self.floor_ratio * n_available))
            n_samples = max(floor_min, round(n_available * weight))
            n_samples = min(n_samples, n_available)

            chosen_idx = rng.choice(n_available, size=n_samples, replace=False)
            sampled.extend(pool[j] for j in chosen_idx)
            per_bin_counts.append(n_samples)

        log.info(
            'curriculum epoch=%d T=%.3f total=%d per_bin=%s',
            epoch, T, len(sampled),
            ' '.join(f'{sb}:{c}' for sb, c in zip(self.sub_bin_names, per_bin_counts)),
        )
        return sampled
