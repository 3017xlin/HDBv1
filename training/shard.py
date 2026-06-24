"""Grouped shard builder for variable-L sub-bin batching."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

SUB_BIN_L: dict[str, int] = {
    '0-19_easy': 8192, '0-19_hard': 16384,
    '20-39_easy': 24576, '20-39_hard': 32768,
    '40-59_easy': 40960, '40-59_hard': 49152,
    '60-79_easy': 57344, '60-79_hard': 65536,
    '80-123_easy': 73728, '80-123_hard': 81920,
}

BATCH_SIZES: dict[str, int] = {
    '0-19_easy': 4, '0-19_hard': 4,
    '20-39_easy': 4, '20-39_hard': 4,
    '40-59_easy': 2, '40-59_hard': 2,
    '60-79_easy': 2, '60-79_hard': 1,
    '80-123_easy': 1, '80-123_hard': 1,
}

BatchTuple = tuple[list[int], str, int, int]


def build_grouped_shard(
    case_ids_this_epoch: list[int],
    sub_bin_map: dict[int, str],
    epoch: int,
    rng: np.random.Generator,
) -> list[BatchTuple]:
    """Build a list of batches grouped by sub-bin, then globally shuffled.

    Each batch contains cases from a single sub-bin so all items share the
    same L value (no padding tokens needed).

    Returns
    -------
    list of (batch_case_ids, sub_bin, L, B) tuples.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for cid in case_ids_this_epoch:
        sb = sub_bin_map[cid]
        groups[sb].append(cid)

    batches: list[BatchTuple] = []
    for sb, ids in groups.items():
        B = BATCH_SIZES[sb]
        L = SUB_BIN_L[sb]

        arr = np.array(ids)
        rng.shuffle(arr)
        ids_shuffled = arr.tolist()

        for lo in range(0, len(ids_shuffled), B):
            chunk = ids_shuffled[lo: lo + B]
            if len(chunk) < B:
                chunk = chunk + [ids_shuffled[0]] * (B - len(chunk))
            batches.append((chunk, sb, L, B))

    if epoch == 0:
        batches.sort(key=lambda t: t[2])
    else:
        rng.shuffle(batches)
    return batches
