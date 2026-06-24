"""Pin-memory loader for pre-processed HDB cache .pt files."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def _identity_collate(batch):
    return batch[0]


class _CachePTDataset(Dataset):
    def __init__(self, cache_dir: str, case_ids: list[str]):
        self.cache_dir = Path(cache_dir)
        self.case_ids = list(case_ids)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[str, dict]:
        cid = self.case_ids[idx]
        pt = torch.load(
            self.cache_dir / f"{cid}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for k, v in pt.items():
            if isinstance(v, torch.Tensor) and k != "encoder_pool":
                pt[k] = v.pin_memory()
        pt["_case_id"] = cid
        return cid, pt


def load_cases_pinned(
    cache_dir: str,
    case_ids: list[str],
    num_workers: int = 30,
    rank: int = 0,
) -> dict[str, dict]:
    """Load and pin-memory all *case_ids* from *cache_dir*.

    Uses a DataLoader with *num_workers* for parallel I/O but iterates
    sequentially (batch_size=1) so at most one extra copy is in flight,
    avoiding OOM from double-buffering all cases at once.

    Returns ``{case_id: pt_dict}`` with every tensor pinned.
    """
    dataset = _CachePTDataset(cache_dir, case_ids)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_identity_collate,
        persistent_workers=False,
    )

    all_data: dict[str, dict] = {}
    total = len(case_ids)
    for i, (cid, pt) in enumerate(loader):
        all_data[cid] = pt
        if rank == 0 and (i + 1) % 25 == 0:
            print(
                f"[loader] pinned {i + 1}/{total} cases", flush=True
            )

    if rank == 0:
        print(f"[loader] pinned {total}/{total} cases (done)", flush=True)
    return all_data
