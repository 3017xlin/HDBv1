"""DDP utilities: init, cleanup, distributed check."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_ddp() -> tuple[int, int, int]:
    """Initialize NCCL DDP. Returns (rank, world_size, local_rank).

    Falls back to single-process (rank=0, world=1) if env vars absent.
    """
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def cleanup_ddp() -> None:
    if is_distributed():
        dist.destroy_process_group()
