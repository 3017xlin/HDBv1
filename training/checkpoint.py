"""Checkpoint save / load with optimizer, scheduler, and SWA state."""
from __future__ import annotations

import os
from collections import OrderedDict

import torch
import torch.nn as nn


def _model_state_bf16(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    inner = model.module if hasattr(model, "module") else model
    sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v in inner.state_dict().items():
        if v.is_floating_point():
            sd[k] = v.detach().to("cpu", dtype=torch.bfloat16)
        else:
            sd[k] = v.detach().to("cpu")
    return sd


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str,
    swa_manager=None,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": _model_state_bf16(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "swa_n_snapshots": len(swa_manager.snapshots) if swa_manager is not None else 0,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> int:
    """Load checkpoint and return the saved epoch number.

    Model weights are cast from bf16 back to float32 on load.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    inner = model.module if hasattr(model, "module") else model
    sd = ckpt["model_state_dict"]
    for k in sd:
        if sd[k].is_floating_point():
            sd[k] = sd[k].to(torch.float32)
    inner.load_state_dict(sd)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"])
