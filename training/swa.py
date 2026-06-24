"""SWA snapshot manager.

Snapshots every epoch during the last ``swa_window`` epochs (default 300-399).
After training, average all snapshots and return a bf16 state dict.
"""
from __future__ import annotations

import gc
from collections import OrderedDict

import torch
import torch.nn as nn


def _state_dict_cpu_bf16(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    inner = model.module if hasattr(model, "module") else model
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v in inner.state_dict().items():
        if v.is_floating_point():
            out[k] = v.detach().to("cpu", dtype=torch.bfloat16).clone()
        else:
            out[k] = v.detach().to("cpu").clone()
    return out


class SWAManager:
    def __init__(
        self,
        swa_window: int = 100,
        num_epochs: int = 400,
        every_epochs: int = 1,
    ):
        self.swa_window = int(swa_window)
        self.num_epochs = int(num_epochs)
        self.every = int(every_epochs)
        self.snapshots: list[OrderedDict[str, torch.Tensor]] = []
        self.window_start = self.num_epochs - self.swa_window

    def accumulate(self, model: nn.Module, epoch: int) -> bool:
        """Take a snapshot if *epoch* falls within the SWA window."""
        if epoch < self.window_start:
            return False
        if (epoch - self.window_start) % self.every != 0:
            return False
        self.snapshots.append(_state_dict_cpu_bf16(model))
        return True

    def has_snapshots(self) -> bool:
        return len(self.snapshots) > 0

    def get_averaged(self) -> OrderedDict[str, torch.Tensor]:
        """Average all accumulated snapshots and return a bf16 state dict."""
        if not self.snapshots:
            raise RuntimeError("No SWA snapshots to average.")
        first = self.snapshots[0]
        averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in first.items():
            if v.is_floating_point():
                averaged[k] = torch.zeros_like(v, dtype=torch.float32)
            else:
                averaged[k] = v.clone()
        for snap in self.snapshots:
            for k, v in snap.items():
                if v.is_floating_point():
                    averaged[k] += v.to(torch.float32)
        n = len(self.snapshots)
        for k, v in averaged.items():
            if v.is_floating_point():
                averaged[k] = (v / n).to(torch.bfloat16)
        self.snapshots.clear()
        gc.collect()
        return averaged
