"""Memory reporting helpers."""
import psutil
import torch


def cpu_rss_gib() -> float:
    return psutil.Process().memory_info().rss / 1024**3


def gpu_peak_gib(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def gpu_alloc_gib(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024**3
