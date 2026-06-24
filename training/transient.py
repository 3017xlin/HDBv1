"""Per-step CPU transient computation using precomputed cache fields.

Pure numpy — runs in background threads, returns numpy arrays for H2D.
Determinism: per-(case_id, epoch) RNG for reproducibility.
"""
from __future__ import annotations

import numpy as np
import torch

from utils.seed import make_rng, per_case_epoch_seed


def _np(x):
    """Convert a torch.Tensor (possibly bf16) or ndarray to float32 numpy."""
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            return x.to(torch.float32).numpy()
        return x.numpy()
    return x


def build_transient1(pt: dict, epoch: int, encoder_k: int = 32) -> np.ndarray:
    """Sample encoder_k points per leaf from precomputed encoder_pool.

    Returns (L, encoder_k, 10) float32.
    """
    case_id = int(pt.get("_case_id", 0))
    rng = make_rng(per_case_epoch_seed(case_id, epoch))
    L = int(pt["L"])
    pool = _np(pt["encoder_pool"])  # (L, 256, 10)
    sampled_idx = rng.integers(0, 256, size=(L, encoder_k))  # (L, encoder_k)
    t1 = pool[np.arange(L)[:, None], sampled_idx]  # (L, encoder_k, 10)
    return t1.astype(np.float32)


def build_transient2(pt: dict, epoch: int, n_query: int = 500_000) -> dict[str, np.ndarray | int]:
    """Sample n_query query points: Gumbel-top-k for volume, uniform for surface.

    Uses precomputed vol_log_sample_weight and point_leaf_id.
    Returns dict with query arrays as numpy.
    """
    case_id = int(pt.get("_case_id", 0))
    rng = make_rng(per_case_epoch_seed(case_id, epoch) ^ 0xA5A5_A5A5)

    vol_reorder_idx = _np(pt["vol_reorder_idx"]).astype(np.int64)
    surf_reorder_idx = _np(pt["surf_reorder_idx"]).astype(np.int64)
    N_vol = vol_reorder_idx.shape[0]
    N_surf = surf_reorder_idx.shape[0]

    # Compute vol/surf split from the passed n_query (2:1 surface ratio)
    n_qs = min(N_surf, int(n_query * N_surf / (N_surf + N_vol / 2.0)))
    n_qv = min(n_query - n_qs, N_vol)
    n_qs = min(n_qs, N_surf)

    # Volume: Gumbel-top-k with precomputed log weights
    log_w = _np(pt["vol_log_sample_weight"]).astype(np.float64)
    u = rng.random(N_vol).astype(np.float64)
    gumbel = -np.log(-np.log(np.clip(u, 1e-10, 1 - 1e-10)))
    perturbed = log_w + gumbel
    vol_choice = np.argpartition(-perturbed, n_qv)[:n_qv]

    # Surface: uniform sampling (no area weighting for HDB)
    surf_choice = rng.choice(N_surf, size=n_qs, replace=False)

    # Gather global indices and concatenate (volume first, then surface)
    vol_global = vol_reorder_idx[vol_choice]
    surf_global = surf_reorder_idx[surf_choice]
    query_idx = np.concatenate([vol_global, surf_global])

    point_pos = _np(pt["point_pos_norm"])
    point_sdf = _np(pt["point_sdf"])
    point_sdf_grad = _np(pt["point_sdf_grad"])
    point_leaf_id = _np(pt["point_leaf_id"])
    point_y_vol = _np(pt["point_y_volume"])
    point_y_surf = _np(pt["point_y_surface"])

    return {
        "query_pos_norm": point_pos[query_idx].astype(np.float32),
        "query_sdf": point_sdf[query_idx].astype(np.float32),
        "query_sdf_grad": point_sdf_grad[query_idx].astype(np.float32),
        "query_leaf_id": point_leaf_id[query_idx].astype(np.int32),
        "query_target_volume": point_y_vol[vol_choice].astype(np.float32),
        "query_target_surface": point_y_surf[surf_choice].astype(np.float32),
        "n_query_vol": n_qv,
    }
