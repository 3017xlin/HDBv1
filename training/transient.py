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
    # per_case_epoch_seed hashes f"{case_id}:{epoch}" so case_id can be
    # any str-able value (int from a legacy index-based identifier, or
    # the str case_name we use today).  No int() cast.
    case_id = pt.get("_case_id", 0)
    rng = make_rng(per_case_epoch_seed(case_id, epoch))
    L = int(pt["L"])
    pool = _np(pt["encoder_pool"])  # (L, 256, 10)
    sampled_idx = rng.integers(0, 256, size=(L, encoder_k))  # (L, encoder_k)
    t1 = pool[np.arange(L)[:, None], sampled_idx]  # (L, encoder_k, 10)
    return t1.astype(np.float32)


def build_transient2(pt: dict, epoch: int, n_query: int = 500_000) -> dict[str, np.ndarray | int]:
    """Sample up to ``n_query`` query points: Gumbel-top-k for volume,
    uniform for surface.  All output arrays are padded to length
    ``n_query`` so that DDP batches stack uniformly without per-case
    trimming.

    Layout of the returned ``[n_query, ...]`` arrays::

        positions 0 .. n_qv-1            : volume queries
        positions n_qv .. n_qv+n_qs-1    : surface queries
        positions n_qv+n_qs .. n_query-1 : padding (zero)

    ``is_surf[i] == True`` exactly when slot ``i`` is a surface query;
    ``valid_mask[i] == True`` when slot ``i`` is real (i.e. not padding).
    The decoder runs BOTH heads on all ``n_query`` slots and the training
    loss uses the masks to credit each slot to exactly one head — no
    information is dropped, no batch trim is needed.

    ``query_target_volume`` and ``query_target_surface`` are likewise
    full-length ``[n_query, *]`` arrays; valid rows are filled with the
    real targets, padding rows are zero.
    """
    case_id = pt.get("_case_id", 0)  # str or int, hashed by per_case_epoch_seed
    rng = make_rng(per_case_epoch_seed(case_id, epoch) ^ 0xA5A5_A5A5)

    vol_reorder_idx = _np(pt["vol_reorder_idx"]).astype(np.int64)
    surf_reorder_idx = _np(pt["surf_reorder_idx"]).astype(np.int64)
    N_vol = vol_reorder_idx.shape[0]
    N_surf = surf_reorder_idx.shape[0]

    # Compute vol/surf split from the passed n_query (2:1 surface ratio)
    n_qs = min(N_surf, int(n_query * N_surf / (N_surf + N_vol / 2.0)))
    n_qv = min(n_query - n_qs, N_vol)
    n_qs = min(n_qs, N_surf)
    n_used = n_qv + n_qs

    # Volume: Gumbel-top-k with precomputed log weights
    log_w = _np(pt["vol_log_sample_weight"]).astype(np.float64)
    u = rng.random(N_vol).astype(np.float64)
    gumbel = -np.log(-np.log(np.clip(u, 1e-10, 1 - 1e-10)))
    perturbed = log_w + gumbel
    vol_choice = np.argpartition(-perturbed, n_qv)[:n_qv]

    # Surface: uniform sampling (no area weighting for HDB)
    surf_choice = rng.choice(N_surf, size=n_qs, replace=False)

    # Gather global indices of the sampled points.
    vol_global = vol_reorder_idx[vol_choice]
    surf_global = surf_reorder_idx[surf_choice]

    point_pos = _np(pt["point_pos_norm"])
    point_sdf = _np(pt["point_sdf"])
    point_sdf_grad = _np(pt["point_sdf_grad"])
    point_leaf_id = _np(pt["point_leaf_id"])
    point_y_vol = _np(pt["point_y_volume"])
    point_y_surf = _np(pt["point_y_surface"])

    # Allocate the full-length [n_query, ...] arrays.  Padding rows
    # (slot n_qv+n_qs .. n_query-1) stay zero; valid_mask is False there.
    query_pos_norm = np.zeros((n_query, 3), dtype=np.float32)
    query_sdf = np.zeros((n_query,), dtype=np.float32)
    query_sdf_grad = np.zeros((n_query, 3), dtype=np.float32)
    query_leaf_id = np.zeros((n_query,), dtype=np.int32)
    query_target_volume = np.zeros(
        (n_query, point_y_vol.shape[1]), dtype=np.float32)
    query_target_surface = np.zeros(
        (n_query, point_y_surf.shape[1]), dtype=np.float32)
    is_surf = np.zeros((n_query,), dtype=bool)
    valid_mask = np.zeros((n_query,), dtype=bool)

    # Volume slots [0, n_qv)
    query_pos_norm[:n_qv] = point_pos[vol_global]
    query_sdf[:n_qv] = point_sdf[vol_global]
    query_sdf_grad[:n_qv] = point_sdf_grad[vol_global]
    query_leaf_id[:n_qv] = point_leaf_id[vol_global]
    query_target_volume[:n_qv] = point_y_vol[vol_choice]
    valid_mask[:n_qv] = True

    # Surface slots [n_qv, n_qv+n_qs)
    s0, s1 = n_qv, n_qv + n_qs
    query_pos_norm[s0:s1] = point_pos[surf_global]
    query_sdf[s0:s1] = point_sdf[surf_global]
    query_sdf_grad[s0:s1] = point_sdf_grad[surf_global]
    query_leaf_id[s0:s1] = point_leaf_id[surf_global]
    query_target_surface[s0:s1] = point_y_surf[surf_choice]
    is_surf[s0:s1] = True
    valid_mask[s0:s1] = True

    return {
        "query_pos_norm": query_pos_norm,
        "query_sdf": query_sdf,
        "query_sdf_grad": query_sdf_grad,
        "query_leaf_id": query_leaf_id,
        "query_target_volume": query_target_volume,
        "query_target_surface": query_target_surface,
        "query_is_surf": is_surf,
        "query_valid_mask": valid_mask,
        "n_query_vol": n_qv,
        "n_query_surf": n_qs,
        "n_query_used": n_used,
    }
