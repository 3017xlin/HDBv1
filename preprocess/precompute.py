"""Precompute 6 fields to add to each cache PT.

Fields:
  1. point_leaf_id      int32  (N_total,)
  2. encoder_pool       bfloat16 (L, 256, 10)
  3. vol_log_sample_weight  float32 (N_vol,)
  4. rope_cos / rope_sin    float32 (L+16, 32)
  5. bigbird_fixed      int32  (L, 80)
  6. n_query_vol / n_query_surf  int scalars
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

# Add project root so quadri imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from quadri.models.rope import precompute_rope_3d


# ── Volume sample weight function ──────────────────────────────────
def volume_sample_weight(sdf: np.ndarray) -> np.ndarray:
    """Piecewise SDF importance weight (physical units, pre-z-score).

    Parameters
    ----------
    sdf : (N,) float32 — corrected (negated) SDF, positive = outside building.

    Returns
    -------
    w : (N,) float32
    """
    w = np.where(
        sdf < -3.0,
        0.05,
        np.where(
            sdf < 0.0,
            6.0,
            np.where(
                sdf < 1.5,
                6.0 + 2.0 * (sdf / 1.5),
                np.where(
                    sdf < 2.5,
                    8.0,
                    np.where(
                        sdf < 5.0,
                        8.0 - 2.0 * (sdf - 2.5) / 2.5,
                        np.where(
                            sdf < 50.0,
                            6.0 - 3.0 * (sdf - 5.0) / 45.0,
                            np.where(
                                sdf < 80.0,
                                3.0 - 2.5 * (sdf - 50.0) / 30.0,
                                np.maximum(0.5 - 0.3 * (sdf - 80.0) / 300.0, 0.1),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return w.astype(np.float32)


# ── Individual precompute functions ────────────────────────────────


def compute_point_leaf_id(
    centroids: np.ndarray,
    all_pos: np.ndarray,
) -> np.ndarray:
    """Assign each point to its nearest centroid.

    Returns
    -------
    point_leaf_id : (N_total,) int32
    """
    tree = cKDTree(centroids)
    _, ids = tree.query(all_pos, k=1)
    return ids.astype(np.int32)


def compute_encoder_pool(
    centroids_norm: np.ndarray,
    all_pos_norm: np.ndarray,
    all_sdf_zscored: np.ndarray,
    all_sdf_grad_zscored: np.ndarray,
    all_is_surface: np.ndarray,
    all_curv_mean_zscored: np.ndarray,
    all_curv_gauss_zscored: np.ndarray,
    top256_idx: np.ndarray,
) -> np.ndarray:
    """Precompute encoder pool: 10-dim features for each centroid's 256 neighbours.

    10 dims: [rel_pos(3), sdf(1), sdf_grad(3), surf_flag(1), curv_mean(1), curv_gauss(1)]

    rel_pos = point_pos_norm[top256] - centroid_norm[:, None, :]
    sdf, sdf_grad, curv values are already z-scored.
    surf_flag is 0/1 (not z-scored).

    Returns
    -------
    pool : (L, 256, 10) stored as bfloat16 (via torch conversion).
    """
    L = centroids_norm.shape[0]

    # Gather neighbour positions and compute relative coords
    nbr_pos = all_pos_norm[top256_idx]  # (L, 256, 3)
    rel_pos = nbr_pos - centroids_norm[:, None, :]  # (L, 256, 3)

    # Gather other features
    nbr_sdf = all_sdf_zscored[top256_idx]  # (L, 256)
    nbr_sdfg = all_sdf_grad_zscored[top256_idx]  # (L, 256, 3)
    nbr_surf = all_is_surface[top256_idx].astype(np.float32)  # (L, 256)
    nbr_cm = all_curv_mean_zscored[top256_idx]  # (L, 256)
    nbr_cg = all_curv_gauss_zscored[top256_idx]  # (L, 256)

    # Stack into (L, 256, 10)
    pool = np.concatenate(
        [
            rel_pos,  # (L, 256, 3)
            nbr_sdf[:, :, None],  # (L, 256, 1)
            nbr_sdfg,  # (L, 256, 3)
            nbr_surf[:, :, None],  # (L, 256, 1)
            nbr_cm[:, :, None],  # (L, 256, 1)
            nbr_cg[:, :, None],  # (L, 256, 1)
        ],
        axis=2,
    ).astype(np.float32)

    # Convert to bfloat16 via torch
    pool_t = torch.from_numpy(pool).to(torch.bfloat16)
    return pool_t


def compute_vol_log_sample_weight(
    original_sdf_volume: np.ndarray,
) -> np.ndarray:
    """Compute log(volume_sample_weight(sdf)) using ORIGINAL (pre-z-score) SDF.

    Parameters
    ----------
    original_sdf_volume : (N_vol,) float32 — corrected but pre-z-score SDF for volume points.

    Returns
    -------
    log_w : (N_vol,) float32
    """
    w = volume_sample_weight(original_sdf_volume)
    return np.log(np.maximum(w, 1e-10)).astype(np.float32)


def compute_rope(
    centroids_norm: np.ndarray,
    rope_scale: np.ndarray,
    head_dim: int = 64,
    base: float = 100.0,
    rope_dims: tuple[int, int, int] = (22, 22, 20),
    register_tokens: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE cos/sin for a single case.

    Parameters
    ----------
    centroids_norm : (L, 3) float32 — normalised centroid positions.
    rope_scale : (3,) float32 — per-axis RoPE scale for this case's sub-bin.

    Returns
    -------
    rope_cos : (L+16, 32) float32
    rope_sin : (L+16, 32) float32
    """
    leaf_t = torch.from_numpy(centroids_norm).float()
    sx, sy, sz = float(rope_scale[0]), float(rope_scale[1]), float(rope_scale[2])
    cos_t, sin_t = precompute_rope_3d(
        leaf_centroid_norm=leaf_t,
        head_dim=head_dim,
        base=base,
        rope_dims=rope_dims,
        scale=(sx, sy, sz),
        register_tokens=register_tokens,
    )
    return cos_t.numpy().astype(np.float32), sin_t.numpy().astype(np.float32)


def compute_bigbird_fixed(
    latent_neighbor_top64: np.ndarray,
    L: int,
    register_tokens: int = 16,
) -> np.ndarray:
    """Precompute BigBird fixed indices: 64 local + 16 register.

    Parameters
    ----------
    latent_neighbor_top64 : (L, 64) int32
    L : int
    register_tokens : int

    Returns
    -------
    bigbird_fixed : (L, 80) int32
    """
    reg_indices = np.arange(L, L + register_tokens, dtype=np.int32)
    reg_block = np.tile(reg_indices, (L, 1))  # (L, 16)
    return np.concatenate([latent_neighbor_top64, reg_block], axis=1).astype(np.int32)


def compute_n_query(
    n_vol: int,
    n_surf: int,
    n_query_total: int = 500_000,
) -> tuple[int, int]:
    """Compute per-case volume and surface query counts (2:1 surface ratio).

    Returns
    -------
    n_query_vol, n_query_surf : int, int
    """
    # Target: surface:volume = 2:1 per-point probability
    n_query_surf = min(n_surf, int(n_query_total * n_surf / (n_surf + n_vol / 2.0)))
    n_query_vol = n_query_total - n_query_surf
    return n_query_vol, n_query_surf


def add_precomputed_fields(
    pt: dict,
    rope_scale: np.ndarray,
    n_query_total: int = 500_000,
    head_dim: int = 64,
    rope_base: float = 100.0,
    rope_dims: tuple[int, int, int] = (22, 22, 20),
    register_tokens: int = 16,
) -> dict:
    """Add all 6 precomputed fields to a (post-z-score) PT dict, in place.

    Expects the PT to already contain z-scored features plus the
    ``original_sdf_volume`` key (pre-z-score SDF for volume points).

    Parameters
    ----------
    pt : dict — the cache PT being built.
    rope_scale : (3,) — RoPE scale for this case's sub-bin.

    Returns the modified *pt*.
    """
    centroids_norm = pt["leaf_centroid_norm"]  # (L, 3) float32
    all_pos_norm = pt["point_pos_norm"]  # (N_total, 3)
    L = centroids_norm.shape[0]

    # 1. point_leaf_id
    pt["point_leaf_id"] = compute_point_leaf_id(centroids_norm, all_pos_norm)

    # 2. encoder_pool (uses z-scored features)
    pt["encoder_pool"] = compute_encoder_pool(
        centroids_norm=centroids_norm,
        all_pos_norm=all_pos_norm,
        all_sdf_zscored=pt["point_sdf"],
        all_sdf_grad_zscored=pt["point_sdf_grad"],
        all_is_surface=pt["point_is_surface"],
        all_curv_mean_zscored=pt["point_curvature_mean"],
        all_curv_gauss_zscored=pt["point_curvature_gauss"],
        top256_idx=pt["latent_point_top256"],
    )

    # 3. vol_log_sample_weight (uses ORIGINAL pre-z-score SDF)
    pt["vol_log_sample_weight"] = compute_vol_log_sample_weight(
        pt["original_sdf_volume"]
    )
    # Remove the temporary original SDF — not needed in the final cache
    del pt["original_sdf_volume"]

    # 4. RoPE cos/sin
    rope_cos, rope_sin = compute_rope(
        centroids_norm=centroids_norm,
        rope_scale=rope_scale,
        head_dim=head_dim,
        base=rope_base,
        rope_dims=rope_dims,
        register_tokens=register_tokens,
    )
    pt["rope_cos"] = rope_cos
    pt["rope_sin"] = rope_sin

    # 5. bigbird_fixed
    pt["bigbird_fixed"] = compute_bigbird_fixed(
        pt["latent_neighbor_top64"], L, register_tokens
    )

    # 6. n_query_vol / n_query_surf
    n_vol = pt["vol_reorder_idx"].shape[0]
    n_surf = pt["surf_reorder_idx"].shape[0]
    nqv, nqs = compute_n_query(n_vol, n_surf, n_query_total)
    pt["n_query_vol"] = nqv
    pt["n_query_surf"] = nqs

    return pt
