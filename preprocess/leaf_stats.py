"""21-dim leaf_stats via fully vectorized top-256 neighbor computation.

Layout:
    [0:6]   cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz
    [6:9]   dist_mean, dist_std, dist_skew
    [9]     density = log1p(256 / (4/3 pi dist_mean^3))
    [10]    com_dist = ||centroid - bbox_center||  (bbox_center = (min+max)/2)
    [11:14] sdf_min, sdf_max, sdf_range
    [14:17] mean_dir (x, y, z)
    [17]    angular_span = 1 - ||mean_dir||
    [18]    curv_mean_avg
    [19]    curv_gauss_avg
    [20]    surf_ratio

NO Python for-loop over L.  Covariance uses rel directly (no double centering).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def compute_leaf_stats_vectorized(
    centroids: np.ndarray,
    all_pos: np.ndarray,
    all_sdf: np.ndarray,
    all_curv_mean: np.ndarray,
    all_curv_gauss: np.ndarray,
    all_is_surface: np.ndarray,
    top256_idx: np.ndarray,
) -> np.ndarray:
    """Compute 21-dim leaf statistics for each centroid, fully vectorized.

    Parameters
    ----------
    centroids : (L, 3) float32
    all_pos : (N_total, 3) float32
    all_sdf : (N_total,) float32
    all_curv_mean : (N_total,) float32
    all_curv_gauss : (N_total,) float32
    all_is_surface : (N_total,) bool
    top256_idx : (L, 256) int32

    Returns
    -------
    stats : (L, 21) float32
    """
    L = centroids.shape[0]
    K = 256

    # Gather neighbor data  — (L, 256, ...)
    nbr_pos = all_pos[top256_idx]  # (L, 256, 3)
    nbr_sdf = all_sdf[top256_idx]  # (L, 256)
    nbr_cm = all_curv_mean[top256_idx]  # (L, 256)
    nbr_cg = all_curv_gauss[top256_idx]  # (L, 256)
    nbr_surf = all_is_surface[top256_idx]  # (L, 256)

    rel = nbr_pos - centroids[:, None, :]  # (L, 256, 3)

    stats = np.zeros((L, 21), dtype=np.float32)

    # ── [0:6] Covariance (NO double centering — use rel directly) ──
    # cov_ij = mean(rel_i * rel_j)
    # Einstein: 'lik,lij->lkj' but we only need 6 unique entries
    rx, ry, rz = rel[:, :, 0], rel[:, :, 1], rel[:, :, 2]
    stats[:, 0] = np.mean(rx * rx, axis=1)  # xx
    stats[:, 1] = np.mean(ry * ry, axis=1)  # yy
    stats[:, 2] = np.mean(rz * rz, axis=1)  # zz
    stats[:, 3] = np.mean(rx * ry, axis=1)  # xy
    stats[:, 4] = np.mean(rx * rz, axis=1)  # xz
    stats[:, 5] = np.mean(ry * rz, axis=1)  # yz

    # ── [6:9] Distance statistics ──
    dists = np.linalg.norm(rel, axis=2)  # (L, 256)
    dm = dists.mean(axis=1)  # (L,)
    ds = dists.std(axis=1)  # (L,)
    z_dists = (dists - dm[:, None]) / np.maximum(ds[:, None], 1e-8)
    dskew = (z_dists ** 3).mean(axis=1)

    stats[:, 6] = dm
    stats[:, 7] = ds
    stats[:, 8] = dskew

    # ── [9] Density ──
    stats[:, 9] = np.log1p(
        float(K) / ((4.0 / 3.0) * np.pi * np.maximum(dm, 1e-8) ** 3)
    )

    # ── [10] COM distance (bbox center) ──
    pos_min = nbr_pos.min(axis=1)  # (L, 3)
    pos_max = nbr_pos.max(axis=1)  # (L, 3)
    bbox_center = 0.5 * (pos_min + pos_max)  # (L, 3)
    stats[:, 10] = np.linalg.norm(centroids - bbox_center, axis=1)

    # ── [11:14] SDF min / max / range ──
    stats[:, 11] = nbr_sdf.min(axis=1)
    stats[:, 12] = nbr_sdf.max(axis=1)
    stats[:, 13] = stats[:, 12] - stats[:, 11]

    # ── [14:17] Mean direction (unit vectors averaged) ──
    norms = np.maximum(dists[:, :, None], 1e-8)  # (L, 256, 1)
    unit_vecs = rel / norms  # (L, 256, 3)
    mean_dir = unit_vecs.mean(axis=1)  # (L, 3)
    stats[:, 14:17] = mean_dir

    # ── [17] Angular span ──
    stats[:, 17] = 1.0 - np.linalg.norm(mean_dir, axis=1)

    # ── [18:20] Curvature averages ──
    stats[:, 18] = nbr_cm.mean(axis=1)
    stats[:, 19] = nbr_cg.mean(axis=1)

    # ── [20] Surface ratio ──
    stats[:, 20] = nbr_surf.astype(np.float32).sum(axis=1) / float(K)

    return stats
