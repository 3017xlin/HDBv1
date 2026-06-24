"""Per-sub-bin RoPE scale computation.

rope_scale[d] = L^(1/3) * extent[d] / geo_mean
where:
  extent[d] = p95[d] - p5[d]  across ALL train centroids (normalised coords)
  geo_mean  = (extent_x * extent_y * extent_z) ^ (1/3)
"""
from __future__ import annotations

import numpy as np


def compute_rope_scales(
    all_train_centroids_norm: np.ndarray,
    sub_bin_L_map: dict[str, int],
) -> dict[str, np.ndarray]:
    """Compute RoPE scale per axis for each sub-bin.

    Parameters
    ----------
    all_train_centroids_norm : (M, 3) float32
        Concatenated normalised centroids from ALL 700 train cases.
    sub_bin_L_map : dict[str, int]
        Mapping from sub-bin name (e.g. ``"0-19_easy"``) to L value.

    Returns
    -------
    scales : dict[str, ndarray]
        ``{sub_bin: np.array([sx, sy, sz])}``  shape ``(3,)`` per entry.
    """
    # Global extent: p5 / p95 across all train centroids
    p5 = np.percentile(all_train_centroids_norm, 5, axis=0)   # (3,)
    p95 = np.percentile(all_train_centroids_norm, 95, axis=0)  # (3,)
    extent = p95 - p5  # (3,)
    extent = np.maximum(extent, 1e-8)  # avoid zero

    geo_mean = float((extent[0] * extent[1] * extent[2]) ** (1.0 / 3.0))
    geo_mean = max(geo_mean, 1e-8)

    scales: dict[str, np.ndarray] = {}
    for sub_bin, L in sub_bin_L_map.items():
        L_cbrt = float(L) ** (1.0 / 3.0)
        scale = L_cbrt * extent / geo_mean  # (3,)
        scales[sub_bin] = scale.astype(np.float32)

    return scales
