"""Panoramic Building Distance (PBD) encoding via Open3D raycasting.

16 horizontal directions (22.5 deg apart).  For each centroid, shoot rays
and encode hit distance as  pbd_k = 1 - exp(-d_k / scale).
"""
from __future__ import annotations

import numpy as np


def compute_pbd(
    centroids: np.ndarray,
    stl_vertices: np.ndarray,
    stl_faces: np.ndarray,
    n_bins: int = 16,
    scale: float = 50.0,
    batch_size: int = 1000,
) -> np.ndarray:
    """Compute PBD feature for each centroid.

    Parameters
    ----------
    centroids : (L, 3) float32 — centroid positions (physical coords).
    stl_vertices : (V, 3) float32
    stl_faces : (F, 3) int32/int64
    n_bins : int — number of horizontal ray directions (default 16).
    scale : float — PBD scale parameter (default 50.0 m).
    batch_size : int — centroids per raycasting batch (default 1000).

    Returns
    -------
    pbd : (L, n_bins) float32 — values in [0, 1].
    """
    import open3d as o3d

    L = centroids.shape[0]

    # Build Open3D raycasting scene
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(stl_vertices.astype(np.float32))
    mesh.triangle.indices = o3d.core.Tensor(stl_faces.astype(np.int32))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    # 16 horizontal directions
    angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    directions = np.stack(
        [np.cos(angles), np.sin(angles), np.zeros(n_bins)], axis=1
    ).astype(np.float32)  # (n_bins, 3)

    pbd = np.ones((L, n_bins), dtype=np.float32)

    for start in range(0, L, batch_size):
        end = min(start + batch_size, L)
        n_batch = end - start
        origins = centroids[start:end]  # (n_batch, 3)

        # Expand: each origin paired with each direction
        origins_exp = np.repeat(origins, n_bins, axis=0)  # (n_batch*n_bins, 3)
        dirs_exp = np.tile(directions, (n_batch, 1))  # (n_batch*n_bins, 3)

        rays = np.hstack([origins_exp, dirs_exp]).astype(np.float32)
        rays_tensor = o3d.core.Tensor(rays)

        result = scene.cast_rays(rays_tensor)
        t_hit = result["t_hit"].numpy().reshape(n_batch, n_bins)
        pbd[start:end] = 1.0 - np.exp(-t_hit / scale)

    return pbd
