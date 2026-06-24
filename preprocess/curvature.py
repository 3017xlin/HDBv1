"""Discrete curvature from STL mesh, assigned to arbitrary query points.

Uses trimesh per-vertex discrete curvature (1-ring, radius=0) and
nearest-neighbor assignment via scipy cKDTree.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_curvature(
    stl_vertices: np.ndarray,
    stl_faces: np.ndarray,
    query_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and Gaussian curvature at query positions.

    Parameters
    ----------
    stl_vertices : (V, 3) float32
    stl_faces : (F, 3) int64/int32
    query_pos : (N, 3) float32 — volume or surface point coordinates

    Returns
    -------
    curv_mean : (N,) float32  — clipped to [-100, 100]
    curv_gauss : (N,) float32 — clipped to [-1000, 1000]
    """
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=stl_vertices,
        faces=stl_faces,
        process=False,
    )

    # Per-vertex discrete curvature (1-ring neighborhood, radius=0)
    curv_mean_vert = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, mesh.vertices, radius=0.0
    )  # (V,)
    curv_gauss_vert = trimesh.curvature.discrete_gaussian_curvature_measure(
        mesh, mesh.vertices, radius=0.0
    )  # (V,)

    # Clip extreme values
    curv_mean_vert = np.clip(curv_mean_vert, -100.0, 100.0)
    curv_gauss_vert = np.clip(curv_gauss_vert, -1000.0, 1000.0)

    # Nearest-neighbor assignment: query → closest STL vertex
    tree = cKDTree(stl_vertices)
    _, idx = tree.query(query_pos, k=1)

    curv_mean = curv_mean_vert[idx].astype(np.float32)
    curv_gauss = curv_gauss_vert[idx].astype(np.float32)

    return curv_mean, curv_gauss
