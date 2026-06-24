"""Neighbor computation: centroid-to-point top-256 and centroid-to-centroid top-64."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_top256(
    centroids: np.ndarray,
    all_pos: np.ndarray,
) -> np.ndarray:
    """For each centroid, find the 256 nearest points in all_pos.

    Parameters
    ----------
    centroids : (L, 3) float32
    all_pos : (N_total, 3) float32

    Returns
    -------
    idx : (L, 256) int32
    """
    tree = cKDTree(all_pos)
    _, idx = tree.query(centroids, k=256)
    return idx.astype(np.int32)


def compute_latent_neighbors(
    centroids: np.ndarray,
    k: int = 64,
) -> np.ndarray:
    """For each centroid, find the *k* nearest *other* centroids.

    Parameters
    ----------
    centroids : (L, 3) float32
    k : int — number of neighbors (default 64).

    Returns
    -------
    idx : (L, k) int32 — indices into *centroids* (self excluded).
    """
    tree = cKDTree(centroids)
    _, idx = tree.query(centroids, k=k + 1)  # +1 to include self
    return idx[:, 1:].astype(np.int32)  # drop self column
