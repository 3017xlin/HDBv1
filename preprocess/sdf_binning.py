"""SDF-based spatial binning and weighted K-means centroid allocation.

15 bins with edges derived from cumulative sums:
  bin 0:  [-inf,    6.25m)
  bin 1:  [6.25m,  15.625m)
  ...
  bin 14: [371.875m, 1100m]

Bin edges: 3.125 * cumsum([2,3,4,...,15]) gives 14 inner boundaries.
Prepend -inf, append 1100 => 16 edges defining 15 bins.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def build_sdf_bin_edges() -> np.ndarray:
    """Return 16 bin edges defining 15 SDF bins.

    Returns
    -------
    edges : (16,) float64
        ``edges[0] = -inf``, ``edges[-1] = 1100.0``.
        ``np.digitize(sdf, edges[1:])`` yields 0-indexed bin IDs.
    """
    cumwidths = np.cumsum(np.arange(2, 16))  # 14 values: [2,5,9,14,20,27,...]
    inner_edges = 3.125 * cumwidths  # 14 inner boundaries
    edges = np.concatenate(
        [
            [float("-inf")],
            inner_edges,
            [1100.0],
        ]
    )
    return edges  # shape (16,)


def assign_sdf_bins(sdf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each SDF value to a bin (0-indexed).

    Parameters
    ----------
    sdf : (N,) — SDF values (already corrected / negated).
    edges : (16,) — from :func:`build_sdf_bin_edges`.

    Returns
    -------
    bin_ids : (N,) int32, values in [0, 14].
    """
    return np.digitize(sdf, edges[1:]).astype(np.int32)


def weighted_kmeans_allocation(
    all_pos: np.ndarray,
    all_sdf: np.ndarray,
    bin_edges: np.ndarray,
    L: int,
    case_id: int = 0,
    alpha: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """Distribute *L* K-means centroids across SDF bins with power-law weighting.

    Parameters
    ----------
    all_pos : (N_total, 3) — all volume + surface point coordinates.
    all_sdf : (N_total,)   — corresponding SDF values (corrected).
    bin_edges : (16,)       — from :func:`build_sdf_bin_edges`.
    L : int                 — total number of latent grid centroids.
    case_id : int           — for random_state diversity.
    alpha : float           — power-law decay exponent (default 0.65).

    Returns
    -------
    centroids : (L, 3) float32
    bin_id : (L,) int8 — bin index of each centroid.
    """
    n_bins = len(bin_edges) - 1  # 15
    bin_ids = np.digitize(all_sdf, bin_edges[1:])  # (N_total,) in [0, n_bins-1]

    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.float64)
    weights = np.array([1.0 / (1 + i) ** alpha for i in range(n_bins)])

    weighted_counts = weights * counts
    total_wc = weighted_counts.sum()
    if total_wc == 0:
        raise ValueError("No points found in any SDF bin — check SDF data.")

    alloc = (L * weighted_counts / total_wc).astype(np.int64)

    # Fix rounding to ensure sum == L
    diff = L - alloc.sum()
    for _ in range(abs(int(diff))):
        if diff > 0:
            candidates = np.where(counts > 0)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.argmax(weighted_counts[candidates])]
            alloc[idx] += 1
            diff -= 1
        else:
            candidates = np.where(alloc > 1)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.argmin(weighted_counts[candidates])]
            alloc[idx] -= 1
            diff += 1

    all_centroids: list[np.ndarray] = []
    all_bin_ids: list[np.ndarray] = []

    for b in range(n_bins):
        n_clusters = int(alloc[b])
        if n_clusters == 0:
            continue
        mask = bin_ids == b
        n_pts = int(mask.sum())
        if n_pts == 0:
            continue

        pts = all_pos[mask]

        if n_pts < n_clusters:
            # Fewer points than requested clusters — duplicate
            centroids_b = pts.copy()
            if n_clusters > n_pts:
                rng = np.random.RandomState(42 + case_id * 1000 + b)
                extra_idx = rng.choice(n_pts, size=n_clusters - n_pts, replace=True)
                centroids_b = np.vstack([centroids_b, pts[extra_idx]])
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=max(10000, n_clusters * 20),
                max_iter=100,
                random_state=42 + case_id * 1000 + b,
                n_init=3,
            )
            kmeans.fit(pts)
            centroids_b = kmeans.cluster_centers_

        all_centroids.append(centroids_b.astype(np.float32))
        all_bin_ids.append(np.full(n_clusters, b, dtype=np.int8))

    centroids = np.vstack(all_centroids).astype(np.float32)
    bin_id = np.concatenate(all_bin_ids)

    assert centroids.shape[0] == L, (
        f"Centroid count {centroids.shape[0]} != L={L}"
    )
    return centroids, bin_id
