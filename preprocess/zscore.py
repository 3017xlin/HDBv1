"""Welford online z-score accumulation and application.

Statistics are computed ONLY from train cases (700) and applied to ALL 800
cases.  norm_stats.json contains ONE set of stats (not per-split).

39 z-scored channels total:
  Targets:   vol(5) + surf(1) = 6
  Point in:  sdf(1) + sdf_grad(3) + curv_mean(1) + curv_gauss(1) = 6
  Leaf in:   sdf(1) + sdf_grad(3) + curv_mean(1) + curv_gauss(1) + leaf_stats(21) = 27
  Total: 39

NOT z-scored: centroid_norm (div 550), PBD (0-1), point_pos_norm, point_is_surface.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


class WelfordAccumulator:
    """Numerically stable online mean / variance via Welford's algorithm.

    Supports batch updates of (N, C) arrays where C is the channel count.
    """

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.n: int = 0
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.M2 = np.zeros(n_channels, dtype=np.float64)

    def update_batch(self, values: np.ndarray) -> None:
        """Update with a batch of shape (N, C) or (N,) when C=1."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        n_new = values.shape[0]
        if n_new == 0:
            return
        mean_new = values.mean(axis=0)
        var_new = values.var(axis=0)
        n_old = self.n
        n_total = n_old + n_new
        delta = mean_new - self.mean
        self.mean = (n_old * self.mean + n_new * mean_new) / n_total
        self.M2 += n_new * var_new + delta ** 2 * n_old * n_new / n_total
        self.n = n_total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.M2 / max(self.n, 1))

    def to_dict(self) -> dict:
        """Return mean/std as plain lists for JSON serialisation."""
        m = self.mean.tolist()
        s = self.std.tolist()
        if self.n_channels == 1:
            return {"mean": m[0], "std": s[0]}
        return {"mean": m, "std": s}


# ── Accumulator names and their channel counts ─────────────────────
_ACCUMULATORS = [
    ("vol", 5),
    ("surf", 1),
    ("pt_sdf", 1),
    ("pt_sdfg", 3),
    ("pt_cm", 1),
    ("pt_cg", 1),
    ("leaf_sdf", 1),
    ("leaf_sdfg", 3),
    ("leaf_cm", 1),
    ("leaf_cg", 1),
    ("leaf_stats", 21),
]


def _build_accumulators() -> dict[str, WelfordAccumulator]:
    return {name: WelfordAccumulator(ch) for name, ch in _ACCUMULATORS}


def _feed_case(accs: dict[str, WelfordAccumulator], pt: dict) -> None:
    """Feed one intermediate-PT dict into all accumulators."""
    accs["vol"].update_batch(pt["point_y_volume"])
    accs["surf"].update_batch(pt["point_y_surface"])

    # Point-level inputs
    sdf = pt["point_sdf"]
    accs["pt_sdf"].update_batch(sdf if sdf.ndim == 2 else sdf[:, None])
    accs["pt_sdfg"].update_batch(pt["point_sdf_grad"])
    cm = pt["point_curvature_mean"]
    accs["pt_cm"].update_batch(cm if cm.ndim == 2 else cm[:, None])
    cg = pt["point_curvature_gauss"]
    accs["pt_cg"].update_batch(cg if cg.ndim == 2 else cg[:, None])

    # Leaf-level inputs
    lsdf = pt["leaf_sdf"]
    accs["leaf_sdf"].update_batch(lsdf if lsdf.ndim == 2 else lsdf[:, None])
    accs["leaf_sdfg"].update_batch(pt["leaf_sdf_grad"])
    lcm = pt["leaf_curvature_mean"]
    accs["leaf_cm"].update_batch(lcm if lcm.ndim == 2 else lcm[:, None])
    lcg = pt["leaf_curvature_gauss"]
    accs["leaf_cg"].update_batch(lcg if lcg.ndim == 2 else lcg[:, None])
    accs["leaf_stats"].update_batch(pt["leaf_stats"])


def compute_norm_stats(
    train_case_ids: list,
    load_fn: Callable,
) -> dict:
    """Compute z-score statistics from train cases only.

    Parameters
    ----------
    train_case_ids : list
        IDs / paths of the 700 train cases.
    load_fn : callable
        ``load_fn(case_id) -> dict`` returning an intermediate PT dict
        (pre-z-score) with keys like ``point_y_volume``, ``leaf_stats``, etc.

    Returns
    -------
    norm_stats : dict
        Keys like ``"vol_mean"``, ``"vol_std"``, ``"leaf_stats_mean"``, etc.
        Scalars for 1-channel features, lists for multi-channel features.
    """
    accs = _build_accumulators()

    for cid in train_case_ids:
        pt = load_fn(cid)
        _feed_case(accs, pt)

    # Flatten into a single dict
    norm_stats: dict = {}
    for name, acc in accs.items():
        d = acc.to_dict()
        norm_stats[f"{name}_mean"] = d["mean"]
        norm_stats[f"{name}_std"] = d["std"]

    return norm_stats


def _zscore(x: np.ndarray, mean, std) -> np.ndarray:
    """Element-wise z-score: (x - mean) / max(std, 1e-8)."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std = np.maximum(std, 1e-8)
    return (x - mean) / std


def apply_zscore(pt: dict, norm_stats: dict) -> dict:
    """Apply z-score normalisation IN PLACE and return the modified dict.

    Applies to:
      point_y_volume, point_y_surface,
      point_sdf, point_sdf_grad, point_curvature_mean, point_curvature_gauss,
      leaf_sdf, leaf_sdf_grad, leaf_curvature_mean, leaf_curvature_gauss,
      leaf_stats

    Does NOT touch:
      leaf_centroid_norm, leaf_pbd, point_pos_norm, point_is_surface
    """
    pt["point_y_volume"] = _zscore(
        pt["point_y_volume"], norm_stats["vol_mean"], norm_stats["vol_std"]
    )
    pt["point_y_surface"] = _zscore(
        pt["point_y_surface"], norm_stats["surf_mean"], norm_stats["surf_std"]
    )

    pt["point_sdf"] = _zscore(
        pt["point_sdf"], norm_stats["pt_sdf_mean"], norm_stats["pt_sdf_std"]
    )
    pt["point_sdf_grad"] = _zscore(
        pt["point_sdf_grad"], norm_stats["pt_sdfg_mean"], norm_stats["pt_sdfg_std"]
    )
    pt["point_curvature_mean"] = _zscore(
        pt["point_curvature_mean"], norm_stats["pt_cm_mean"], norm_stats["pt_cm_std"]
    )
    pt["point_curvature_gauss"] = _zscore(
        pt["point_curvature_gauss"], norm_stats["pt_cg_mean"], norm_stats["pt_cg_std"]
    )

    pt["leaf_sdf"] = _zscore(
        pt["leaf_sdf"], norm_stats["leaf_sdf_mean"], norm_stats["leaf_sdf_std"]
    )
    pt["leaf_sdf_grad"] = _zscore(
        pt["leaf_sdf_grad"], norm_stats["leaf_sdfg_mean"], norm_stats["leaf_sdfg_std"]
    )
    pt["leaf_curvature_mean"] = _zscore(
        pt["leaf_curvature_mean"], norm_stats["leaf_cm_mean"], norm_stats["leaf_cm_std"]
    )
    pt["leaf_curvature_gauss"] = _zscore(
        pt["leaf_curvature_gauss"], norm_stats["leaf_cg_mean"], norm_stats["leaf_cg_std"]
    )
    pt["leaf_stats"] = _zscore(
        pt["leaf_stats"], norm_stats["leaf_stats_mean"], norm_stats["leaf_stats_std"]
    )

    return pt


def save_norm_stats(norm_stats: dict, path: str | Path) -> None:
    """Serialise norm_stats to JSON, converting numpy arrays to lists."""

    def _convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return v

    out = {k: _convert(v) for k, v in norm_stats.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_norm_stats(path: str | Path) -> dict:
    """Load norm_stats from JSON."""
    with open(path, "r") as f:
        return json.load(f)
