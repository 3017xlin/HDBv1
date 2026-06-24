#!/usr/bin/env python
"""HDB preprocessing pipeline — build the complete cache of 800 z-scored PTs.

Usage::

    python hdb/preprocess/build_cache.py --config hdb/config.yaml

Flow:
  1. Load config + manifest.json
  2. Phase 0: Anomaly detection on all cases -> filter
  3. Phase 1-4: Per-case processing (multiprocessing, ~40 workers)
       a. Load physical PT
       b. SDF fix (negate volume_sdf, volume_sdf_grad)
       c. Coordinate normalisation (pos / 550)
       d. Curvature computation from STL
       e. Training transform: p_train = p_phys + 9.81 * z
       f. SDF binning + K-means -> centroids
       g. Neighbors: top256, top64
       h. leaf_stats (21 dims, vectorized)
       i. PBD computation
       j. Save intermediate PT (pre-zscore)
  4. Welford pass: compute norm_stats from 700 train cases
  5. Z-score pass: apply z-score to all 800 cases
  6. Precompute pass: add 6 precomputed fields
  7. RoPE scale computation
  8. Save final cache PTs + norm_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Ensure project root on sys.path so `from preprocess.X import Y`
# resolves regardless of the project directory name.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocess.anomaly_detect import compute_case_stats, detect_anomalies_mad
from preprocess.curvature import compute_curvature
from preprocess.leaf_stats import compute_leaf_stats_vectorized
from preprocess.neighbors import compute_latent_neighbors, compute_top256
from preprocess.pbd import compute_pbd
from preprocess.precompute import add_precomputed_fields
from preprocess.rope_scale import compute_rope_scales
from preprocess.sdf_binning import build_sdf_bin_edges, weighted_kmeans_allocation
from preprocess.zscore import (
    apply_zscore,
    compute_norm_stats,
    load_norm_stats,
    save_norm_stats,
)

# ── Physical constants ─────────────────────────────────────────────
G_GRAVITY = 9.81  # Pa/m  (kinematic pressure detrend coefficient)
COORD_DIVISOR = 550.0


# ── Config helpers ─────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_manifest(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ── Phase 0: Anomaly detection ────────────────────────────────────


def _compute_stats_for_anomaly(case_path: str) -> dict:
    """Load a physical PT and compute the 19 anomaly-detection indicators."""
    pt = torch.load(case_path, map_location="cpu", weights_only=False)
    # Convert tensors to numpy for compute_case_stats
    pt_np: dict[str, Any] = {}
    for k, v in pt.items():
        if isinstance(v, torch.Tensor):
            pt_np[k] = v.numpy()
        else:
            pt_np[k] = v
    return compute_case_stats(pt_np)


def run_anomaly_detection(
    all_case_paths: list[str],
    num_workers: int = 40,
) -> dict[str, dict]:
    """Scan all physical PTs and return per-case anomaly results."""
    print(f"[Phase 0] Anomaly detection: scanning {len(all_case_paths)} cases ...")
    with Pool(num_workers) as pool:
        cases_stats = list(
            tqdm(
                pool.imap(_compute_stats_for_anomaly, all_case_paths),
                total=len(all_case_paths),
                desc="Anomaly stats",
            )
        )
    results = detect_anomalies_mad(cases_stats)
    n_anomaly = sum(1 for v in results.values() if v["is_anomaly"])
    print(f"[Phase 0] Detected {n_anomaly} anomalous cases out of {len(results)}")
    return results


# ── Phase 1-4: Per-case processing ────────────────────────────────


def process_single_case(
    case_info: dict,
    raw_dir: str,
    intermediate_dir: str,
    bin_edges: np.ndarray,
    coord_divisor: float,
    alpha: float = 0.65,
) -> str:
    """Process one case: load physical PT, compute all features, save intermediate.

    Parameters
    ----------
    case_info : dict with keys "case_name", "sub_bin", "L", "case_id", "pt_path".
    raw_dir : root directory for raw PTs.
    intermediate_dir : where to save intermediate PTs.
    bin_edges : SDF bin edges from build_sdf_bin_edges().

    Returns
    -------
    case_name : str — echoed back for progress tracking.
    """
    case_name = case_info["case_name"]
    sub_bin = case_info["sub_bin"]
    L = case_info["L"]
    case_id = case_info["case_id"]
    pt_path = case_info["pt_path"]

    # ── Load physical PT ──
    pt_raw = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Convert all tensors to numpy
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return x

    # ── Extract raw fields ──
    volume_fields = _to_np(pt_raw["volume_fields"]).astype(np.float32)  # (N_vol, 5)
    surface_fields = _to_np(pt_raw["surface_fields"]).astype(np.float32)  # (N_surf, 4)
    volume_pos = _to_np(pt_raw["volume_pos"]).astype(np.float32)  # (N_vol, 3)
    surface_pos = _to_np(pt_raw["surface_pos"]).astype(np.float32)  # (N_surf, 3)
    volume_sdf = _to_np(pt_raw["volume_sdf"]).astype(np.float32)  # (N_vol,)
    volume_sdf_grad = _to_np(pt_raw["volume_sdf_grad"]).astype(np.float32)  # (N_vol, 3)
    stl_vertices = _to_np(pt_raw["stl_vertices"]).astype(np.float32)
    stl_faces = _to_np(pt_raw["stl_faces"]).astype(np.int64)

    N_vol = volume_pos.shape[0]
    N_surf = surface_pos.shape[0]

    # ── (b) SDF fix: negate ──
    volume_sdf = -volume_sdf
    volume_sdf_grad = -volume_sdf_grad

    # Keep original SDF for vol_log_sample_weight (pre-z-score)
    original_sdf_volume = volume_sdf.copy()

    # ── Surface SDF: physically on building wall → SDF ≈ 0 ──
    # Surface points sit on the building surface; their true SDF is 0.
    # Grad at surface points: use nearest volume point's gradient as proxy.
    from scipy.spatial import cKDTree as _cKDTree

    vol_tree = _cKDTree(volume_pos)
    _, surf_nn_idx = vol_tree.query(surface_pos, k=1)
    surface_sdf = np.zeros(N_surf, dtype=np.float32)
    surface_sdf_grad = volume_sdf_grad[surf_nn_idx]

    # ── Concatenate volume + surface ──
    all_pos = np.concatenate([volume_pos, surface_pos], axis=0)  # (N_total, 3)
    all_sdf = np.concatenate([volume_sdf, surface_sdf], axis=0)
    all_sdf_grad = np.concatenate([volume_sdf_grad, surface_sdf_grad], axis=0)
    is_surface = np.concatenate(
        [np.zeros(N_vol, dtype=bool), np.ones(N_surf, dtype=bool)]
    )
    N_total = all_pos.shape[0]

    # ── (c) Coordinate normalisation ──
    all_pos_norm = all_pos / coord_divisor
    stl_vertices_norm = stl_vertices / coord_divisor  # for PBD if needed

    # ── (d) Curvature computation from STL ──
    all_curv_mean, all_curv_gauss = compute_curvature(stl_vertices, stl_faces, all_pos)

    # ── (e) Training transform: p_train = p_phys + 9.81 * z ──
    # Volume: channel 3 is p
    z_vol = volume_pos[:, 2]
    volume_fields[:, 3] = volume_fields[:, 3] + G_GRAVITY * z_vol

    # Surface: only use p (channel 0) — HDB surface has 1 output channel
    z_surf = surface_pos[:, 2]
    surface_p_detrended = surface_fields[:, 0] + G_GRAVITY * z_surf

    # ── (f) SDF binning + K-means ──
    centroids, bin_id = weighted_kmeans_allocation(
        all_pos=all_pos,
        all_sdf=all_sdf,
        bin_edges=bin_edges,
        L=L,
        case_id=case_id,
        alpha=alpha,
    )
    centroids_norm = centroids / coord_divisor

    # ── (g) Neighbors ──
    top256_idx = compute_top256(centroids, all_pos)  # (L, 256) int32
    top64_idx = compute_latent_neighbors(centroids, k=64)  # (L, 64) int32

    # ── (h) leaf_stats (21 dims) ──
    leaf_stats = compute_leaf_stats_vectorized(
        centroids=centroids,
        all_pos=all_pos,
        all_sdf=all_sdf,
        all_curv_mean=all_curv_mean,
        all_curv_gauss=all_curv_gauss,
        all_is_surface=is_surface,
        top256_idx=top256_idx,
    )

    # ── Leaf-level SDF / SDF_grad / curvature (at centroid locations) ──
    cent_tree = _cKDTree(all_pos)
    _, cent_nn = cent_tree.query(centroids, k=1)
    leaf_sdf = all_sdf[cent_nn].astype(np.float32)
    leaf_sdf_grad = all_sdf_grad[cent_nn].astype(np.float32)
    leaf_curv_mean = all_curv_mean[cent_nn].astype(np.float32)
    leaf_curv_gauss = all_curv_gauss[cent_nn].astype(np.float32)

    # ── (i) PBD computation ──
    pbd = compute_pbd(
        centroids=centroids,
        stl_vertices=stl_vertices,
        stl_faces=stl_faces.astype(np.int32),
        n_bins=16,
        scale=50.0,
    )

    # ── Surface areas ──
    surface_areas = _to_np(pt_raw.get("surface_areas", np.ones(N_surf, dtype=np.float32)))
    if isinstance(surface_areas, torch.Tensor):
        surface_areas = surface_areas.numpy()
    surface_areas = surface_areas.astype(np.float32)

    # ── Reorder indices ──
    vol_reorder_idx = np.arange(N_vol, dtype=np.int64)
    surf_reorder_idx = np.arange(N_vol, N_vol + N_surf, dtype=np.int64)

    # ── Build intermediate PT dict ──
    pt_out: dict[str, Any] = {
        # Leaf / centroid data
        "leaf_centroid_norm": centroids_norm.astype(np.float32),
        "leaf_stats": leaf_stats.astype(np.float32),
        "leaf_sdf": leaf_sdf,
        "leaf_sdf_grad": leaf_sdf_grad,
        "leaf_curvature_mean": leaf_curv_mean,
        "leaf_curvature_gauss": leaf_curv_gauss,
        "leaf_pbd": pbd.astype(np.float32),
        "leaf_bin_id": bin_id,
        # Neighbor indices
        "latent_point_top256": top256_idx,
        "latent_neighbor_top64": top64_idx,
        # Point data
        "point_pos_norm": all_pos_norm.astype(np.float32),
        "point_sdf": all_sdf.astype(np.float32),
        "point_sdf_grad": all_sdf_grad.astype(np.float32),
        "point_curvature_mean": all_curv_mean.astype(np.float32),
        "point_curvature_gauss": all_curv_gauss.astype(np.float32),
        "point_is_surface": is_surface,
        # Training targets
        "point_y_volume": volume_fields.astype(np.float32),  # (N_vol, 5)
        "point_y_surface": surface_p_detrended[:, None].astype(np.float32),  # (N_surf, 1)
        "vol_reorder_idx": vol_reorder_idx,
        "surf_reorder_idx": surf_reorder_idx,
        # Keep original SDF for vol_log_sample_weight (computed later)
        "original_sdf_volume": original_sdf_volume,
        # Metadata
        "case_name": case_name,
        "sub_bin": sub_bin,
        "L": L,
        "surface_areas": surface_areas,
        # STL (for visualisation)
        "stl_vertices": stl_vertices.astype(np.float32),
        "stl_faces": stl_faces,
    }

    # ── Save intermediate PT ──
    out_path = os.path.join(intermediate_dir, f"{case_name}.pt")
    torch.save(pt_out, out_path)

    return case_name


# ── Welford pass ───────────────────────────────────────────────────


def run_welford_pass(
    train_case_ids: list[str],
    intermediate_dir: str,
) -> dict:
    """Compute z-score statistics from 700 train cases."""
    print(f"[Phase 5] Welford pass: {len(train_case_ids)} train cases ...")

    def load_fn(case_id):
        path = os.path.join(intermediate_dir, f"{case_id}.pt")
        return torch.load(path, map_location="cpu", weights_only=False)

    norm_stats = compute_norm_stats(train_case_ids, load_fn)
    return norm_stats


# ── Z-score pass ───────────────────────────────────────────────────


def _zscore_single_case(args):
    """Apply z-score to a single intermediate PT and overwrite."""
    case_name, intermediate_dir, norm_stats = args
    path = os.path.join(intermediate_dir, f"{case_name}.pt")
    pt = torch.load(path, map_location="cpu", weights_only=False)
    pt = apply_zscore(pt, norm_stats)
    torch.save(pt, path)
    return case_name


def run_zscore_pass(
    all_case_names: list[str],
    intermediate_dir: str,
    norm_stats: dict,
    num_workers: int = 40,
) -> None:
    """Apply z-score normalisation to all 800 cases using train stats."""
    print(f"[Phase 6] Z-score pass: {len(all_case_names)} cases ...")
    args_list = [(cn, intermediate_dir, norm_stats) for cn in all_case_names]
    with Pool(num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(_zscore_single_case, args_list),
                total=len(all_case_names),
                desc="Z-score",
            )
        )


# ── Precompute pass ────────────────────────────────────────────────


def _precompute_single_case(args):
    """Add precomputed fields to a single (post-z-score) PT."""
    case_name, intermediate_dir, cache_dir, rope_scales, cfg = args
    path = os.path.join(intermediate_dir, f"{case_name}.pt")
    pt = torch.load(path, map_location="cpu", weights_only=False)

    sub_bin = pt["sub_bin"]
    rope_scale = rope_scales[sub_bin]

    model_cfg = cfg.get("model", {})
    pt = add_precomputed_fields(
        pt,
        rope_scale=rope_scale,
        n_query_total=cfg.get("sampling", {}).get("N_query", 500_000),
        head_dim=model_cfg.get("head_dim", 64),
        rope_base=model_cfg.get("rope_base", 100.0),
        rope_dims=tuple(model_cfg.get("rope_dims", [22, 22, 20])),
        register_tokens=model_cfg.get("register_tokens", 16),
    )

    # Save to final cache directory
    out_path = os.path.join(cache_dir, f"{case_name}.pt")
    torch.save(pt, out_path)
    return case_name


def run_precompute_pass(
    all_case_names: list[str],
    intermediate_dir: str,
    cache_dir: str,
    rope_scales: dict[str, np.ndarray],
    cfg: dict,
    num_workers: int = 40,
) -> None:
    """Add precomputed fields and write final cache PTs."""
    print(f"[Phase 7] Precompute pass: {len(all_case_names)} cases ...")
    args_list = [
        (cn, intermediate_dir, cache_dir, rope_scales, cfg)
        for cn in all_case_names
    ]
    # Use sequential processing because precompute involves scipy/torch
    # and some operations don't release the GIL well with multiprocessing
    # For large datasets, consider reducing workers to avoid memory pressure
    workers = min(num_workers, 20)  # cap to avoid memory issues
    with Pool(workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(_precompute_single_case, args_list),
                total=len(all_case_names),
                desc="Precompute",
            )
        )


# ── RoPE scale computation ────────────────────────────────────────


def compute_all_rope_scales(
    train_case_names: list[str],
    intermediate_dir: str,
    sub_bin_L_map: dict[str, int],
) -> dict[str, np.ndarray]:
    """Collect all train centroids and compute per-sub-bin RoPE scales."""
    print("[Phase 7-pre] Computing RoPE scales from train centroids ...")
    all_centroids: list[np.ndarray] = []
    for cn in tqdm(train_case_names, desc="Collecting centroids"):
        path = os.path.join(intermediate_dir, f"{cn}.pt")
        pt = torch.load(path, map_location="cpu", weights_only=False)
        all_centroids.append(np.asarray(pt["leaf_centroid_norm"]))

    all_train_centroids = np.concatenate(all_centroids, axis=0)
    return compute_rope_scales(all_train_centroids, sub_bin_L_map)


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="HDB preprocessing pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--workers", type=int, default=40, help="Number of workers")
    parser.add_argument(
        "--skip-anomaly", action="store_true", help="Skip anomaly detection"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_workers = args.workers

    # ── Resolve paths ──
    data_cfg = cfg.get("data", {})
    cache_dir = os.path.expanduser(data_cfg.get("cache_dir", "~/scratch/HDB_cache"))
    raw_dir = os.path.expanduser(data_cfg.get("physical_pt_dir", "~/scratch/HDB"))
    manifest_path = os.path.expanduser(
        data_cfg.get("manifest_path", os.path.join(raw_dir, "manifest.json"))
    )
    intermediate_dir = os.path.join(cache_dir, "_intermediate")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    # ── Load manifest ──
    manifest = load_manifest(manifest_path)
    sub_bin_L = cfg.get("sub_bin_L", {})

    # Build case info list from manifest
    all_cases: list[dict] = []
    train_case_names: list[str] = []
    all_case_names: list[str] = []

    splits = manifest.get("splits", {})
    for split_name, cases in splits.items():
        for i, case_entry in enumerate(cases):
            # case_entry can be a dict or a string
            if isinstance(case_entry, dict):
                cn = case_entry["case_name"]
                sb = case_entry.get("sub_bin", "unknown")
            else:
                cn = str(case_entry)
                sb = "unknown"

            pt_path = os.path.join(raw_dir, f"{cn}.pt")
            L = sub_bin_L.get(sb, 40960)

            info = {
                "case_name": cn,
                "sub_bin": sb,
                "L": L,
                "case_id": len(all_cases),
                "pt_path": pt_path,
                "split": split_name,
            }
            all_cases.append(info)
            all_case_names.append(cn)
            if split_name == "train":
                train_case_names.append(cn)

    print(f"Manifest loaded: {len(all_cases)} total cases, "
          f"{len(train_case_names)} train")

    # ── Phase 0: Anomaly detection ──
    if not args.skip_anomaly:
        all_pt_paths = [c["pt_path"] for c in all_cases]
        anomaly_results = run_anomaly_detection(all_pt_paths, num_workers)

        # Filter out anomalous cases
        clean_cases = []
        for c in all_cases:
            result = anomaly_results.get(c["case_name"], {})
            if not result.get("is_anomaly", False):
                clean_cases.append(c)
            else:
                print(f"  Excluded: {c['case_name']} ({result.get('rule', '?')})")

        all_cases = clean_cases
        all_case_names = [c["case_name"] for c in all_cases]
        train_case_names = [c["case_name"] for c in all_cases if c["split"] == "train"]
        print(f"After filtering: {len(all_cases)} cases, "
              f"{len(train_case_names)} train")

    # ── Phase 1-4: Per-case processing ──
    print(f"\n[Phase 1-4] Processing {len(all_cases)} cases "
          f"with {num_workers} workers ...")
    bin_edges = build_sdf_bin_edges()
    coord_divisor = cfg.get("preprocessing", {}).get(
        "coord_divisor", COORD_DIVISOR
    )

    alpha = cfg.get("preprocessing", {}).get("kmeans_alpha", 0.65)

    process_fn = partial(
        process_single_case,
        raw_dir=raw_dir,
        intermediate_dir=intermediate_dir,
        bin_edges=bin_edges,
        coord_divisor=coord_divisor,
        alpha=alpha,
    )

    t0 = time.time()
    with Pool(num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_fn, all_cases),
                total=len(all_cases),
                desc="Per-case processing",
            )
        )
    print(f"  Phase 1-4 completed in {time.time() - t0:.1f}s")

    # ── Phase 5: Welford pass ──
    t0 = time.time()
    norm_stats = run_welford_pass(train_case_names, intermediate_dir)
    norm_stats_path = os.path.join(cache_dir, "norm_stats.json")
    save_norm_stats(norm_stats, norm_stats_path)
    print(f"  Welford pass completed in {time.time() - t0:.1f}s")
    print(f"  norm_stats saved to {norm_stats_path}")

    # ── Phase 6: Z-score pass ──
    t0 = time.time()
    run_zscore_pass(all_case_names, intermediate_dir, norm_stats, num_workers)
    print(f"  Z-score pass completed in {time.time() - t0:.1f}s")

    # ── Phase 7: RoPE scale + Precompute pass ──
    t0 = time.time()
    rope_scales = compute_all_rope_scales(
        train_case_names, intermediate_dir, sub_bin_L
    )
    print(f"  RoPE scales computed in {time.time() - t0:.1f}s")

    # Append rope_scales to norm_stats.json
    norm_stats_full = load_norm_stats(norm_stats_path)
    norm_stats_full["rope_scales"] = {
        k: v.tolist() for k, v in rope_scales.items()
    }
    save_norm_stats(norm_stats_full, norm_stats_path)

    t0 = time.time()
    run_precompute_pass(
        all_case_names, intermediate_dir, cache_dir, rope_scales, cfg, num_workers
    )
    print(f"  Precompute pass completed in {time.time() - t0:.1f}s")

    # ── Cleanup intermediate dir (optional) ──
    print(f"\n[Done] Cache built at: {cache_dir}")
    print(f"  Total cases: {len(all_case_names)}")
    print(f"  Train: {len(train_case_names)}")
    print(f"  norm_stats.json includes rope_scales for {len(rope_scales)} sub-bins")
    print(f"  Intermediate PTs at: {intermediate_dir}")
    print("  (Remove intermediate dir manually if no longer needed)")


if __name__ == "__main__":
    main()
