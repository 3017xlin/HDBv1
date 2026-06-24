#!/usr/bin/env python3
"""HDB Diagnostic Script 1: Data & Cache Integrity Checker.

Validates the entire data pipeline BEFORE training so you don't waste
GPU hours discovering a corrupted .pt or shape mismatch on step 3000.

Checks:
  1. manifest.json exists, parses, has train/val splits
  2. Every case listed in manifest has a corresponding .pt in cache_dir
  3. Every .pt loads successfully and contains ALL required keys
  4. Tensor shapes / dtypes are internally consistent (L matches, etc.)
  5. No NaN/Inf in any tensor
  6. norm_stats.json exists and contains expected keys + rope_scales
  7. Sub-bin distribution summary
  8. Quick spot-check: load 1 case fully and dry-run prepare_one_case

Usage:
    python debug_check_data.py --config config.yaml [--max-cases 0]
    (--max-cases 0 means check ALL cases; default 10 for quick scan)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


REQUIRED_PT_KEYS = [
    "leaf_centroid_norm",
    "leaf_stats",
    "leaf_sdf",
    "leaf_sdf_grad",
    "leaf_curvature_mean",
    "leaf_curvature_gauss",
    "leaf_pbd",
    "latent_point_top256",
    "latent_neighbor_top64",
    "point_pos_norm",
    "point_sdf",
    "point_sdf_grad",
    "point_curvature_mean",
    "point_curvature_gauss",
    "point_is_surface",
    "point_y_volume",
    "point_y_surface",
    "vol_reorder_idx",
    "surf_reorder_idx",
    "sub_bin",
    "L",
    "surface_areas",
    "encoder_pool",
    "point_leaf_id",
    "vol_log_sample_weight",
    "rope_cos",
    "rope_sin",
    "bigbird_fixed",
    "n_query_vol",
    "n_query_surf",
]

SUB_BIN_L = {
    '0-19_easy': 8192, '0-19_hard': 16384,
    '20-39_easy': 24576, '20-39_hard': 32768,
    '40-59_easy': 40960, '40-59_hard': 49152,
    '60-79_easy': 57344, '60-79_hard': 65536,
    '80-123_easy': 73728, '80-123_hard': 81920,
}


def _is_tensor_like(x):
    return isinstance(x, (torch.Tensor, np.ndarray))


def _to_np(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            return x.to(torch.float32).numpy()
        return x.numpy()
    return np.asarray(x)


def _check_nan_inf(name, x):
    arr = _to_np(x).astype(np.float32) if _is_tensor_like(x) else None
    if arr is None:
        return []
    issues = []
    if np.any(np.isnan(arr)):
        n = int(np.isnan(arr).sum())
        issues.append(f"  NaN detected in '{name}': {n} values")
    if np.any(np.isinf(arr)):
        n = int(np.isinf(arr).sum())
        issues.append(f"  Inf detected in '{name}': {n} values")
    return issues


def check_manifest(manifest_path):
    print(f"\n{'='*60}")
    print(f"[1] MANIFEST CHECK: {manifest_path}")
    print(f"{'='*60}")
    if not os.path.exists(manifest_path):
        print(f"  FAIL: manifest not found at {manifest_path}")
        return None

    with open(manifest_path) as f:
        manifest = json.load(f)

    if "splits" not in manifest:
        print(f"  FAIL: manifest missing 'splits' key. Keys found: {list(manifest.keys())}")
        return None

    splits = manifest["splits"]
    for split_name in ["train", "val"]:
        if split_name not in splits:
            print(f"  WARN: missing split '{split_name}'")
        else:
            print(f"  {split_name}: {len(splits[split_name])} cases")

    if "train_eval" in manifest:
        print(f"  train_eval: {len(manifest['train_eval'])} cases")
    else:
        print(f"  WARN: no 'train_eval' key in manifest")

    total = sum(len(v) for v in splits.values())
    print(f"  Total cases across all splits: {total}")
    print(f"  PASS")
    return manifest


def check_norm_stats(norm_stats_path):
    print(f"\n{'='*60}")
    print(f"[2] NORM STATS CHECK: {norm_stats_path}")
    print(f"{'='*60}")
    if not os.path.exists(norm_stats_path):
        print(f"  FAIL: norm_stats.json not found")
        return None

    with open(norm_stats_path) as f:
        ns = json.load(f)

    expected_top = ["leaf_stats", "leaf_sdf", "leaf_sdf_grad",
                    "leaf_curvature_mean", "leaf_curvature_gauss",
                    "point_sdf", "point_sdf_grad",
                    "point_curvature_mean", "point_curvature_gauss",
                    "point_y_volume", "point_y_surface"]

    found = []
    missing = []
    for k in expected_top:
        if k in ns:
            found.append(k)
        else:
            missing.append(k)

    print(f"  Found z-score keys: {len(found)}/{len(expected_top)}")
    if missing:
        print(f"  WARN missing z-score keys: {missing}")

    if "rope_scales" in ns:
        rs = ns["rope_scales"]
        print(f"  rope_scales: {len(rs)} sub-bins: {list(rs.keys())}")
    else:
        print(f"  WARN: no 'rope_scales' in norm_stats")

    print(f"  PASS")
    return ns


def check_single_pt(cache_dir, case_id, verbose=True):
    pt_path = os.path.join(cache_dir, f"{case_id}.pt")
    issues = []

    if not os.path.exists(pt_path):
        return [f"  MISSING: {pt_path}"]

    try:
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return [f"  LOAD ERROR for {case_id}: {e}"]

    missing_keys = [k for k in REQUIRED_PT_KEYS if k not in pt]
    if missing_keys:
        issues.append(f"  Missing keys: {missing_keys}")

    extra_keys = [k for k in pt.keys() if k not in REQUIRED_PT_KEYS
                  and k not in ("case_name", "stl_vertices", "stl_faces")]
    if extra_keys and verbose:
        issues.append(f"  Extra keys (info only): {extra_keys}")

    L = int(pt.get("L", 0))
    sub_bin = pt.get("sub_bin", "unknown")
    expected_L = SUB_BIN_L.get(sub_bin)

    if expected_L is not None and L != expected_L:
        issues.append(f"  L mismatch: pt['L']={L} but sub_bin='{sub_bin}' expects {expected_L}")

    shape_checks = {
        "leaf_centroid_norm": (L, 3),
        "leaf_stats": (L, 21),
        "leaf_sdf": (L,),
        "leaf_sdf_grad": (L, 3),
        "leaf_curvature_mean": (L,),
        "leaf_curvature_gauss": (L,),
        "leaf_pbd": (L, 16),
        "latent_point_top256": (L, 256),
        "latent_neighbor_top64": (L, 64),
    }

    for key, expected_shape in shape_checks.items():
        if key in pt:
            actual = tuple(_to_np(pt[key]).shape)
            if actual != expected_shape:
                issues.append(f"  Shape mismatch '{key}': got {actual}, expected {expected_shape}")

    if "point_pos_norm" in pt:
        N_total = _to_np(pt["point_pos_norm"]).shape[0]
        for key in ["point_sdf", "point_sdf_grad", "point_curvature_mean",
                     "point_curvature_gauss", "point_is_surface", "point_leaf_id"]:
            if key in pt:
                n = _to_np(pt[key]).shape[0]
                if n != N_total:
                    issues.append(f"  Length mismatch '{key}': {n} vs N_total={N_total}")

    if "vol_reorder_idx" in pt and "surf_reorder_idx" in pt:
        N_vol = _to_np(pt["vol_reorder_idx"]).shape[0]
        N_surf = _to_np(pt["surf_reorder_idx"]).shape[0]
        if "point_pos_norm" in pt:
            if N_vol + N_surf != N_total:
                issues.append(f"  N_vol({N_vol}) + N_surf({N_surf}) != N_total({N_total})")

        if "point_y_volume" in pt:
            pv_shape = _to_np(pt["point_y_volume"]).shape
            if pv_shape[0] != N_vol:
                issues.append(f"  point_y_volume rows={pv_shape[0]} != N_vol={N_vol}")
            if len(pv_shape) > 1 and pv_shape[1] != 5:
                issues.append(f"  point_y_volume cols={pv_shape[1]} != 5")

        if "point_y_surface" in pt:
            ps_shape = _to_np(pt["point_y_surface"]).shape
            if ps_shape[0] != N_surf:
                issues.append(f"  point_y_surface rows={ps_shape[0]} != N_surf={N_surf}")

    if "encoder_pool" in pt:
        ep = pt["encoder_pool"]
        ep_shape = tuple(ep.shape) if isinstance(ep, (torch.Tensor, np.ndarray)) else None
        if ep_shape and ep_shape != (L, 256, 10):
            issues.append(f"  encoder_pool shape {ep_shape} != ({L}, 256, 10)")

    if "bigbird_fixed" in pt:
        bb = _to_np(pt["bigbird_fixed"])
        if bb.shape != (L, 80):
            issues.append(f"  bigbird_fixed shape {bb.shape} != ({L}, 80)")

    if "rope_cos" in pt:
        rc = _to_np(pt["rope_cos"])
        expected_rope_rows = L + 16
        if rc.shape[0] != expected_rope_rows:
            issues.append(f"  rope_cos rows={rc.shape[0]} != L+16={expected_rope_rows}")

    for key in REQUIRED_PT_KEYS:
        if key in pt and _is_tensor_like(pt[key]):
            nan_inf = _check_nan_inf(key, pt[key])
            issues.extend(nan_inf)

    return issues


def check_cases(cache_dir, manifest, max_cases):
    print(f"\n{'='*60}")
    print(f"[3] CACHE PT CHECK: {cache_dir}")
    print(f"{'='*60}")

    all_case_ids = []
    splits = manifest.get("splits", {})
    for split_name, cases in splits.items():
        for c in cases:
            cid = c if isinstance(c, str) else str(c)
            all_case_ids.append(cid)

    total = len(all_case_ids)
    if max_cases > 0:
        check_ids = all_case_ids[:max_cases]
        print(f"  Checking {len(check_ids)}/{total} cases (use --max-cases 0 for all)")
    else:
        check_ids = all_case_ids
        print(f"  Checking ALL {total} cases")

    sub_bin_counts = {}
    n_pass = 0
    n_fail = 0
    all_issues = {}

    for i, cid in enumerate(check_ids):
        issues = check_single_pt(cache_dir, cid, verbose=(i < 3))
        if issues:
            n_fail += 1
            all_issues[cid] = issues
            if n_fail <= 5:
                print(f"\n  FAIL: {cid}")
                for iss in issues:
                    print(f"    {iss}")
        else:
            n_pass += 1
            pt_path = os.path.join(cache_dir, f"{cid}.pt")
            if os.path.exists(pt_path):
                pt = torch.load(pt_path, map_location="cpu", weights_only=False)
                sb = pt.get("sub_bin", "unknown")
                sub_bin_counts[sb] = sub_bin_counts.get(sb, 0) + 1

        if (i + 1) % 50 == 0:
            print(f"  ... checked {i+1}/{len(check_ids)}  pass={n_pass} fail={n_fail}")

    print(f"\n  Results: {n_pass} PASS, {n_fail} FAIL out of {len(check_ids)} checked")
    if n_fail > 5:
        print(f"  (showing first 5 failures, {n_fail - 5} more suppressed)")

    if sub_bin_counts:
        print(f"\n  Sub-bin distribution (from passed cases):")
        for sb in sorted(sub_bin_counts.keys()):
            print(f"    {sb}: {sub_bin_counts[sb]} cases (L={SUB_BIN_L.get(sb, '?')})")

    return n_fail == 0


def dry_run_prepare(cache_dir, manifest):
    print(f"\n{'='*60}")
    print(f"[4] DRY-RUN: prepare_one_case on first train case")
    print(f"{'='*60}")

    train_ids = manifest.get("splits", {}).get("train", [])
    if not train_ids:
        print(f"  SKIP: no train cases in manifest")
        return True

    cid = train_ids[0] if isinstance(train_ids[0], str) else str(train_ids[0])
    pt_path = os.path.join(cache_dir, f"{cid}.pt")
    if not os.path.exists(pt_path):
        print(f"  SKIP: {pt_path} not found")
        return True

    try:
        from hdb.dataset.prefetcher import prepare_one_case
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
        pt["_case_id"] = cid

        print(f"  Loading case '{cid}' (sub_bin={pt.get('sub_bin')}, L={pt.get('L')})")
        item = prepare_one_case(pt, 0, epoch=0, encoder_k=32, n_query=500_000)

        print(f"  prepare_one_case returned {len(item)} keys:")
        for k, v in sorted(item.items()):
            if isinstance(v, torch.Tensor):
                print(f"    {k:30s}  {str(v.shape):20s}  {v.dtype}")
            elif isinstance(v, np.ndarray):
                print(f"    {k:30s}  {str(v.shape):20s}  np.{v.dtype}")
            else:
                print(f"    {k:30s}  = {v}")

        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v.float()).any():
                    print(f"  FAIL: NaN in prepared item['{k}']")
                    return False
                if torch.isinf(v.float()).any():
                    print(f"  FAIL: Inf in prepared item['{k}']")
                    return False

        print(f"  PASS: prepare_one_case succeeded, no NaN/Inf")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def dry_run_stack_batch(cache_dir, manifest):
    print(f"\n{'='*60}")
    print(f"[5] DRY-RUN: stack_batch with 2 cases from same sub_bin")
    print(f"{'='*60}")

    train_ids = manifest.get("splits", {}).get("train", [])
    if len(train_ids) < 2:
        print(f"  SKIP: fewer than 2 train cases")
        return True

    try:
        from hdb.dataset.prefetcher import prepare_one_case, stack_batch

        pt_by_sb = {}
        for cid_raw in train_ids:
            cid = cid_raw if isinstance(cid_raw, str) else str(cid_raw)
            pt_path = os.path.join(cache_dir, f"{cid}.pt")
            if not os.path.exists(pt_path):
                continue
            pt = torch.load(pt_path, map_location="cpu", weights_only=False)
            sb = pt.get("sub_bin", "unknown")
            if sb not in pt_by_sb:
                pt_by_sb[sb] = []
            pt_by_sb[sb].append((cid, pt))
            if any(len(v) >= 2 for v in pt_by_sb.values()):
                break

        pair_sb = None
        for sb, pts in pt_by_sb.items():
            if len(pts) >= 2:
                pair_sb = sb
                break

        if pair_sb is None:
            print(f"  SKIP: couldn't find 2 cases from same sub_bin in first few cases")
            return True

        items = []
        for i, (cid, pt) in enumerate(pt_by_sb[pair_sb][:2]):
            pt["_case_id"] = i
            item = prepare_one_case(pt, i, epoch=0, encoder_k=32, n_query=500_000)
            items.append(item)

        n_qv = min(it["n_query_vol"] for it in items)
        from hdb.dataset.prefetcher import _trim_queries_to_nqv
        for it in items:
            _trim_queries_to_nqv(it, n_qv, it["n_query_vol"])

        batch = stack_batch(items)
        print(f"  Stacked batch from sub_bin='{pair_sb}' (B=2):")
        for k, v in sorted(batch.items()):
            if isinstance(v, torch.Tensor):
                print(f"    {k:30s}  {str(v.shape):25s}  {v.dtype}")
            else:
                print(f"    {k:30s}  = {v}")

        print(f"  PASS: stack_batch succeeded")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="HDB Data & Cache Integrity Checker")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--max-cases", type=int, default=10,
                        help="Max cases to check (0 = all)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    cache_dir = os.path.expanduser(data_cfg.get("cache_dir", "~/scratch/HDB_cache"))
    manifest_path = os.path.expanduser(data_cfg.get("manifest_path", "~/scratch/manifest.json"))
    norm_stats_path = os.path.expanduser(data_cfg.get("norm_stats_path",
                                                        os.path.join(cache_dir, "norm_stats.json")))

    manifest_in_cache = os.path.join(cache_dir, "manifest.json")

    print("=" * 60)
    print("HDB DATA & CACHE INTEGRITY CHECKER")
    print("=" * 60)
    print(f"  config:        {args.config}")
    print(f"  cache_dir:     {cache_dir}")
    print(f"  manifest_path: {manifest_path}")
    print(f"  norm_stats:    {norm_stats_path}")
    print(f"  max_cases:     {args.max_cases if args.max_cases > 0 else 'ALL'}")

    all_ok = True

    manifest = check_manifest(manifest_path)
    if manifest is None:
        print(f"\n  Trying manifest inside cache_dir: {manifest_in_cache}")
        manifest = check_manifest(manifest_in_cache)
    if manifest is None:
        print("\nABORT: Cannot proceed without a valid manifest.")
        sys.exit(1)

    check_norm_stats(norm_stats_path)

    cases_ok = check_cases(cache_dir, manifest, args.max_cases)
    if not cases_ok:
        all_ok = False

    prep_ok = dry_run_prepare(cache_dir, manifest)
    if not prep_ok:
        all_ok = False

    stack_ok = dry_run_stack_batch(cache_dir, manifest)
    if not stack_ok:
        all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("ALL CHECKS PASSED - data pipeline looks healthy")
    else:
        print("SOME CHECKS FAILED - fix issues above before training")
    print(f"{'='*60}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
