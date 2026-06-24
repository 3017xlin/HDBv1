#!/usr/bin/env python3
"""Build manifest.json for HDB preprocessing pipeline.

Pipeline (the one true ordering):

  Step 1.  Scan ``physical_pt_dir`` for every ``*.pt`` (≈ 7154 cases).
           One parallel pass per PT extracting in a single load:
             * the 19 MAD anomaly indicators (preprocess/anomaly_detect.py)
             * ``n_vol``  — ``volume_pos.shape[0]``
             * ``n_surf`` — ``surface_pos.shape[0]``
             * ``n_buildings`` — DBSCAN clusters on ``stl_vertices``

  Step 2.  Run ``detect_anomalies_mad`` over the 19-indicator matrix and
           drop anomalous cases.

  Step 3.  Range by HARD cutoff on ``n_buildings``::

               n ≤ 19  → "0-19"
               n ≤ 39  → "20-39"
               n ≤ 59  → "40-59"
               n ≤ 79  → "60-79"
               else    → "80-123"

           Within each range, sort by ``n_vol``; lower half ⇒ ``easy``,
           upper half ⇒ ``hard``.  Exactly 10 sub-bins.

  Step 4.  From each sub-bin, sample WITHOUT replacement:
             70 train + 5 val + 5 test  (defaults; override via CLI).

  Step 5.  Write ``manifest.json``.

After this, the slow preprocess does NOT need to redo anomaly:

    nohup python preprocess/build_cache.py --config config.yaml \
        --workers 80 --skip-anomaly > ~/scratch/preprocess.log 2>&1 &
    tail -f ~/scratch/preprocess.log

Usage::

    nohup python build_manifest.py --config config.yaml --workers 80 \
        > ~/scratch/build_manifest.log 2>&1 &
    tail -f ~/scratch/build_manifest.log

    # If you want to re-split (different seed / counts) without re-reading
    # every PT, point at a feature cache:
    nohup python build_manifest.py --config config.yaml --workers 80 \
        --features-cache ~/scratch/manifest_features.json \
        > ~/scratch/build_manifest.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

# ── Make `from hdb.preprocess.X import Y` (existing convention) and a
#    bare `from preprocess.X import Y` (when the tarball extracts as
#    HDBv1-<branch>/ without an `hdb/` symlink) both work. ────────────
_REPO_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _REPO_ROOT.parent):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from hdb.preprocess.anomaly_detect import (
        compute_case_stats, detect_anomalies_mad,
    )
except ImportError:
    from preprocess.anomaly_detect import (  # type: ignore
        compute_case_stats, detect_anomalies_mad,
    )


SUB_BIN_RANGES = ["0-19", "20-39", "40-59", "60-79", "80-123"]
DIFFICULTIES = ["easy", "hard"]
SUB_BIN_ORDER = [f"{r}_{d}" for r in SUB_BIN_RANGES for d in DIFFICULTIES]


def range_from_nbuildings(n: int) -> str:
    """Hard cutoffs — NOT quantiles."""
    if n <= 19:
        return "0-19"
    if n <= 39:
        return "20-39"
    if n <= 59:
        return "40-59"
    if n <= 79:
        return "60-79"
    return "80-123"


# ─── per-case worker ───────────────────────────────────────────────

_DBSCAN_EPS = 1.0
_DBSCAN_MIN_SAMPLES = 5
_DBSCAN_MAX_VERTS = 200_000  # cap subsample size for huge meshes


def _set_dbscan(eps: float, min_samples: int, max_verts: int) -> None:
    global _DBSCAN_EPS, _DBSCAN_MIN_SAMPLES, _DBSCAN_MAX_VERTS
    _DBSCAN_EPS = eps
    _DBSCAN_MIN_SAMPLES = min_samples
    _DBSCAN_MAX_VERTS = max_verts


def _per_case_worker(pt_path: str) -> dict | None:
    """Load one PT, compute anomaly stats + n_vol + n_surf + n_buildings."""
    try:
        import torch
        from sklearn.cluster import DBSCAN

        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
        pt_np: dict = {}
        for k, v in pt.items():
            if isinstance(v, torch.Tensor):
                pt_np[k] = v.numpy()
            else:
                pt_np[k] = v

        case_name = os.path.basename(pt_path)[:-3]
        pt_np.setdefault("case_name", case_name)

        # 19 anomaly indicators
        stats = compute_case_stats(pt_np)

        # Point counts
        n_vol = int(pt_np["volume_pos"].shape[0])
        n_surf = int(pt_np["surface_pos"].shape[0])

        # n_buildings via DBSCAN on STL vertices
        stl_vertices = np.asarray(pt_np["stl_vertices"], dtype=np.float32)
        if stl_vertices.shape[0] > _DBSCAN_MAX_VERTS:
            rng = np.random.RandomState(42)
            idx = rng.choice(stl_vertices.shape[0],
                             size=_DBSCAN_MAX_VERTS, replace=False)
            stl_for_db = stl_vertices[idx]
        else:
            stl_for_db = stl_vertices

        db = DBSCAN(eps=_DBSCAN_EPS,
                    min_samples=_DBSCAN_MIN_SAMPLES,
                    n_jobs=1).fit(stl_for_db)
        labels = db.labels_
        # exclude noise (-1) from cluster count
        n_buildings = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
        n_buildings = max(n_buildings, 1)  # degenerate-safe lower bound

        out = {
            "case_name": case_name,
            "pt_path": pt_path,
            "n_vol": n_vol,
            "n_surf": n_surf,
            "n_buildings": n_buildings,
        }
        for k, v in stats.items():
            if k == "case_name":
                continue
            out[k] = v
        return out
    except Exception as e:
        return {"case_name": os.path.basename(pt_path)[:-3],
                "pt_path": pt_path,
                "_error": f"{type(e).__name__}: {e}"}


def _init_pool(eps: float, min_samples: int, max_verts: int) -> None:
    _set_dbscan(eps, min_samples, max_verts)


# ─── splits ─────────────────────────────────────────────────────────

def fixed_count_split(
    cases: list[dict],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """Per sub-bin, sample without replacement: n_train + n_val + n_test."""
    import random
    rng = random.Random(seed)
    by_sb: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_sb[c["sub_bin"]].append(c)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    short: dict[str, dict[str, int]] = {}

    for sb in SUB_BIN_ORDER:
        lst = sorted(by_sb.get(sb, []), key=lambda c: c["case_name"])
        rng.shuffle(lst)
        n_avail = len(lst)
        # Greedy allocation respecting available count.
        take_train = min(n_train, n_avail)
        take_val = min(n_val, n_avail - take_train)
        take_test = min(n_test, n_avail - take_train - take_val)

        if take_train < n_train or take_val < n_val or take_test < n_test:
            short[sb] = {
                "available": n_avail,
                "requested": n_train + n_val + n_test,
                "got_train": take_train,
                "got_val": take_val,
                "got_test": take_test,
            }

        for i, c in enumerate(lst):
            if i < take_train:
                bucket = "train"
            elif i < take_train + take_val:
                bucket = "val"
            elif i < take_train + take_val + take_test:
                bucket = "test"
            else:
                continue  # rest of sub-bin discarded (per the spec)
            splits[bucket].append({
                "case_name": c["case_name"],
                "sub_bin": c["sub_bin"],
                "n_buildings": c["n_buildings"],
                "n_vol": c["n_vol"],
                "n_surf": c["n_surf"],
            })
    for k in splits:
        splits[k].sort(key=lambda d: d["case_name"])
    return splits, short


# ─── main ───────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--n-train", type=int, default=70,
                   help="Train cases per sub-bin (default 70)")
    p.add_argument("--n-val", type=int, default=5,
                   help="Val cases per sub-bin (default 5)")
    p.add_argument("--n-test", type=int, default=5,
                   help="Test cases per sub-bin (default 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--physical-pt-dir", default=None,
                   help="Override cfg.data.physical_pt_dir.")
    p.add_argument("--out", default=None,
                   help="Override cfg.data.manifest_path output location.")
    p.add_argument("--features-cache", default=None,
                   help="JSON file caching the per-case features (anomaly "
                        "indicators + n_vol + n_surf + n_buildings). If "
                        "present, reuse instead of re-reading every PT.")
    p.add_argument("--dbscan-eps", type=float, default=1.0,
                   help="DBSCAN eps (meters, default 1.0)")
    p.add_argument("--dbscan-min-samples", type=int, default=5,
                   help="DBSCAN min_samples (default 5)")
    p.add_argument("--dbscan-max-verts", type=int, default=200_000,
                   help="Subsample STL vertices to at most this many "
                        "before DBSCAN (default 200000)")
    p.add_argument("--no-anomaly-filter", action="store_true",
                   help="Skip anomaly filtering (NOT recommended).")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    data_cfg = cfg.get("data", {}) or {}

    raw_dir = args.physical_pt_dir or data_cfg.get("physical_pt_dir",
                                                   "~/scratch/HDB")
    raw_dir = str(Path(os.path.expanduser(raw_dir)).resolve())
    out_path = args.out or data_cfg.get("manifest_path",
                                        "~/scratch/manifest.json")
    out_path = str(Path(os.path.expanduser(out_path)).resolve())

    if not os.path.isdir(raw_dir):
        print(f"ERROR: physical_pt_dir does not exist: {raw_dir}",
              file=sys.stderr)
        return 1

    pt_paths = sorted(str(p) for p in Path(raw_dir).glob("*.pt"))
    if not pt_paths:
        print(f"ERROR: no *.pt files found under {raw_dir}", file=sys.stderr)
        return 1

    print(f"[manifest] {len(pt_paths)} cases under {raw_dir}")
    print(f"[manifest] output: {out_path}")
    print(f"[manifest] workers: {args.workers}  seed: {args.seed}")
    print(f"[manifest] per sub-bin: train={args.n_train} "
          f"val={args.n_val} test={args.n_test} "
          f"(total = {args.n_train + args.n_val + args.n_test} × 10 = "
          f"{(args.n_train + args.n_val + args.n_test) * 10})")
    print(f"[manifest] DBSCAN eps={args.dbscan_eps} "
          f"min_samples={args.dbscan_min_samples}  "
          f"max_verts={args.dbscan_max_verts}")

    # ── Step 1: features (anomaly stats + counts + DBSCAN n_buildings) ──
    feats: list[dict]
    if args.features_cache and os.path.exists(args.features_cache):
        with open(args.features_cache) as f:
            feats = json.load(f)
        print(f"[step 1] loaded features from cache: "
              f"{args.features_cache} ({len(feats)} cases)")
    else:
        print(f"[step 1] computing features in parallel ...")
        feats = []
        with Pool(args.workers,
                  initializer=_init_pool,
                  initargs=(args.dbscan_eps,
                            args.dbscan_min_samples,
                            args.dbscan_max_verts)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_per_case_worker,
                                                       pt_paths)):
                if rec is not None:
                    feats.append(rec)
                if (i + 1) % 100 == 0 or (i + 1) == len(pt_paths):
                    print(f"  [{i + 1}/{len(pt_paths)}] features extracted",
                          flush=True)
        if args.features_cache:
            os.makedirs(os.path.dirname(args.features_cache) or ".",
                        exist_ok=True)
            with open(args.features_cache, "w") as f:
                json.dump(feats, f)
            print(f"[step 1] wrote features cache: {args.features_cache}")

    # Drop records that errored out during loading.
    bad = [f for f in feats if "_error" in f]
    feats = [f for f in feats if "_error" not in f]
    if bad:
        print(f"[step 1] WARNING: {len(bad)} cases failed to load:")
        for f in bad[:10]:
            print(f"    {f['case_name']}: {f.get('_error')}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")

    # ── Step 2: anomaly filter ──
    if args.no_anomaly_filter:
        kept = feats
        anomaly_results: dict[str, dict] = {}
        print(f"[step 2] anomaly filter SKIPPED (--no-anomaly-filter)")
    else:
        anomaly_results = detect_anomalies_mad(feats)
        kept = [f for f in feats
                if not anomaly_results.get(f["case_name"], {}).get(
                    "is_anomaly", False)]
        n_drop = len(feats) - len(kept)
        print(f"[step 2] anomaly: dropped {n_drop} / {len(feats)} "
              f"({100 * n_drop / max(len(feats), 1):.1f} %)  "
              f"→ {len(kept)} kept")
        by_rule: dict[str, int] = defaultdict(int)
        for cn, r in anomaly_results.items():
            if r.get("is_anomaly"):
                by_rule[r.get("rule", "?")] += 1
        for rule, c in sorted(by_rule.items()):
            print(f"    rule={rule:<18s} {c}")

    # ── Step 3: range (hard cutoff on n_buildings) + easy/hard by n_vol ──
    for f in kept:
        f["range"] = range_from_nbuildings(f["n_buildings"])

    by_range: dict[str, list[dict]] = defaultdict(list)
    for f in kept:
        by_range[f["range"]].append(f)

    range_n_vol_median: dict[str, int] = {}
    for r in SUB_BIN_RANGES:
        lst = by_range.get(r, [])
        if not lst:
            continue
        lst.sort(key=lambda c: c["n_vol"])
        half = len(lst) // 2
        if lst:
            range_n_vol_median[r] = int(lst[half - 1]["n_vol"]) \
                                    if half > 0 else int(lst[0]["n_vol"])
        for i, c in enumerate(lst):
            c["sub_bin"] = f"{r}_{'easy' if i < half else 'hard'}"

    per_sub_bin_total: dict[str, int] = defaultdict(int)
    for c in kept:
        per_sub_bin_total[c["sub_bin"]] += 1

    print(f"[step 3] sub-bin populations (after binning, before sampling):")
    for sb in SUB_BIN_ORDER:
        n = per_sub_bin_total.get(sb, 0)
        need = args.n_train + args.n_val + args.n_test
        flag = "" if n >= need else f"  ⚠ < {need}"
        print(f"    {sb:16s}: {n}{flag}")

    # ── Step 4: per sub-bin fixed-count sampling ──
    splits, short = fixed_count_split(
        kept, args.n_train, args.n_val, args.n_test, args.seed)

    if short:
        print(f"[step 4] WARNING: {len(short)} sub-bin(s) under-supplied:")
        for sb, info in short.items():
            print(f"    {sb}: {info}")

    # ── Step 5: write manifest ──
    n_buildings_sorted = sorted(c["n_buildings"] for c in kept)
    n_vol_sorted = sorted(c["n_vol"] for c in kept)
    n_surf_sorted = sorted(c["n_surf"] for c in kept)

    manifest = {
        "splits": splits,
        "seed": args.seed,
        "physical_pt_dir": raw_dir,
        "n_scanned": len(feats) + len(bad),
        "n_load_failed": len(bad),
        "n_dropped_anomaly": (len(feats) - len(kept))
                              if not args.no_anomaly_filter else 0,
        "n_kept_after_anomaly": len(kept),
        "counts": {k: len(v) for k, v in splits.items()},
        "per_sub_bin_pool": {sb: per_sub_bin_total.get(sb, 0)
                             for sb in SUB_BIN_ORDER},
        "per_sub_bin_under_supplied": short,
        "range_n_vol_easy_hard_threshold": range_n_vol_median,
        "n_buildings_summary": {
            "min": int(n_buildings_sorted[0]) if n_buildings_sorted else 0,
            "max": int(n_buildings_sorted[-1]) if n_buildings_sorted else 0,
            "median": int(n_buildings_sorted[len(n_buildings_sorted) // 2])
                      if n_buildings_sorted else 0,
        },
        "n_vol_summary": {
            "min": int(n_vol_sorted[0]) if n_vol_sorted else 0,
            "max": int(n_vol_sorted[-1]) if n_vol_sorted else 0,
            "median": int(n_vol_sorted[len(n_vol_sorted) // 2])
                      if n_vol_sorted else 0,
        },
        "n_surf_summary": {
            "min": int(n_surf_sorted[0]) if n_surf_sorted else 0,
            "max": int(n_surf_sorted[-1]) if n_surf_sorted else 0,
            "median": int(n_surf_sorted[len(n_surf_sorted) // 2])
                      if n_surf_sorted else 0,
        },
        "dbscan": {
            "eps": args.dbscan_eps,
            "min_samples": args.dbscan_min_samples,
            "max_verts": args.dbscan_max_verts,
        },
        "anomaly_results": {
            cn: {"is_anomaly": r.get("is_anomaly", False),
                 "rule": r.get("rule", "normal")}
            for cn, r in anomaly_results.items()
        } if not args.no_anomaly_filter else {},
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[manifest] split counts:")
    for k, v in manifest["counts"].items():
        print(f"    {k:6s}: {v}")
    print(f"\n[manifest] wrote {out_path}")
    print("\nNext step (anomaly already done — pass --skip-anomaly):")
    print("  nohup python preprocess/build_cache.py --config config.yaml "
          "--workers 80 --skip-anomaly > ~/scratch/preprocess.log 2>&1 &")
    print("  tail -f ~/scratch/preprocess.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
