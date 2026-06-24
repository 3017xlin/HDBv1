#!/usr/bin/env python3
"""Build manifest.json for HDB preprocessing pipeline.

Pipeline (the one true ordering):

  1. Scan ``physical_pt_dir`` for every ``*.pt`` (≈ 7154 cases).
  2. Single parallel pass over every PT extracting in one load:
       * the 19 MAD anomaly indicators (preprocess/anomaly_detect.py)
       * ``n_buildings`` — connected components of ``stl_faces``
       * ``n_vol``       — ``volume_pos.shape[0]``
  3. Run ``detect_anomalies_mad`` → flag anomalous cases → drop them.
  4. Assign each surviving case to a **range** by ``n_buildings``
     (hard cutoffs, not quantiles)::

        n ≤ 19  → "0-19"
        n ≤ 39  → "20-39"
        n ≤ 59  → "40-59"
        n ≤ 79  → "60-79"
        else    → "80-123"

  5. Within each range, sort by ``n_vol``; lower half ⇒ ``easy``,
     upper half ⇒ ``hard``.  Gives exactly 10 sub-bins.
  6. Per sub-bin stratified, without-replacement split into
     train / val / test (default 70 / 15 / 15).
  7. Write ``manifest.json`` (path from ``cfg.data.manifest_path``).

After this, the slow preprocess does NOT need to redo anomaly:

    python preprocess/build_cache.py --config config.yaml \
        --workers 80 --skip-anomaly

Usage::

    python build_manifest.py --config config.yaml --workers 80
    # Optional: cache the heavy per-case features so subsequent runs with
    # different --train-frac / --seed don't re-read every PT.
    python build_manifest.py --config config.yaml --workers 80 \
        --features-cache ~/scratch/manifest_features.json
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

# ── Allow both `from hdb.preprocess.X import Y` (existing build_cache
#    layout) and a plain `from preprocess.X import Y` (when extracted as
#    HDBv1-<branch>/ without an `hdb/` symlink). ────────────────────────
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


# ─── range mapping (hard cutoffs, NOT quantiles) ───────────────────

def range_from_nbuildings(n: int) -> str:
    if n <= 19:
        return "0-19"
    if n <= 39:
        return "20-39"
    if n <= 59:
        return "40-59"
    if n <= 79:
        return "60-79"
    return "80-123"


# ─── per-case worker (loads PT once, returns everything) ───────────

def _per_case_worker(pt_path: str) -> dict:
    """Load one PT, compute anomaly stats + n_buildings + n_vol."""
    import torch
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    pt = torch.load(pt_path, map_location="cpu", weights_only=False)

    # tensors → numpy for compute_case_stats and downstream ops
    pt_np: dict = {}
    for k, v in pt.items():
        if isinstance(v, torch.Tensor):
            pt_np[k] = v.numpy()
        else:
            pt_np[k] = v

    case_name = os.path.basename(pt_path)[:-3]  # strip ".pt"
    pt_np.setdefault("case_name", case_name)

    # Anomaly indicators (19 floats)
    stats = compute_case_stats(pt_np)

    # n_vol
    n_vol = int(pt_np["volume_pos"].shape[0])

    # n_buildings — connected components of the STL face graph
    faces = np.asarray(pt_np["stl_faces"], dtype=np.int64)
    n_vert = int(faces.max()) + 1
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.bool_)
    graph = csr_matrix((data, (rows, cols)), shape=(n_vert, n_vert))
    n_buildings, _ = connected_components(graph, directed=False)

    out = {
        "case_name": case_name,
        "pt_path": pt_path,
        "n_vol": n_vol,
        "n_buildings": int(n_buildings),
    }
    # Inject the 19 anomaly indicators directly (stats already has case_name)
    for k, v in stats.items():
        if k == "case_name":
            continue
        out[k] = v
    return out


# ─── splits ─────────────────────────────────────────────────────────

def stratified_split(
    cases: list[dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[dict]]:
    import random
    rng = random.Random(seed)
    by_sb: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_sb[c["sub_bin"]].append(c)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for sb in SUB_BIN_ORDER:
        lst = sorted(by_sb.get(sb, []), key=lambda c: c["case_name"])
        rng.shuffle(lst)
        n = len(lst)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for i, c in enumerate(lst):
            if i < n_train:
                bucket = "train"
            elif i < n_train + n_val:
                bucket = "val"
            else:
                bucket = "test"
            splits[bucket].append({
                "case_name": c["case_name"],
                "sub_bin": c["sub_bin"],
                "n_buildings": c["n_buildings"],
                "n_vol": c["n_vol"],
            })
    for k in splits:
        splits[k].sort(key=lambda d: d["case_name"])
    return splits


# ─── main ───────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--physical-pt-dir", default=None,
                   help="Override cfg.data.physical_pt_dir.")
    p.add_argument("--out", default=None,
                   help="Override cfg.data.manifest_path output location.")
    p.add_argument("--features-cache", default=None,
                   help="JSON file caching per-case features. If exists, "
                        "load it instead of re-reading every PT.")
    p.add_argument("--no-anomaly-filter", action="store_true",
                   help="Skip anomaly filtering (NOT recommended).")
    args = p.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        print(f"ERROR: --train-frac + --val-frac = "
              f"{args.train_frac + args.val_frac:.2f} leaves no test split.",
              file=sys.stderr)
        return 2

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

    print(f"[manifest] scanning {len(pt_paths)} cases under {raw_dir}")
    print(f"[manifest] output: {out_path}")
    print(f"[manifest] workers: {args.workers}  seed: {args.seed}")
    print(f"[manifest] splits: train={args.train_frac:.2f} "
          f"val={args.val_frac:.2f} "
          f"test={1.0 - args.train_frac - args.val_frac:.2f}")

    # ── 1+2. Per-case features (loads every PT once) ──
    feats: list[dict]
    if args.features_cache and os.path.exists(args.features_cache):
        with open(args.features_cache) as f:
            feats = json.load(f)
        print(f"[manifest] loaded features from cache: "
              f"{args.features_cache} ({len(feats)} cases)")
    else:
        print(f"[manifest] computing features in parallel ...")
        feats = []
        with Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(_per_case_worker,
                                                       pt_paths)):
                feats.append(rec)
                if (i + 1) % 100 == 0 or (i + 1) == len(pt_paths):
                    print(f"  [{i + 1}/{len(pt_paths)}] features extracted",
                          flush=True)
        if args.features_cache:
            os.makedirs(os.path.dirname(args.features_cache) or ".",
                        exist_ok=True)
            with open(args.features_cache, "w") as f:
                json.dump(feats, f)
            print(f"[manifest] wrote features cache: {args.features_cache}")

    # ── 3. Anomaly filter (MAD 19) ──
    if args.no_anomaly_filter:
        kept = feats
        anomaly_results: dict[str, dict] = {}
        print(f"[anomaly] SKIPPED (--no-anomaly-filter)")
    else:
        # detect_anomalies_mad wants the same 19 fields per case (plus name).
        # Our feats already contain those (compute_case_stats output spread in).
        anomaly_results = detect_anomalies_mad(feats)
        kept = [f for f in feats
                if not anomaly_results.get(f["case_name"], {}).get(
                    "is_anomaly", False)]
        n_drop = len(feats) - len(kept)
        print(f"[anomaly] {n_drop} dropped / {len(feats)} scanned "
              f"({100 * n_drop / max(len(feats), 1):.1f} %)")
        # Print rule breakdown
        by_rule: dict[str, int] = defaultdict(int)
        for cn, r in anomaly_results.items():
            if r.get("is_anomaly"):
                by_rule[r.get("rule", "?")] += 1
        for rule, c in sorted(by_rule.items()):
            print(f"  rule={rule:<18s} {c}")

    # ── 4. Range by n_buildings (hard cutoffs) ──
    for f in kept:
        f["range"] = range_from_nbuildings(f["n_buildings"])

    # ── 5. Within each range, easy/hard by n_vol median ──
    by_range: dict[str, list[dict]] = defaultdict(list)
    for f in kept:
        by_range[f["range"]].append(f)
    for r, lst in by_range.items():
        lst.sort(key=lambda c: c["n_vol"])
        half = len(lst) // 2
        for i, c in enumerate(lst):
            c["sub_bin"] = f"{r}_{'easy' if i < half else 'hard'}"

    # ── 6. Stratified train/val/test ──
    splits = stratified_split(kept, args.train_frac, args.val_frac, args.seed)

    # ── 7. Write manifest ──
    per_sb_total: dict[str, int] = defaultdict(int)
    per_sb_n_vol_median: dict[str, int] = {}
    for c in kept:
        per_sb_total[c["sub_bin"]] += 1
    # report the median n_vol per range (for sanity-checking easy/hard cut)
    for r, lst in by_range.items():
        if lst:
            n_vols = sorted(c["n_vol"] for c in lst)
            mid = n_vols[len(n_vols) // 2]
            per_sb_n_vol_median[r] = int(mid)

    n_buildings_sorted = sorted(c["n_buildings"] for c in kept)
    n_vol_sorted = sorted(c["n_vol"] for c in kept)

    manifest = {
        "splits": splits,
        "seed": args.seed,
        "physical_pt_dir": raw_dir,
        "n_scanned": len(feats),
        "n_dropped_anomaly": len(feats) - len(kept),
        "n_kept": len(kept),
        "counts": {k: len(v) for k, v in splits.items()},
        "per_sub_bin_total": {sb: per_sb_total.get(sb, 0)
                              for sb in SUB_BIN_ORDER},
        "range_n_vol_median_threshold": per_sb_n_vol_median,
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
        "anomaly_results": {
            cn: {"is_anomaly": r.get("is_anomaly", False),
                 "rule": r.get("rule", "normal")}
            for cn, r in anomaly_results.items()
        } if not args.no_anomaly_filter else {},
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n[manifest] split counts:")
    for k, v in manifest["counts"].items():
        print(f"  {k:6s}: {v}")
    print("\n[manifest] per sub-bin (kept):")
    for sb in SUB_BIN_ORDER:
        print(f"  {sb:16s}: {per_sb_total.get(sb, 0)}")
    print(f"\n[manifest] wrote {out_path}")
    print("\nNext step (anomaly already done — pass --skip-anomaly):")
    print("  nohup python preprocess/build_cache.py --config config.yaml "
          "--workers 80 --skip-anomaly > ~/scratch/preprocess.log 2>&1 &")
    print("  tail -f ~/scratch/preprocess.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
