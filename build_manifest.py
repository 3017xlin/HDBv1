#!/usr/bin/env python3
"""Build manifest.json for HDB preprocessing pipeline.

This is the file that ``preprocess/build_cache.py`` consumes via
``cfg['data']['manifest_path']`` (default ``~/scratch/manifest.json``).

It scans ``cfg['data']['physical_pt_dir']`` (default ``~/scratch/HDB``),
classifies every ``*.pt`` case into one of the 10 sub-bins defined in
``cfg['sub_bin_L']`` (``0-19_easy`` … ``80-123_hard``), and writes a
stratified 70 / 15 / 15 train / val / test split.

Classification rules
--------------------
* ``--rule size``       — file size in bytes (FAST: no PT loaded).
* ``--rule ntotal``     — ``volume_pos.shape[0] + surface_pos.shape[0]``
                          (loads each PT once, parallel).
* ``--rule ncomponents``— count connected mesh components from
                          ``stl_faces`` via union-find = number of
                          buildings.  Slowest but most faithful to the
                          "0-19 … 80-123" naming.

For ``size`` / ``ntotal`` the 5 numeric ranges are assigned by
**quintile** of the chosen metric across all cases; ``easy`` / ``hard``
is the within-range median split.

Output JSON layout (consumed by ``preprocess/build_cache.py`` and
``training/loop.py``)::

    {
      "splits": {
        "train": [{"case_name": "...", "sub_bin": "0-19_easy"}, ...],
        "val":   [...],
        "test":  [...]
      },
      "rule":   "size",
      "seed":   42,
      "counts": {"train": 560, "val": 120, "test": 120},
      "per_sub_bin": {"0-19_easy": 80, ...}
    }

Usage::

    python build_manifest.py --config config.yaml          # default: size rule
    python build_manifest.py --config config.yaml --rule ntotal --workers 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import yaml

SUB_BIN_RANGES = ["0-19", "20-39", "40-59", "60-79", "80-123"]
DIFFICULTIES = ["easy", "hard"]
SUB_BIN_ORDER = [f"{r}_{d}" for r in SUB_BIN_RANGES for d in DIFFICULTIES]


# ─── classification metric helpers ───────────────────────────────────

def metric_size(pt_path: str) -> int:
    return os.path.getsize(pt_path)


def metric_ntotal(pt_path: str) -> int:
    """N_vol + N_surf from a physical PT (loaded once on CPU)."""
    import torch  # local import so --rule size doesn't pay the cost
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    vol_pos = pt["volume_pos"]
    surf_pos = pt["surface_pos"]
    n_vol = vol_pos.shape[0]
    n_surf = surf_pos.shape[0]
    return int(n_vol + n_surf)


def metric_ncomponents(pt_path: str) -> int:
    """Number of disconnected mesh components in stl_faces = buildings."""
    import numpy as np
    import torch
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    faces = pt["stl_faces"]
    if hasattr(faces, "numpy"):
        faces = faces.numpy()
    faces = np.asarray(faces).astype(np.int64)
    n_vert = int(faces.max()) + 1

    # Union-find on vertices via face edges.
    parent = np.arange(n_vert, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[ra] = rb
            ra = rb
        if ra != rc:
            parent[ra] = rc

    roots = {find(v) for v in range(n_vert)}
    return len(roots)


_METRIC_FNS = {
    "size": metric_size,
    "ntotal": metric_ntotal,
    "ncomponents": metric_ncomponents,
}


def _metric_worker(args):
    rule, pt_path = args
    fn = _METRIC_FNS[rule]
    return os.path.basename(pt_path)[:-3], fn(pt_path)  # strip .pt


# ─── range mapping ──────────────────────────────────────────────────

def range_from_ncomponents(n: int) -> str:
    if n <= 19:
        return "0-19"
    if n <= 39:
        return "20-39"
    if n <= 59:
        return "40-59"
    if n <= 79:
        return "60-79"
    return "80-123"


def assign_quintile_range(metrics: dict[str, float]) -> dict[str, str]:
    """Sort cases by metric → split into 5 equal-sized quintiles."""
    items = sorted(metrics.items(), key=lambda kv: kv[1])
    n = len(items)
    out: dict[str, str] = {}
    for i, (name, _) in enumerate(items):
        q = min(4, int(5 * i / n))
        out[name] = SUB_BIN_RANGES[q]
    return out


def assign_difficulty(
    metrics: dict[str, float], range_of: dict[str, str]
) -> dict[str, str]:
    """Within each range, smaller-metric half = easy, larger half = hard."""
    by_range: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for name, m in metrics.items():
        by_range[range_of[name]].append((name, m))

    out: dict[str, str] = {}
    for r, lst in by_range.items():
        lst.sort(key=lambda kv: kv[1])
        half = len(lst) // 2
        for i, (name, _) in enumerate(lst):
            out[name] = "easy" if i < half else "hard"
    return out


# ─── splits ─────────────────────────────────────────────────────────

def stratified_split(
    sub_bin_of: dict[str, str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[dict]]:
    import random
    rng = random.Random(seed)
    by_sb: dict[str, list[str]] = defaultdict(list)
    for name, sb in sub_bin_of.items():
        by_sb[sb].append(name)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for sb, names in by_sb.items():
        names = sorted(names)            # determinism before shuffle
        rng.shuffle(names)
        n = len(names)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for i, cn in enumerate(names):
            if i < n_train:
                bucket = "train"
            elif i < n_train + n_val:
                bucket = "val"
            else:
                bucket = "test"
            splits[bucket].append({"case_name": cn, "sub_bin": sb})

    for k in splits:
        splits[k].sort(key=lambda d: d["case_name"])
    return splits


# ─── main ───────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--rule", choices=list(_METRIC_FNS), default="size",
                   help="Classification metric (default: size).")
    p.add_argument("--workers", type=int, default=40,
                   help="Parallel workers when --rule loads PTs (default 40).")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--physical-pt-dir", default=None,
                   help="Override cfg.data.physical_pt_dir.")
    p.add_argument("--out", default=None,
                   help="Override cfg.data.manifest_path output location.")
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

    print(f"[manifest] {len(pt_paths)} cases found in {raw_dir}")
    print(f"[manifest] rule={args.rule}  workers={args.workers}")
    print(f"[manifest] output: {out_path}")

    # ── compute metric for every case ──
    metrics: dict[str, float] = {}
    work = [(args.rule, p) for p in pt_paths]
    if args.rule == "size":
        for r, pp in work:
            name, m = _metric_worker((r, pp))
            metrics[name] = m
    else:
        with Pool(args.workers) as pool:
            for i, (name, m) in enumerate(pool.imap_unordered(_metric_worker, work)):
                metrics[name] = m
                if (i + 1) % 50 == 0 or (i + 1) == len(work):
                    print(f"  [{args.rule}] {i + 1}/{len(work)} cases processed",
                          flush=True)

    # ── range assignment ──
    if args.rule == "ncomponents":
        range_of = {name: range_from_ncomponents(int(v))
                    for name, v in metrics.items()}
    else:
        range_of = assign_quintile_range(metrics)

    # ── difficulty assignment (size-within-range proxy) ──
    if args.rule == "ncomponents":
        # Reuse file-size as the easy/hard tie-breaker so this rule still
        # produces both halves.
        size_metric = {name: metric_size(os.path.join(raw_dir, name + ".pt"))
                       for name in metrics}
        diff_of = assign_difficulty(size_metric, range_of)
    else:
        diff_of = assign_difficulty(metrics, range_of)

    sub_bin_of = {name: f"{range_of[name]}_{diff_of[name]}"
                  for name in metrics}

    # ── stratified split ──
    splits = stratified_split(sub_bin_of, args.train_frac, args.val_frac,
                              args.seed)

    per_sb_counts: dict[str, int] = defaultdict(int)
    for sb in sub_bin_of.values():
        per_sb_counts[sb] += 1

    manifest = {
        "splits": splits,
        "rule": args.rule,
        "seed": args.seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "per_sub_bin": {sb: per_sb_counts.get(sb, 0) for sb in SUB_BIN_ORDER},
        "physical_pt_dir": raw_dir,
        "n_cases": len(metrics),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n[manifest] split counts:")
    for k, v in manifest["counts"].items():
        print(f"  {k:6s}: {v}")
    print("\n[manifest] per sub-bin (all splits):")
    for sb in SUB_BIN_ORDER:
        print(f"  {sb:16s}: {per_sb_counts.get(sb, 0)}")
    print(f"\n[manifest] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
