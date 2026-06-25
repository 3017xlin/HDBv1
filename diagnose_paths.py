#!/usr/bin/env python3
"""Pre-flight diagnostic for the HDB preprocessing + training pipeline.

Run BEFORE invoking ``build_manifest.py`` and ``preprocess/build_cache.py``
to confirm the host has everything in place — physical PTs, writable
output dirs, free disk, all Python deps, and one sample PT that loads
cleanly with the expected schema.

This script does NOT need a GPU (a CPU node like
``qsub -I -l select=1:ncpus=80:mem=1600G`` is fine).

Usage::

    python diagnose_paths.py --config config.yaml
    python diagnose_paths.py --config config.yaml --check-gpu   # optional

Exits 0 on success, 1 on any failure, 2 on warnings only.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

REQUIRED_PT_KEYS = [
    "volume_fields", "surface_fields",
    "volume_pos", "surface_pos",
    "volume_sdf", "volume_sdf_grad",
    "stl_vertices", "stl_faces",
]

# Keys that must be present in a post-precompute cache PT (after Pass 2
# of preprocess/build_cache.py: z-score + precompute).
REQUIRED_CACHE_PT_KEYS = [
    # Leaf-level
    "leaf_centroid_norm", "leaf_stats", "leaf_sdf", "leaf_sdf_grad",
    "leaf_curvature_mean", "leaf_curvature_gauss", "leaf_pbd", "leaf_bin_id",
    # Neighbor indices
    "latent_point_top256", "latent_neighbor_top64",
    # Point-level
    "point_pos_norm", "point_sdf", "point_sdf_grad",
    "point_curvature_mean", "point_curvature_gauss", "point_is_surface",
    "point_leaf_id",
    # Targets
    "point_y_volume", "point_y_surface",
    "vol_reorder_idx", "surf_reorder_idx",
    # Precomputed (added by add_precomputed_fields)
    "encoder_pool", "vol_log_sample_weight",
    "rope_cos", "rope_sin", "bigbird_fixed",
    "n_query_vol", "n_query_surf",
    # Meta
    "case_name", "sub_bin", "L",
]

# Fields that should be ~N(0,1) after z-score (sanity check post-preprocess).
ZSCORED_FIELDS_TO_CHECK = [
    "point_y_volume", "point_y_surface",
    "leaf_stats", "leaf_sdf", "leaf_sdf_grad",
]

# norm_stats.json must contain mean+std for each accumulator name.
REQUIRED_NORM_STAT_NAMES = [
    "vol", "surf",
    "pt_sdf", "pt_sdfg", "pt_cm", "pt_cg",
    "leaf_sdf", "leaf_sdfg", "leaf_cm", "leaf_cg", "leaf_stats",
]

SUB_BIN_ORDER = [
    "0-19_easy", "0-19_hard", "20-39_easy", "20-39_hard",
    "40-59_easy", "40-59_hard", "60-79_easy", "60-79_hard",
    "80-123_easy", "80-123_hard",
]

REQUIRED_PACKAGES = [
    "torch", "numpy", "scipy", "sklearn", "trimesh",
    "rtree", "open3d", "tqdm", "yaml",
]

EXPECTED_CONFIG_KEYS = {
    "data": ["cache_dir", "manifest_path", "physical_pt_dir"],
    "sub_bin_L": [],
    "training": ["lr", "num_epochs"],
    "model": ["latent_dim", "num_layers"],
}


# ─── helpers ────────────────────────────────────────────────────────

class Banner:
    OK = "[ OK ]"
    WARN = "[WARN]"
    FAIL = "[FAIL]"


_warnings = 0
_failures = 0


def ok(msg: str) -> None:
    print(f"  {Banner.OK} {msg}", flush=True)


def warn(msg: str) -> None:
    global _warnings
    _warnings += 1
    print(f"  {Banner.WARN} {msg}", flush=True)


def fail(msg: str) -> None:
    global _failures
    _failures += 1
    print(f"  {Banner.FAIL} {msg}", flush=True)


def heading(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}", flush=True)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f}"


# ─── steps ──────────────────────────────────────────────────────────

def step_config(config_path: str) -> dict | None:
    heading("STEP 1 — Load config.yaml")
    if not os.path.exists(config_path):
        fail(f"config file not found: {config_path}")
        return None
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        fail(f"YAML parse error: {e}")
        return None

    ok(f"loaded {config_path}")
    for section, keys in EXPECTED_CONFIG_KEYS.items():
        if section not in cfg:
            fail(f"missing top-level section: {section}")
            continue
        for k in keys:
            if k not in cfg[section]:
                fail(f"missing config key: {section}.{k}")

    # Echo resolved paths
    data_cfg = cfg.get("data", {})
    for k in ("cache_dir", "manifest_path", "physical_pt_dir",
              "norm_stats_path"):
        raw = data_cfg.get(k)
        if raw is None:
            continue
        resolved = str(Path(os.path.expanduser(raw)).resolve())
        print(f"    data.{k}: {raw}  →  {resolved}")

    return cfg


def step_imports() -> None:
    heading("STEP 2 — Python deps")
    print(f"    python: {sys.version.split()[0]} ({sys.executable})")
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg:<12s} {ver}")
        except Exception as e:
            fail(f"{pkg:<12s} import failed: {e}")


def step_physical_pt_dir(cfg: dict, override: str | None) -> tuple[str, list[str]]:
    heading("STEP 3 — physical_pt_dir + sample PT")
    raw = override or cfg.get("data", {}).get("physical_pt_dir", "~/scratch/HDB")
    raw_dir = str(Path(os.path.expanduser(raw)).resolve())
    if override:
        print(f"    (override) physical_pt_dir = {raw_dir}")

    if not os.path.isdir(raw_dir):
        fail(f"directory does not exist: {raw_dir}")
        return raw_dir, []

    pt_paths = sorted(str(p) for p in Path(raw_dir).glob("*.pt"))
    print(f"    {raw_dir}: {len(pt_paths)} *.pt file(s)")
    if not pt_paths:
        fail("no .pt files found — preprocessing has nothing to consume.")
        return raw_dir, []

    sizes = [os.path.getsize(p) for p in pt_paths]
    total = sum(sizes)
    print(f"    total size: {fmt_bytes(total)} "
          f"(min={fmt_bytes(min(sizes))}, "
          f"max={fmt_bytes(max(sizes))}, "
          f"mean={fmt_bytes(total // len(sizes))})")

    ok(f"{len(pt_paths)} cases ready to preprocess")
    return raw_dir, pt_paths


def step_sample_pt(pt_paths: list[str]) -> None:
    heading("STEP 4 — Load one sample PT + key check")
    if not pt_paths:
        warn("no sample to load (physical_pt_dir was empty)")
        return
    try:
        import torch
    except ImportError as e:
        fail(f"torch unavailable: {e}")
        return

    sample = pt_paths[0]
    try:
        pt = torch.load(sample, map_location="cpu", weights_only=False)
    except Exception as e:
        fail(f"torch.load({sample}) raised {type(e).__name__}: {e}")
        return

    ok(f"loaded {os.path.basename(sample)}")
    if not isinstance(pt, dict):
        fail(f"sample PT is type={type(pt).__name__}, expected dict")
        return

    missing = [k for k in REQUIRED_PT_KEYS if k not in pt]
    if missing:
        fail(f"missing keys in PT: {missing}")
    else:
        ok(f"all {len(REQUIRED_PT_KEYS)} required keys present")

    if "volume_pos" in pt and "surface_pos" in pt:
        n_vol = pt["volume_pos"].shape[0]
        n_surf = pt["surface_pos"].shape[0]
        print(f"    N_vol={n_vol:,}  N_surf={n_surf:,}  "
              f"N_total={n_vol + n_surf:,}")

    if "stl_faces" in pt:
        faces = pt["stl_faces"]
        n_faces = faces.shape[0] if hasattr(faces, "shape") else len(faces)
        print(f"    stl_faces: {n_faces:,} triangles")


def step_writable_dirs(cfg: dict) -> str:
    heading("STEP 5 — Writable output dirs")
    data_cfg = cfg.get("data", {})
    cache_dir = os.path.expanduser(
        data_cfg.get("cache_dir", "~/scratch/cacheHDB"))
    manifest_path = os.path.expanduser(
        data_cfg.get("manifest_path", "~/scratch/manifest.json"))
    ckpt_dir = os.path.expanduser(
        cfg.get("checkpoint", {}).get("save_dir", "~/scratch/HDB_ckpt"))

    for label, p in (("cache_dir", cache_dir),
                     ("manifest_path", manifest_path),
                     ("checkpoint.save_dir", ckpt_dir)):
        target_dir = os.path.dirname(p) if p.endswith(".json") else p
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as e:
            fail(f"cannot mkdir {target_dir} for {label}: {e}")
            continue
        if os.access(target_dir, os.W_OK):
            ok(f"{label}: {target_dir}  (writable)")
        else:
            fail(f"{label}: {target_dir}  NOT writable")

    if os.path.exists(manifest_path):
        warn(f"manifest already exists: {manifest_path}  "
             f"(build_manifest.py will overwrite)")
    else:
        print(f"    manifest_path is absent — run build_manifest.py next.")

    return cache_dir


def step_disk_space(cache_dir: str, raw_total_bytes: int) -> None:
    heading("STEP 6 — Disk space estimate")
    try:
        usage = shutil.disk_usage(os.path.dirname(cache_dir) or "/")
    except Exception as e:
        warn(f"disk_usage failed: {e}")
        return
    free = usage.free
    # Cache is roughly the same size as raw + a fat intermediate dir.
    needed = max(int(raw_total_bytes * 2.5), 10 * 1024 ** 3)
    print(f"    free at {os.path.dirname(cache_dir)}: {fmt_bytes(free)}")
    print(f"    estimated need (cache + intermediate): {fmt_bytes(needed)}")
    if free < needed:
        warn(f"free space < estimated need")
    else:
        ok("free space looks sufficient")


# ─── post-preprocess steps ──────────────────────────────────────────

def step_manifest_structure(cfg: dict) -> dict | None:
    heading("POST STEP A — manifest.json structure")
    manifest_path = os.path.expanduser(
        cfg.get("data", {}).get("manifest_path", "~/scratch/manifest.json"))
    manifest_path = str(Path(manifest_path).resolve())
    if not os.path.exists(manifest_path):
        fail(f"manifest does not exist: {manifest_path}  "
             f"(run build_manifest.py first)")
        return None
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        fail(f"manifest JSON parse error: {e}")
        return None

    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        fail(f"manifest.splits is not a dict: {type(splits).__name__}")
        return None

    for split in ("train", "val", "test"):
        if split not in splits:
            fail(f"manifest.splits.{split} missing")
            continue
        entries = splits[split]
        if not isinstance(entries, list):
            fail(f"manifest.splits.{split} is not a list "
                 f"({type(entries).__name__})")
            continue
        # Check first entry shape — catches "list of dicts but missing
        # case_name" (would break the training loop) and "list of bare
        # strings" (old format) alike.
        if entries:
            e = entries[0]
            if isinstance(e, dict):
                if "case_name" not in e:
                    fail(f"manifest.splits.{split}[0] is dict but has no "
                         f"'case_name' field; keys = {sorted(e.keys())}")
                else:
                    ok(f"{split}: {len(entries)} entries (dict format)")
            elif isinstance(e, str):
                ok(f"{split}: {len(entries)} entries (string format)")
            else:
                fail(f"manifest.splits.{split}[0] has unexpected type "
                     f"{type(e).__name__}")
        else:
            warn(f"{split}: empty list")

    # Cross-check physical_pt_dir consistency.
    cfg_phys = os.path.expanduser(
        cfg.get("data", {}).get("physical_pt_dir", ""))
    cfg_phys = str(Path(cfg_phys).resolve()) if cfg_phys else ""
    man_phys = manifest.get("physical_pt_dir", "")
    if cfg_phys and man_phys and cfg_phys != man_phys:
        warn(f"manifest.physical_pt_dir={man_phys} differs from "
             f"cfg.data.physical_pt_dir={cfg_phys}")
    print(f"    manifest path: {manifest_path}")
    print(f"    physical_pt_dir (manifest): {man_phys}")
    return manifest


def step_cache_files_exist(cfg: dict, manifest: dict) -> list[tuple[str, str]]:
    heading("POST STEP B — cache PTs exist for every manifest entry")
    cache_dir = os.path.expanduser(
        cfg.get("data", {}).get("cache_dir", "~/scratch/cacheHDB"))
    cache_dir = str(Path(cache_dir).resolve())
    if not os.path.isdir(cache_dir):
        fail(f"cache_dir does not exist: {cache_dir}")
        return []

    samples: list[tuple[str, str]] = []   # (split, case_name) — one per split
    missing: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split, entries in manifest.get("splits", {}).items():
        if split not in missing:
            missing[split] = []
        for e in entries:
            cn = e["case_name"] if isinstance(e, dict) else str(e)
            p = os.path.join(cache_dir, f"{cn}.pt")
            if not os.path.exists(p):
                missing[split].append(cn)
        present = len(entries) - len(missing.get(split, []))
        total = len(entries)
        if missing.get(split):
            fail(f"{split}: {present}/{total} cache PTs present "
                 f"(missing {len(missing[split])})")
            for cn in missing[split][:5]:
                print(f"      missing: {cn}.pt")
            if len(missing[split]) > 5:
                print(f"      ... and {len(missing[split]) - 5} more")
        else:
            ok(f"{split}: all {total} cache PTs present")
            if entries:
                e0 = entries[0]
                cn = e0["case_name"] if isinstance(e0, dict) else str(e0)
                samples.append((split, cn))
    print(f"    cache_dir: {cache_dir}")
    return samples


def step_cache_pt_schema(cfg: dict, samples: list[tuple[str, str]]) -> None:
    heading("POST STEP C — cache PT schema + z-score sanity")
    if not samples:
        warn("no cache PT samples to load (cache empty?)")
        return
    try:
        import torch
        import numpy as np
    except ImportError as e:
        fail(f"torch / numpy unavailable: {e}")
        return

    cache_dir = os.path.expanduser(
        cfg.get("data", {}).get("cache_dir", "~/scratch/cacheHDB"))
    cache_dir = str(Path(cache_dir).resolve())

    for split, cn in samples:
        path = os.path.join(cache_dir, f"{cn}.pt")
        try:
            pt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            fail(f"{split} sample '{cn}.pt' load failed: {e}")
            continue
        missing = [k for k in REQUIRED_CACHE_PT_KEYS if k not in pt]
        if missing:
            fail(f"{split} '{cn}.pt' missing keys: {missing}")
        else:
            ok(f"{split} '{cn}.pt' has all {len(REQUIRED_CACHE_PT_KEYS)} "
               f"required keys")

        # Reject the leftover pre-zscore key — it should be deleted in
        # Pass 2.
        if "original_sdf_volume" in pt:
            warn(f"{split} '{cn}.pt' still has 'original_sdf_volume' "
                 f"(Pass 2 should have removed it)")

        # Z-score sanity: each field should be roughly ~N(0,1).
        for k in ZSCORED_FIELDS_TO_CHECK:
            if k not in pt:
                continue
            v = pt[k]
            if hasattr(v, "numpy"):
                v = v.numpy()
            v = np.asarray(v, dtype=np.float64)
            if v.size == 0:
                continue
            m = float(v.mean())
            s = float(v.std())
            tag = ""
            if abs(m) > 2.0 or s < 0.1 or s > 5.0:
                tag = f"  ⚠ outside expected ~N(0,1) range"
                _warnings_inc()
            print(f"    {split} '{cn}' {k:24s}: mean={m:+.3f}  std={s:+.3f}"
                  f"{tag}")


def _warnings_inc():
    """Bump warnings counter without printing a separate WARN line."""
    global _warnings
    _warnings += 1


def step_norm_stats(cfg: dict) -> None:
    heading("POST STEP D — norm_stats.json + rope_scales")
    cache_dir = os.path.expanduser(
        cfg.get("data", {}).get("cache_dir", "~/scratch/cacheHDB"))
    ns_path = os.path.expanduser(
        cfg.get("data", {}).get(
            "norm_stats_path",
            os.path.join(cache_dir, "norm_stats.json")))
    ns_path = str(Path(ns_path).resolve())
    if not os.path.exists(ns_path):
        fail(f"norm_stats.json not found: {ns_path}")
        return
    try:
        with open(ns_path) as f:
            ns = json.load(f)
    except Exception as e:
        fail(f"norm_stats.json parse error: {e}")
        return

    missing = []
    for name in REQUIRED_NORM_STAT_NAMES:
        if f"{name}_mean" not in ns or f"{name}_std" not in ns:
            missing.append(name)
    if missing:
        fail(f"norm_stats missing mean/std for: {missing}")
    else:
        ok(f"all {len(REQUIRED_NORM_STAT_NAMES)} accumulator stats present")

    rs = ns.get("rope_scales")
    if not isinstance(rs, dict):
        fail("norm_stats.rope_scales missing or not a dict")
    else:
        missing_sb = [sb for sb in SUB_BIN_ORDER if sb not in rs]
        if missing_sb:
            fail(f"rope_scales missing sub-bins: {missing_sb}")
        else:
            ok(f"rope_scales present for all 10 sub-bins")
    print(f"    norm_stats path: {ns_path}")


def step_gpu() -> None:
    heading("STEP 7 — GPU check (optional)")
    try:
        import torch
    except ImportError:
        warn("torch not importable, skipping")
        return
    if not torch.cuda.is_available():
        warn("torch.cuda.is_available() == False  "
             "(fine on a CPU node — required on the training node)")
        return
    n = torch.cuda.device_count()
    ok(f"{n} CUDA device(s)")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"    [{i}] {props.name}  "
              f"{props.total_memory / 1e9:.1f} GB")


# ─── main ───────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--stage", choices=["pre", "post", "all"], default="pre",
                   help="`pre`: checks needed BEFORE build_manifest/build_cache "
                        "(config, deps, raw PTs, dirs, disk). "
                        "`post`: checks needed AFTER build_cache, BEFORE "
                        "training (manifest format, cache PTs, schema, "
                        "norm_stats, rope_scales). "
                        "`all`: run both. Default: pre.")
    p.add_argument("--physical-pt-dir", default=None,
                   help="Override cfg.data.physical_pt_dir for this check.")
    p.add_argument("--check-gpu", action="store_true",
                   help="Also run a CUDA visibility check.")
    args = p.parse_args()

    cfg = step_config(args.config)
    if cfg is None:
        print("\nABORT: cannot continue without a valid config.")
        return 1

    if args.stage in ("pre", "all"):
        step_imports()
        raw_dir, pt_paths = step_physical_pt_dir(cfg, args.physical_pt_dir)
        step_sample_pt(pt_paths)
        cache_dir = step_writable_dirs(cfg)
        raw_total = sum(os.path.getsize(p) for p in pt_paths) if pt_paths else 0
        step_disk_space(cache_dir, raw_total)

    if args.stage in ("post", "all"):
        manifest = step_manifest_structure(cfg)
        if manifest is not None:
            samples = step_cache_files_exist(cfg, manifest)
            step_cache_pt_schema(cfg, samples)
            step_norm_stats(cfg)

    if args.check_gpu:
        step_gpu()

    heading("SUMMARY")
    print(f"  failures: {_failures}")
    print(f"  warnings: {_warnings}")
    if _failures:
        if args.stage == "pre":
            print("\n  DIAGNOSTIC FAILED — fix the [FAIL] items above before "
                  "running build_manifest.py.")
        else:
            print("\n  DIAGNOSTIC FAILED — fix the [FAIL] items above before "
                  "starting training.")
        return 1
    if _warnings:
        print("\n  diagnostic passed with warnings.")
        return 2
    if args.stage == "pre":
        print("\n  diagnostic passed cleanly. "
              "Next: python build_manifest.py --config config.yaml")
    elif args.stage == "post":
        print("\n  diagnostic passed cleanly. "
              "Next: python debug_train_dryrun.py --config config.yaml")
    else:
        print("\n  diagnostic passed cleanly. Ready to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
