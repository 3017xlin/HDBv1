#!/usr/bin/env python3
"""Evaluation entry point for HDB 3D urban wind CFD surrogate model.

Usage:
    torchrun --nproc_per_node=4 hdb/evaluate.py \\
        --config hdb/config.yaml --checkpoint CKPT_PATH [--swa]

Workflow:
  1. Initialize DDP (4 ranks)
  2. Load model from checkpoint (or SWA averaged weights)
  3. Load 50 test cases into pinned memory (split across ranks)
  4. Run test_full_inference: full-point inference with 4M-point decoder chunks
  5. Gather metrics to rank 0
  6. Generate visualizations for the median-error case
  7. Print summary table of per-field relative L2 metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HDB 3D wind CFD surrogate model on test set")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--swa", action="store_true", default=False,
        help="Treat checkpoint as SWA averaged weights")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results and visualizations "
             "(default: alongside checkpoint)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config with tilde expansion."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    path_keys = [
        ("data", "cache_dir"),
        ("data", "manifest_path"),
        ("data", "norm_stats_path"),
        ("data", "physical_pt_dir"),
        ("checkpoint", "save_dir"),
    ]
    for section, key in path_keys:
        if section in cfg and key in cfg[section]:
            cfg[section][key] = str(
                Path(cfg[section][key]).expanduser().resolve())
    return cfg


def load_norm_stats(path: str) -> dict:
    """Load norm_stats.json."""
    with open(path, "r") as f:
        return json.load(f)


def load_manifest(path: str) -> dict:
    """Load manifest.json."""
    with open(path, "r") as f:
        return json.load(f)


def print_summary_table(summary: dict[str, dict[str, float]]) -> None:
    """Print a formatted table of per-field relative L2 metrics."""
    header = f"{'Field':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("TEST SET EVALUATION RESULTS")
    print(sep)
    print(header)
    print(sep)

    for field_key in ["Ux_rel_L2", "Uy_rel_L2", "Uz_rel_L2",
                      "p_vol_rel_L2", "nut_rel_L2", "p_surf_rel_L2"]:
        if field_key in summary:
            s = summary[field_key]
            label = field_key.replace("_rel_L2", "")
            print(f"{label:<15} {s['mean']:>10.6f} {s['std']:>10.6f} "
                  f"{s['min']:>10.6f} {s['max']:>10.6f}")

    print(sep)

    # Overall mean across fields
    all_means = [summary[k]["mean"] for k in summary
                 if k.endswith("_rel_L2")]
    if all_means:
        overall = float(np.mean(all_means))
        print(f"{'OVERALL':<15} {overall:>10.6f}")
    print(sep + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # --- DDP init ---
    from training.ddp import init_ddp, cleanup_ddp
    rank, world, local = init_ddp()
    device = (torch.device("cuda", local) if torch.cuda.is_available()
              else torch.device("cpu"))

    if rank == 0:
        print(f"[evaluate] world_size={world}, device={device}", flush=True)
        print(f"[evaluate] checkpoint: {args.checkpoint}", flush=True)
        t0 = time.time()

    # --- Load model ---
    from models import HDB3DModel
    model = HDB3DModel(cfg).to(device)

    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Cast bf16 to fp32 for stable loading
    sd_fp = {k: v.to(torch.float32) if v.is_floating_point() else v
             for k, v in sd.items()}
    model.load_state_dict(sd_fp, strict=False)
    model.eval()

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[evaluate] model loaded ({n_params / 1e6:.1f}M params)",
              flush=True)

    # --- Load norm stats and manifest ---
    norm_stats = load_norm_stats(cfg["data"]["norm_stats_path"])
    manifest = load_manifest(cfg["data"]["manifest_path"])

    # --- Load test cases (sharded across ranks) ---
    test_ids = manifest["test_ids"]
    my_test_ids = sorted(test_ids[rank::world])

    if rank == 0:
        print(f"[evaluate] {len(test_ids)} test cases total, "
              f"{len(my_test_ids)} on this rank", flush=True)

    from dataset.loaders import load_cases_pinned
    num_workers = min(int(cfg["training"].get("num_workers", 30)), 8)
    my_case_pts = load_cases_pinned(
        cfg["data"]["cache_dir"], my_test_ids,
        num_workers=num_workers, rank=rank)

    # --- Full-point test inference ---
    from evaluation.test_eval import test_full_inference
    chunk_size = int(cfg.get("evaluation", {}).get("test_chunk_size", 4_000_000))

    result = test_full_inference(
        model, my_case_pts, norm_stats, device, cfg,
        chunk_size=chunk_size)

    # --- Rank 0: visualizations and summary ---
    if rank == 0 and result:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = str(Path(args.checkpoint).parent / "eval_results")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save per-case metrics as JSON
        per_case_scalar = {}
        for cid, m in result.get("per_case", {}).items():
            per_case_scalar[cid] = {k: v for k, v in m.items()
                                    if not k.startswith("_")}
        metrics_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "per_case": per_case_scalar,
                "summary": result.get("summary", {}),
                "median_case_id": result.get("median_case_id"),
            }, f, indent=2)
        print(f"[evaluate] metrics saved to {metrics_path}", flush=True)

        # Generate visualizations
        from evaluation.viz import generate_test_visualizations
        viz_paths = generate_test_visualizations(result, output_dir, cfg)
        for p in viz_paths:
            print(f"[evaluate] saved viz: {p}", flush=True)

        # Print summary
        print_summary_table(result.get("summary", {}))

        elapsed = time.time() - t0
        print(f"[evaluate] total time: {elapsed:.1f}s", flush=True)

    # --- Cleanup ---
    cleanup_ddp()


if __name__ == "__main__":
    main()
