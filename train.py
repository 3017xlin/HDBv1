#!/usr/bin/env python3
"""Training entry point for HDB 3D urban wind CFD surrogate model.

Usage:
    torchrun --nproc_per_node=4 hdb/train.py --config hdb/config.yaml [--resume CKPT_PATH]

This script parses CLI arguments, loads the YAML config, and delegates to the
training loop (hdb.training.loop.main). It is intentionally thin so that
training logic lives in a testable module.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HDB 3D wind CFD surrogate model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config.yaml")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config with tilde expansion on path fields."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Expand ~ in path-like fields
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Lazy import to keep startup fast and avoid loading torch before
    # torchrun has set up environment variables.
    from hdb.training.loop import main as training_main

    training_main(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
