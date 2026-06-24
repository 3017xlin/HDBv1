#!/usr/bin/env python3
"""HDB Diagnostic Script 2: Single-Case Training Dry-Run.

Runs exactly 1 forward + backward pass on 1 case WITHOUT torchrun / DDP,
on a single GPU. Prints exhaustive shape/dtype/value diagnostics at every
stage so you can pinpoint exactly where things break.

Checks:
  1. Config loading
  2. Model instantiation + parameter summary
  3. Load 1 cache PT and prepare_one_case
  4. Build BigBird BlockMask on GPU
  5. H2D transfer
  6. GPU IDW computation
  7. Forward pass (bf16 autocast)
  8. Loss computation
  9. Backward pass + gradient stats
  10. Optimizer step
  11. Memory summary

Usage:
    python debug_train_dryrun.py --config config.yaml [--case-index 0]

Runs on GPU 0. No torchrun needed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _fmt_tensor(t, name=""):
    if isinstance(t, torch.Tensor):
        mn = t.float().min().item()
        mx = t.float().max().item()
        has_nan = bool(torch.isnan(t.float()).any())
        has_inf = bool(torch.isinf(t.float()).any())
        return (f"{name:35s} shape={str(list(t.shape)):25s} dtype={str(t.dtype):15s} "
                f"device={str(t.device):10s} min={mn:+.4e} max={mx:+.4e}"
                f"{' NaN!' if has_nan else ''}{' Inf!' if has_inf else ''}")
    return f"{name:35s} = {t}"


def banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="HDB single-case training dry-run")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--case-index", type=int, default=0,
                        help="Index into train split to pick test case (default: 0)")
    parser.add_argument("--n-query", type=int, default=50000,
                        help="Number of query points (reduced for speed, default 50000)")
    args = parser.parse_args()

    # ================================================================ CONFIG
    banner("STEP 1: Load Config")
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    path_keys = [("data", "cache_dir"), ("data", "manifest_path"),
                 ("data", "norm_stats_path"), ("data", "physical_pt_dir"),
                 ("checkpoint", "save_dir")]
    for section, key in path_keys:
        if section in cfg and key in cfg[section]:
            cfg[section][key] = str(Path(cfg[section][key]).expanduser().resolve())

    print(f"  cache_dir:     {cfg['data']['cache_dir']}")
    print(f"  manifest_path: {cfg['data']['manifest_path']}")
    print(f"  model.latent_dim={cfg['model']['latent_dim']}, "
          f"num_layers={cfg['model']['num_layers']}, "
          f"num_heads={cfg['model']['num_heads']}")
    print(f"  PASS")

    # ================================================================ DEVICE
    banner("STEP 2: GPU Check")
    if not torch.cuda.is_available():
        print("  FAIL: No CUDA GPU available. This script requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  PASS")

    # ================================================================ MODEL
    banner("STEP 3: Model Instantiation")
    try:
        from hdb.models import HDB3DModel
        model = HDB3DModel(cfg).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Model dtype:          {next(model.parameters()).dtype}")

        for name, module in [("encoder", model.encoder),
                             ("vit", model.vit),
                             ("decoder", model.decoder)]:
            n = sum(p.numel() for p in module.parameters())
            print(f"    {name:12s}: {n:>12,} params")

        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ DATA
    banner("STEP 4: Load Manifest + 1 Case")
    import json
    cache_dir = cfg["data"]["cache_dir"]
    manifest_path = cfg["data"]["manifest_path"]

    manifest_candidates = [
        manifest_path,
        os.path.join(cache_dir, "manifest.json"),
    ]
    manifest = None
    for mp in manifest_candidates:
        if os.path.exists(mp):
            with open(mp) as f:
                manifest = json.load(f)
            print(f"  Loaded manifest from: {mp}")
            break

    if manifest is None:
        print(f"  FAIL: No manifest found at {manifest_candidates}")
        sys.exit(1)

    train_ids = manifest["splits"]["train"]
    idx = min(args.case_index, len(train_ids) - 1)
    case_id = train_ids[idx]
    if isinstance(case_id, dict):
        case_id = case_id["case_name"]
    case_id = str(case_id)

    pt_path = os.path.join(cache_dir, f"{case_id}.pt")
    print(f"  Using case: '{case_id}' (index {idx})")
    print(f"  PT path: {pt_path}")

    if not os.path.exists(pt_path):
        print(f"  FAIL: {pt_path} does not exist")
        sys.exit(1)

    try:
        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
        L = int(pt["L"])
        sub_bin = pt.get("sub_bin", "unknown")
        print(f"  Loaded: L={L}, sub_bin='{sub_bin}'")
        print(f"  Keys in PT: {sorted(pt.keys())}")

        for k in ["leaf_centroid_norm", "leaf_stats", "leaf_pbd",
                   "encoder_pool", "point_pos_norm", "point_y_volume",
                   "point_y_surface", "vol_reorder_idx", "surf_reorder_idx",
                   "bigbird_fixed", "rope_cos", "rope_sin"]:
            if k in pt:
                v = pt[k]
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    shape = tuple(v.shape) if isinstance(v, torch.Tensor) else v.shape
                    dt = str(v.dtype)
                    print(f"    {k:30s}  shape={str(shape):25s}  dtype={dt}")

        N_vol = pt["vol_reorder_idx"].shape[0] if isinstance(pt["vol_reorder_idx"], (torch.Tensor, np.ndarray)) else len(pt["vol_reorder_idx"])
        N_surf = pt["surf_reorder_idx"].shape[0] if isinstance(pt["surf_reorder_idx"], (torch.Tensor, np.ndarray)) else len(pt["surf_reorder_idx"])
        print(f"  N_vol={N_vol}, N_surf={N_surf}, N_total={N_vol + N_surf}")
        print(f"  n_query_vol={pt.get('n_query_vol', '?')}, n_query_surf={pt.get('n_query_surf', '?')}")
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL loading PT: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ PREPARE
    banner("STEP 5: prepare_one_case (CPU)")
    try:
        from hdb.dataset.prefetcher import prepare_one_case, stack_batch
        pt["_case_id"] = 0

        n_query = args.n_query
        print(f"  Using n_query={n_query} (reduced for debug speed)")

        t0 = time.time()
        item = prepare_one_case(pt, case_id=0, epoch=0,
                                encoder_k=32, n_query=n_query)
        dt = time.time() - t0
        print(f"  prepare_one_case took {dt:.2f}s")

        print(f"\n  Prepared item fields:")
        for k in sorted(item.keys()):
            v = item[k]
            if isinstance(v, torch.Tensor):
                print(f"    {_fmt_tensor(v, k)}")
            else:
                print(f"    {k:35s} = {v}")

        batch_cpu = stack_batch([item])
        print(f"\n  stack_batch (B=1) done. Batch keys: {len(batch_cpu)}")
        n_qv = item["n_query_vol"]
        batch_cpu["n_query_vol"] = n_qv
        batch_cpu["L"] = L
        print(f"  n_query_vol = {n_qv}")
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ BIGBIRD
    banner("STEP 6: Build BigBird BlockMask (GPU)")
    try:
        from hdb.models.bigbird import build_block_mask_direct
        R = 16

        t0 = time.time()
        flex_mask = build_block_mask_direct(
            batch_cpu["bigbird_key_idx"], L=L, R=R, device=device)
        dt = time.time() - t0
        print(f"  BlockMask built in {dt:.2f}s")
        print(f"  flex_mask type: {type(flex_mask)}")
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ H2D
    banner("STEP 7: Host-to-Device Transfer")
    try:
        batch = {}
        for k, v in batch_cpu.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            else:
                batch[k] = v
        torch.cuda.synchronize()

        print(f"  Transferred {len([k for k,v in batch.items() if isinstance(v, torch.Tensor)])} tensors to GPU")
        for k, v in sorted(batch.items()):
            if isinstance(v, torch.Tensor):
                print(f"    {_fmt_tensor(v, k)}")
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ IDW
    banner("STEP 8: GPU IDW Computation")
    try:
        from hdb.models.idw import gpu_idw
        t0 = time.time()
        idw_idx, idw_w = gpu_idw(
            batch["query_pos_norm"],
            batch["leaf_centroid_norm"],
            batch["latent_neighbor_top64"],
            batch["query_leaf_id"],
            idw_k=8)
        torch.cuda.synchronize()
        dt = time.time() - t0

        print(f"  gpu_idw took {dt:.3f}s")
        print(f"    {_fmt_tensor(idw_idx, 'idw_indices')}")
        print(f"    {_fmt_tensor(idw_w, 'idw_weights')}")

        if (idw_w.float().sum(dim=-1) - 1.0).abs().max().item() > 0.01:
            print(f"  WARN: IDW weights don't sum to 1.0 (max deviation: "
                  f"{(idw_w.float().sum(dim=-1) - 1.0).abs().max().item():.4f})")
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ FORWARD
    banner("STEP 9: Forward Pass (bf16 autocast)")
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        t0 = time.time()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_vol, pred_surf = model(
                leaf_centroid_norm=batch["leaf_centroid_norm"],
                leaf_stats=batch["leaf_stats"],
                leaf_sdf=batch["leaf_sdf"],
                leaf_sdf_grad=batch["leaf_sdf_grad"],
                leaf_curvature_mean=batch["leaf_curvature_mean"],
                leaf_curvature_gauss=batch["leaf_curvature_gauss"],
                leaf_pbd=batch["leaf_pbd"],
                transient1=batch["transient1"],
                query_pos_norm=batch["query_pos_norm"],
                query_sdf=batch["query_sdf"],
                query_sdf_grad=batch["query_sdf_grad"],
                idw_indices=idw_idx,
                idw_weights=idw_w,
                rope_cos=batch["rope_cos"],
                rope_sin=batch["rope_sin"],
                flex_mask=flex_mask,
                n_query_vol=n_qv,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0

        print(f"  Forward pass took {dt:.2f}s")
        print(f"    {_fmt_tensor(pred_vol, 'pred_vol')}")
        print(f"    {_fmt_tensor(pred_surf, 'pred_surf')}")

        target_vol = batch["query_target_volume"][:, :n_qv]
        target_surf = batch["query_target_surface"]
        print(f"    {_fmt_tensor(target_vol, 'target_vol')}")
        print(f"    {_fmt_tensor(target_surf, 'target_surf')}")

        if pred_vol.shape != target_vol.shape:
            print(f"  FAIL: pred_vol shape {pred_vol.shape} != target {target_vol.shape}")
            sys.exit(1)
        if pred_surf.shape != target_surf.shape:
            print(f"  FAIL: pred_surf shape {pred_surf.shape} != target {target_surf.shape}")
            sys.exit(1)

        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ LOSS
    banner("STEP 10: Loss Computation")
    try:
        import torch.nn.functional as F
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_vol = F.mse_loss(pred_vol, target_vol)
            loss_surf = F.mse_loss(pred_surf, target_surf)
            loss = loss_vol + loss_surf

        print(f"  loss_vol  = {loss_vol.item():.6f}")
        print(f"  loss_surf = {loss_surf.item():.6f}")
        print(f"  loss_total= {loss.item():.6f}")

        if torch.isnan(loss):
            print(f"  FAIL: Loss is NaN!")
            sys.exit(1)
        if torch.isinf(loss):
            print(f"  FAIL: Loss is Inf!")
            sys.exit(1)
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ BACKWARD
    banner("STEP 11: Backward Pass + Gradient Stats")
    try:
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0

        print(f"  Backward took {dt:.2f}s")

        total_grad_norm = 0.0
        nan_grads = []
        inf_grads = []
        zero_grads = []

        for name, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad.float()
                gn = g.norm().item()
                total_grad_norm += gn ** 2
                if torch.isnan(g).any():
                    nan_grads.append(name)
                if torch.isinf(g).any():
                    inf_grads.append(name)
                if gn == 0.0:
                    zero_grads.append(name)
            else:
                zero_grads.append(f"{name} (no grad)")

        total_grad_norm = total_grad_norm ** 0.5
        print(f"  Total gradient L2 norm: {total_grad_norm:.6f}")

        if nan_grads:
            print(f"  FAIL: NaN gradients in {len(nan_grads)} params:")
            for n in nan_grads[:5]:
                print(f"    - {n}")
            sys.exit(1)

        if inf_grads:
            print(f"  FAIL: Inf gradients in {len(inf_grads)} params:")
            for n in inf_grads[:5]:
                print(f"    - {n}")
            sys.exit(1)

        if zero_grads:
            print(f"  WARN: {len(zero_grads)} params have zero/no gradient:")
            for n in zero_grads[:10]:
                print(f"    - {n}")

        print(f"\n  Per-module gradient norms:")
        for mod_name in ["encoder", "vit", "decoder"]:
            mod = getattr(model, mod_name)
            gn = sum(p.grad.float().norm().item() ** 2
                     for p in mod.parameters() if p.grad is not None) ** 0.5
            print(f"    {mod_name:12s}: {gn:.6f}")

        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ OPTIM STEP
    banner("STEP 12: Optimizer Step")
    try:
        import math
        base_lr = float(cfg["training"]["lr"])
        wd = float(cfg["training"]["weight_decay"])
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                weight_decay=wd, fused=True)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(cfg["training"].get("max_grad_norm", 1.0)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"  Optimizer step completed successfully")
        print(f"  lr={base_lr:.1e}, weight_decay={wd}")

        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f"  FAIL: NaN in param '{name}' after optimizer step!")
                sys.exit(1)
            if torch.isinf(p).any():
                print(f"  FAIL: Inf in param '{name}' after optimizer step!")
                sys.exit(1)

        print(f"  Post-step param check: no NaN/Inf")
        print(f"  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ================================================================ MEMORY
    banner("STEP 13: Memory Summary")
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    print(f"  Peak GPU memory allocated: {peak_mem:.2f} GB")
    print(f"  GPU memory reserved:       {reserved:.2f} GB")

    try:
        import psutil
        proc = psutil.Process()
        rss = proc.memory_info().rss / 1e9
        print(f"  CPU RSS: {rss:.2f} GB")
    except ImportError:
        print(f"  (psutil not available for CPU memory)")

    # ================================================================ SUMMARY
    banner("FINAL SUMMARY")
    print(f"  Case:     '{case_id}' (sub_bin='{sub_bin}', L={L})")
    print(f"  n_query:  {n_query} (vol={n_qv})")
    print(f"  Forward:  pred_vol={list(pred_vol.shape)}, pred_surf={list(pred_surf.shape)}")
    print(f"  Loss:     total={loss.item():.6f} (vol={loss_vol.item():.6f}, surf={loss_surf.item():.6f})")
    print(f"  Grad norm: {total_grad_norm:.6f}")
    print(f"  GPU peak:  {peak_mem:.2f} GB")
    print()
    print(f"  ALL 13 STEPS PASSED")
    print(f"  The training pipeline should work end-to-end.")
    print(f"  You can now run full training with torchrun.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
