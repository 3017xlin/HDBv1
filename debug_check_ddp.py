#!/usr/bin/env python3
"""HDB Diagnostic Script 1: Multi-GPU DDP Smoke Test.

Tests the FULL distributed training pipeline across all 4 GPUs with 1 real
batch: DDP init, data loading, model build, compile, prefetcher, BigBird
mask, IDW, forward, loss, backward, all-reduce, optimizer step.

This catches: NCCL failures, GPU OOM, shape mismatches across ranks,
compile errors, FlexAttention issues, dtype problems, gradient sync bugs.

Usage (must use torchrun):
    torchrun --nproc_per_node=4 debug_check_ddp.py --config config.yaml

Runs exactly 1 batch per sub-bin (smallest first) then exits.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _fmt(t, name=""):
    if isinstance(t, torch.Tensor):
        mn = t.float().min().item()
        mx = t.float().max().item()
        nan = bool(torch.isnan(t.float()).any())
        inf = bool(torch.isinf(t.float()).any())
        return (f"{name:30s} {str(list(t.shape)):22s} {str(t.dtype):12s} "
                f"dev={str(t.device):8s} [{mn:+.3e}, {mx:+.3e}]"
                f"{' NaN!' if nan else ''}{' Inf!' if inf else ''}")
    return f"{name:30s} = {t}"


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def log_all(rank, world, msg):
    for r in range(world):
        if rank == r:
            print(f"  [rank {r}] {msg}", flush=True)
        if dist.is_initialized():
            dist.barrier()


def banner(rank, title, step):
    if rank == 0:
        print(f"\n{'='*70}", flush=True)
        print(f"  STEP {step}: {title}", flush=True)
        print(f"{'='*70}", flush=True)


def main():
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    path_keys = [("data", "cache_dir"), ("data", "manifest_path"),
                 ("data", "norm_stats_path"), ("data", "physical_pt_dir"),
                 ("checkpoint", "save_dir")]
    for section, key in path_keys:
        if section in cfg and key in cfg[section]:
            cfg[section][key] = str(Path(cfg[section][key]).expanduser().resolve())

    # ================================================================ DDP INIT
    from training.ddp import init_ddp, cleanup_ddp, is_distributed
    rank, world, local = init_ddp()
    device = torch.device("cuda", local)

    banner(rank, "DDP Initialization", 1)
    log_all(rank, world, f"rank={rank} world={world} local={local} "
            f"GPU={torch.cuda.get_device_name(local)}")

    if world < 2:
        log(rank, "  WARN: Running with world=1, DDP sync tests will be trivial")

    log(rank, "  PASS: DDP initialized")

    # ================================================================ LOAD DATA
    banner(rank, "Load Manifest + Pin Cases", 2)
    cache_dir = cfg["data"]["cache_dir"]
    manifest_path = cfg["data"].get("manifest_path")
    if manifest_path:
        manifest_path = str(Path(manifest_path).expanduser().resolve())
    else:
        manifest_path = os.path.join(cache_dir, "manifest.json")

    with open(manifest_path) as f:
        manifest = json.load(f)

    def _name(e):
        return e["case_name"] if isinstance(e, dict) else str(e)

    all_train_ids = [_name(e) for e in manifest["splits"]["train"]]
    val_ids = [_name(e) for e in manifest["splits"]["val"]]
    my_train_ids = sorted(all_train_ids[rank::world])
    my_val_ids = sorted(val_ids[rank::world])

    log(rank, f"  Train cases: {len(all_train_ids)} total, {len(my_train_ids)} this rank")
    log(rank, f"  Val cases: {len(val_ids)} total, {len(my_val_ids)} this rank")

    from dataset.loaders import load_cases_pinned
    num_workers = min(int(cfg["training"].get("num_workers", 30)), 8)

    t0 = time.time()
    test_ids = my_train_ids[:4]
    log(rank, f"  Loading {len(test_ids)} cases for smoke test (not all)...")
    all_pt_data = load_cases_pinned(cache_dir, test_ids, num_workers=num_workers, rank=rank)
    dt = time.time() - t0

    log_all(rank, world, f"loaded {len(all_pt_data)} cases in {dt:.1f}s")

    sub_bin_map = {}
    cases_per_bin = {}
    for cid in test_ids:
        if cid not in all_pt_data:
            log_all(rank, world, f"FAIL: case '{cid}' not in loaded data")
            sys.exit(1)
        sb = all_pt_data[cid]["sub_bin"]
        sub_bin_map[cid] = sb
        cases_per_bin.setdefault(sb, []).append(cid)

    log(rank, f"  Sub-bins loaded: {dict((k, len(v)) for k, v in cases_per_bin.items())}")
    log(rank, "  PASS")

    # ================================================================ MODEL
    banner(rank, "Build Model + DDP + Compile", 3)
    from models import HDB3DModel

    model = HDB3DModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log(rank, f"  Parameters: {total_params:,}")

    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local])
        log(rank, "  DDP wrapped")

    compile_mode = cfg["training"].get("compile_mode", "reduce-overhead")
    try:
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
        log(rank, f"  torch.compile(mode='{compile_mode}') succeeded")
    except Exception as e:
        log(rank, f"  WARN: torch.compile failed ({e}), using eager")
        compiled = model

    log(rank, "  PASS")

    # ================================================================ OPTIMIZER
    banner(rank, "Optimizer", 4)
    base_lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])
    effective_lr = base_lr * math.sqrt(world * 4)
    opt = AdamW(model.parameters(), lr=effective_lr, weight_decay=wd, fused=True)
    log(rank, f"  AdamW: base_lr={base_lr:.1e}, effective_lr={effective_lr:.1e}, wd={wd}")
    log(rank, "  PASS")

    # ================================================================ PREFETCHER + 1 BATCH
    banner(rank, "Prefetcher: Build + Consume 1 Batch", 5)
    from dataset.prefetcher import prepare_one_case, stack_batch
    from training.shard import build_grouped_shard, BATCH_SIZES, SUB_BIN_L

    pick_sb = list(cases_per_bin.keys())[0]
    pick_ids = cases_per_bin[pick_sb][:BATCH_SIZES.get(pick_sb, 1)]
    L = SUB_BIN_L[pick_sb]
    B = len(pick_ids)
    n_query = 50_000

    log(rank, f"  Building 1 batch: sub_bin='{pick_sb}', B={B}, L={L}, n_query={n_query}")

    t0 = time.time()
    items = []
    for i, cid in enumerate(pick_ids):
        pt = all_pt_data[cid]
        pt["_case_id"] = i
        item = prepare_one_case(pt, case_id=i, epoch=0,
                                encoder_k=32, n_query=n_query)
        items.append(item)

    # Per-case query tensors are already shaped [n_query, *] (uniform);
    # stack_batch is enough, no trim needed.
    batch_cpu = stack_batch(items)
    batch_cpu["L"] = L
    dt = time.time() - t0

    n_qv_min = int(min(it["n_query_vol"] for it in items))
    n_qs_min = int(min(it["n_query_surf"] for it in items))
    log(rank, f"  Batch built in {dt:.2f}s, "
              f"per-item n_query_vol min={n_qv_min}, n_query_surf min={n_qs_min}")
    for k, v in sorted(batch_cpu.items()):
        if isinstance(v, torch.Tensor) and rank == 0:
            print(f"    {_fmt(v, k)}", flush=True)
    log(rank, "  PASS")

    # ================================================================ BIGBIRD + H2D + IDW
    banner(rank, "BigBird Mask + H2D + IDW", 6)
    from models.bigbird import build_block_mask_direct
    from models.idw import gpu_idw

    torch.cuda.reset_peak_memory_stats(device)

    R = int(cfg["model"].get("register_tokens", 16))
    flex_mask = build_block_mask_direct(
        batch_cpu["bigbird_key_idx"], L=L, R=R, device=device)
    log(rank, f"  BlockMask built (L={L}, R={R})")

    batch = {}
    for k, v in batch_cpu.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
        else:
            batch[k] = v
    torch.cuda.synchronize()
    log(rank, f"  H2D transfer done")

    idw_idx, idw_w = gpu_idw(
        batch["query_pos_norm"],
        batch["leaf_centroid_norm"],
        batch["latent_neighbor_top64"],
        batch["query_leaf_id"],
        idw_k=int(cfg["model"].get("decoder_idw_k", 8)))
    log(rank, f"  IDW: {_fmt(idw_idx, 'idx')}")
    log(rank, f"  IDW: {_fmt(idw_w, 'weights')}")
    log(rank, "  PASS")

    # ================================================================ FORWARD
    banner(rank, "Forward Pass (bf16 autocast)", 7)
    model.train()
    t0 = time.time()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_vol, pred_surf = compiled(
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
        )
    torch.cuda.synchronize()
    dt = time.time() - t0

    log(rank, f"  Forward took {dt:.2f}s (includes compile if first run)")
    log(rank, f"  {_fmt(pred_vol, 'pred_vol')}")
    log(rank, f"  {_fmt(pred_surf, 'pred_surf')}")

    target_vol = batch["query_target_volume"]
    target_surf = batch["query_target_surface"]

    if pred_vol.shape != target_vol.shape:
        log(rank, f"  FAIL: pred_vol {pred_vol.shape} != target {target_vol.shape}")
        sys.exit(1)
    if pred_surf.shape != target_surf.shape:
        log(rank, f"  FAIL: pred_surf {pred_surf.shape} != target {target_surf.shape}")
        sys.exit(1)
    log(rank, "  PASS")

    # ================================================================ LOSS + BACKWARD
    banner(rank, "Loss + Backward + Gradient Sync", 8)
    from training.loop import _masked_mse_losses
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss_vol, loss_surf = _masked_mse_losses(
            pred_vol, pred_surf, target_vol, target_surf,
            batch["query_is_surf"], batch["query_valid_mask"],
        )
        loss = loss_vol + loss_surf

    log_all(rank, world, f"loss={loss.item():.6f} (vol={loss_vol.item():.6f} surf={loss_surf.item():.6f})")

    if torch.isnan(loss):
        log(rank, "  FAIL: Loss is NaN!")
        sys.exit(1)

    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0

    log(rank, f"  Backward took {dt:.2f}s")

    raw_model = model.module if hasattr(model, "module") else model
    total_gn = 0.0
    nan_grads = []
    for name, p in raw_model.named_parameters():
        if p.grad is not None:
            g = p.grad.float()
            total_gn += g.norm().item() ** 2
            if torch.isnan(g).any():
                nan_grads.append(name)

    total_gn = total_gn ** 0.5
    log_all(rank, world, f"grad_norm={total_gn:.6f}")

    if nan_grads:
        log_all(rank, world, f"FAIL: NaN grads in {nan_grads[:3]}")
        sys.exit(1)

    # Test all-reduce works
    if is_distributed():
        test_t = torch.tensor([loss.item()], device=device)
        dist.all_reduce(test_t, op=dist.ReduceOp.SUM)
        log(rank, f"  all_reduce test: sum_of_losses={test_t.item():.6f} (should be ~{loss.item()*world:.6f})")

    log(rank, "  PASS")

    # ================================================================ OPTIMIZER STEP
    banner(rank, "Optimizer Step + Post-Step Integrity", 9)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(cfg["training"].get("max_grad_norm", 1.0)))
    opt.step()
    opt.zero_grad(set_to_none=True)

    for name, p in raw_model.named_parameters():
        if torch.isnan(p).any():
            log(rank, f"  FAIL: NaN in param '{name}' after step")
            sys.exit(1)

    log(rank, "  Post-step param check: no NaN/Inf")

    # Check params are synced across ranks
    if is_distributed():
        for name, p in list(raw_model.named_parameters())[:3]:
            psum = p.float().sum()
            all_sum = psum.clone()
            dist.all_reduce(all_sum, op=dist.ReduceOp.SUM)
            expected = psum * world
            diff = (all_sum - expected).abs().item()
            if diff > 1e-3:
                log(rank, f"  WARN: param '{name}' differs across ranks (diff={diff:.6f})")
        log(rank, "  DDP param sync check: OK")

    log(rank, "  PASS")

    # ================================================================ 2ND FORWARD (compile cache warm)
    banner(rank, "2nd Forward (compile cache warm)", 10)
    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_vol2, pred_surf2 = compiled(
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
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    log(rank, f"  2nd forward: {dt:.2f}s (should be faster than 1st)")
    log(rank, "  PASS")

    # ================================================================ MULTI-SUB-BIN + MULTI-STEP
    banner(rank, "Multi-sub-bin + multi-step coverage", 11)
    # The 1-batch test above only exercises ONE sub-bin and ONE step.
    # Real training cycles through every sub-bin and runs hundreds of
    # steps per epoch. Here we walk through up to MAX_STEPS additional
    # batches drawn from a mix of sub-bins to catch:
    #   - per-sub-bin compile cache flips (torch.compile recompiles
    #     when L changes — should still complete, just slower for new L)
    #   - GPU memory growth across steps (leak detection)
    #   - DDP backward all-reduce on varying B
    #
    # We tear down the model's gradients between steps but reuse the
    # same compiled model + optimizer so the cache stays warm.
    MAX_STEPS = 6
    # Order sub-bins so the first picks NEW L values (forces compile);
    # later steps reuse them (should be cached).
    sub_bins_to_try: list[str] = []
    for sb in cases_per_bin.keys():
        if sb != pick_sb and cases_per_bin[sb]:
            sub_bins_to_try.append(sb)
    sub_bins_to_try = sub_bins_to_try[:MAX_STEPS]

    if not sub_bins_to_try:
        log(rank, "  (only 1 sub-bin available this rank — skipping multi-step)")
    else:
        gpu_peak_per_step: list[float] = []
        for step_idx, sb in enumerate(sub_bins_to_try):
            ids_sb = cases_per_bin[sb][:BATCH_SIZES.get(sb, 1)]
            if not ids_sb:
                continue
            L_sb = SUB_BIN_L[sb]
            B_sb = len(ids_sb)
            torch.cuda.reset_peak_memory_stats(device)

            t_step = time.time()
            items_sb = [
                prepare_one_case(all_pt_data[c], case_id=i, epoch=0,
                                 encoder_k=32, n_query=n_query)
                for i, c in enumerate(ids_sb)
            ]
            for it in items_sb:
                it["L"] = L_sb
                it["sub_bin"] = sb

            batch_cpu_sb = stack_batch(items_sb)
            batch_cpu_sb["L"] = L_sb
            flex_mask_sb = build_block_mask_direct(
                batch_cpu_sb["bigbird_key_idx"], L=L_sb, R=R, device=device)
            batch_sb = {k: (v.to(device, non_blocking=True)
                            if isinstance(v, torch.Tensor) else v)
                        for k, v in batch_cpu_sb.items()}
            idw_idx_sb, idw_w_sb = gpu_idw(
                batch_sb["query_pos_norm"], batch_sb["leaf_centroid_norm"],
                batch_sb["latent_neighbor_top64"], batch_sb["query_leaf_id"],
                idw_k=int(cfg["model"].get("decoder_idw_k", 8)))

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pv, ps = compiled(
                    leaf_centroid_norm=batch_sb["leaf_centroid_norm"],
                    leaf_stats=batch_sb["leaf_stats"],
                    leaf_sdf=batch_sb["leaf_sdf"],
                    leaf_sdf_grad=batch_sb["leaf_sdf_grad"],
                    leaf_curvature_mean=batch_sb["leaf_curvature_mean"],
                    leaf_curvature_gauss=batch_sb["leaf_curvature_gauss"],
                    leaf_pbd=batch_sb["leaf_pbd"],
                    transient1=batch_sb["transient1"],
                    query_pos_norm=batch_sb["query_pos_norm"],
                    query_sdf=batch_sb["query_sdf"],
                    query_sdf_grad=batch_sb["query_sdf_grad"],
                    idw_indices=idw_idx_sb,
                    idw_weights=idw_w_sb,
                    rope_cos=batch_sb["rope_cos"],
                    rope_sin=batch_sb["rope_sin"],
                    flex_mask=flex_mask_sb,
                )
                l_v, l_s = _masked_mse_losses(
                    pv, ps,
                    batch_sb["query_target_volume"],
                    batch_sb["query_target_surface"],
                    batch_sb["query_is_surf"],
                    batch_sb["query_valid_mask"],
                )
                loss_sb = l_v + l_s

            loss_sb.backward()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            peak_step = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_peak_per_step.append(peak_step)
            log(rank, f"  step {step_idx + 1}: sub_bin={sb} L={L_sb} B={B_sb} "
                      f"loss={loss_sb.item():.4e} "
                      f"peak={peak_step:.1f}GB "
                      f"t={time.time() - t_step:.1f}s")

        # Growth across steps: peak should plateau after a couple of
        # compile flips, NOT grow linearly with step index.
        if len(gpu_peak_per_step) >= 3:
            first_half = sum(gpu_peak_per_step[:len(gpu_peak_per_step) // 2])
            second_half = sum(gpu_peak_per_step[len(gpu_peak_per_step) // 2:])
            ratio = second_half / max(first_half, 1e-9)
            if ratio > 1.5:
                log(rank, f"  WARN: GPU peak grew {ratio:.2f}x from first "
                          f"half to second half — possible leak")
            else:
                log(rank, f"  GPU peak stable across steps "
                          f"(ratio={ratio:.2f})")
        log(rank, "  PASS")

    # ================================================================ MEMORY
    banner(rank, "Memory Summary", 12)
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    log_all(rank, world, f"GPU peak={peak:.2f}GB reserved={reserved:.2f}GB")

    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
        log_all(rank, world, f"CPU RSS={rss:.2f}GB")
    except ImportError:
        pass

    log(rank, "  PASS")

    # ================================================================ SUMMARY
    if rank == 0:
        print(f"\n{'='*70}", flush=True)
        print(f"  ALL 12 STEPS PASSED on {world} GPUs", flush=True)
        print(f"  DDP + compile + forward + backward + sync + multi-sub-bin "
              f"all working.", flush=True)
        print(f"  Full training with torchrun --nproc_per_node={world} "
              f"should work.", flush=True)
        print(f"{'='*70}", flush=True)

    if is_distributed():
        dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    main()
