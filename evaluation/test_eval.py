"""Test full-point inference with physical-unit metrics.

DDP across 4 ranks. For each rank's shard of 50 test cases:
  1. Load case from pinned memory
  2. Encoder + ViT once -> (enc_feat, vit_feat) on GPU
  3. Decoder chunked at 4M points per chunk
  4. Denormalize to physical units (un-zscore + reverse p detrend)
  5. Compute per-field relative L2
  6. Two-pass: metrics first, then viz for median-error case
"""
from __future__ import annotations

import json
import os
import os.path as osp
import pickle
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from hdb.models.encoder import build_leaf_aggregate
from hdb.models.idw import gpu_idw
from hdb.training.ddp import is_distributed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def relative_l2_scalar(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative L2 error for a scalar field: ||pred-gt||_2 / ||gt||_2."""
    diff = pred.float() - target.float()
    return float(torch.norm(diff) / torch.norm(target.float()).clamp(min=1e-12))


def relative_l2_vector(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative L2 for a multi-component vector field (flattened)."""
    diff = (pred.float() - target.float()).reshape(-1)
    ref = target.float().reshape(-1)
    return float(torch.norm(diff) / torch.norm(ref).clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Denormalization
# ---------------------------------------------------------------------------

def denormalize_volume(pred_z: torch.Tensor, norm_stats: dict,
                       pos_z_physical: torch.Tensor
                       ) -> torch.Tensor:
    """Un-zscore volume predictions and reverse p detrend.

    pred_z: (N, 5) z-scored [Ux, Uy, Uz, p_det, nut]
    pos_z_physical: (N,) z coordinate in physical metres (point_pos_norm[:,2] * 550)
    norm_stats: dict with vol_mean (5,) and vol_std (5,)

    Returns (N, 5) physical [Ux, Uy, Uz, p_phys, nut].
    """
    vol_mean = torch.as_tensor(norm_stats["vol_mean"], dtype=torch.float32,
                               device=pred_z.device)
    vol_std = torch.as_tensor(norm_stats["vol_std"], dtype=torch.float32,
                              device=pred_z.device)
    pred_phys = pred_z.float() * vol_std + vol_mean  # (N, 5)

    # Reverse detrend for p (channel 3): p_phys = p_detrended - 9.81 * z
    G = 9.81
    pred_phys[:, 3] = pred_phys[:, 3] - G * pos_z_physical

    return pred_phys


def denormalize_surface(pred_z: torch.Tensor, norm_stats: dict,
                        pos_z_physical: torch.Tensor
                        ) -> torch.Tensor:
    """Un-zscore surface predictions and reverse p detrend.

    pred_z: (N, 1) z-scored [p_det]
    pos_z_physical: (N,) z coordinate in physical metres
    norm_stats: dict with surf_mean (1,) and surf_std (1,)

    Returns (N, 1) physical [p_phys].
    """
    surf_mean = torch.as_tensor(norm_stats["surf_mean"], dtype=torch.float32,
                                device=pred_z.device)
    surf_std = torch.as_tensor(norm_stats["surf_std"], dtype=torch.float32,
                               device=pred_z.device)
    pred_phys = pred_z.float() * surf_std + surf_mean  # (N, 1)

    G = 9.81
    pred_phys[:, 0] = pred_phys[:, 0] - G * pos_z_physical

    return pred_phys


# ---------------------------------------------------------------------------
# Chunked decoding
# ---------------------------------------------------------------------------

def _get_model(model):
    """Unwrap DDP if needed."""
    return model.module if hasattr(model, "module") else model


def _decode_chunk(model, enc_feat: torch.Tensor, vit_feat: torch.Tensor,
                  pt: dict, lo: int, hi: int, is_volume: bool,
                  device: torch.device, idw_k: int = 8
                  ) -> torch.Tensor:
    """Decode a contiguous chunk [lo, hi) from either volume or surface points.

    Returns raw z-scored predictions: (N_chunk, C).
    """
    # Determine which reorder index to use
    if is_volume:
        reorder = pt["vol_reorder_idx"]
        global_idx = reorder[lo:hi]
    else:
        reorder = pt["surf_reorder_idx"]
        global_idx = reorder[lo:hi]

    if isinstance(global_idx, torch.Tensor):
        global_idx = global_idx.long()
    else:
        global_idx = torch.as_tensor(global_idx, dtype=torch.long)

    # Gather query features
    point_pos_norm = pt["point_pos_norm"][global_idx].to(device, non_blocking=True)
    point_sdf = pt["point_sdf"][global_idx].to(device, non_blocking=True)
    point_sdf_grad = pt["point_sdf_grad"][global_idx].to(device, non_blocking=True)
    point_leaf_id = pt["point_leaf_id"][global_idx].long().to(device, non_blocking=True)

    # Build IDW neighbor candidates
    leaf_centroid_norm = pt["leaf_centroid_norm"].to(device, non_blocking=True)
    neighbor_top64 = pt["latent_neighbor_top64"].to(device, non_blocking=True)

    idw_idx, idw_w = gpu_idw(
        point_pos_norm, leaf_centroid_norm, neighbor_top64,
        point_leaf_id, idw_k=idw_k)

    # n_query_vol: if volume chunk, all queries are volume; if surface, all surface
    n_chunk = hi - lo
    if is_volume:
        n_qv = n_chunk
    else:
        n_qv = 0

    raw_model = _get_model(model)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_vol, pred_surf = raw_model.decoder(
            enc_feat, vit_feat,
            point_pos_norm.unsqueeze(0),
            point_sdf.unsqueeze(0).to(torch.bfloat16),
            point_sdf_grad.unsqueeze(0).to(torch.bfloat16),
            idw_idx.unsqueeze(0),
            idw_w.unsqueeze(0).to(torch.bfloat16),
            n_query_vol=n_qv,
        )

    if is_volume:
        return pred_vol.squeeze(0).float()
    else:
        return pred_surf.squeeze(0).float()


# ---------------------------------------------------------------------------
# Encode (encoder + ViT, once per case)
# ---------------------------------------------------------------------------

def _encode_case(model, pt: dict, device: torch.device, cfg: dict
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoder + ViT for a single test case (B=1).

    Returns (enc_feat, vit_feat) each (1, L, dim).
    """
    register_tokens = int(cfg["model"].get("register_tokens", 16))
    n_random = int(cfg["model"].get("bigbird_random", 32))

    # Leaf tensors
    leaf_keys = [
        "leaf_centroid_norm", "leaf_stats", "leaf_sdf",
        "leaf_sdf_grad", "leaf_curvature_mean", "leaf_curvature_gauss",
        "leaf_pbd",
    ]
    batch: dict[str, torch.Tensor] = {}
    for k in leaf_keys:
        batch[k] = pt[k].unsqueeze(0).to(device, non_blocking=True)

    L = pt["leaf_centroid_norm"].shape[0]

    # Transient1: for test, use a fixed seed (seed=42, baked)
    # Sample from encoder_pool with fixed seed
    pool = pt["encoder_pool"]  # (L, 256, 10) bf16
    encoder_k = int(cfg["model"].get("encoder_k", 32))
    rng = np.random.default_rng(42)
    sampled_idx = rng.integers(0, 256, size=(L, encoder_k))
    if isinstance(pool, torch.Tensor):
        t1 = pool[torch.arange(L)[:, None], torch.from_numpy(sampled_idx)]
    else:
        t1 = pool[np.arange(L)[:, None], sampled_idx]
    t1 = torch.as_tensor(np.asarray(t1), dtype=torch.float32)
    batch["transient1"] = t1.unsqueeze(0).to(device, non_blocking=True)

    # BigBird: for test, use all local+register (no random) for determinism
    fixed = pt["bigbird_fixed"]  # (L, 80) int32
    if isinstance(fixed, torch.Tensor):
        fixed_np = fixed.numpy()
    else:
        fixed_np = np.asarray(fixed)
    # Still append n_random keys for shape compatibility, use deterministic seed
    rng_bb = np.random.RandomState(42)
    random_keys = rng_bb.randint(0, L, size=(L, n_random)).astype(np.int32)
    bb_idx = np.concatenate([fixed_np, random_keys], axis=1)
    batch["bigbird_key_idx"] = torch.from_numpy(bb_idx).to(
        torch.int32).unsqueeze(0).to(device, non_blocking=True)

    # RoPE (precomputed)
    rope_cos = pt["rope_cos"].unsqueeze(0).to(device, non_blocking=True)
    rope_sin = pt["rope_sin"].unsqueeze(0).to(device, non_blocking=True)

    # FlexAttention BlockMask
    from hdb.models.bigbird import build_block_mask_direct
    flex_mask = build_block_mask_direct(
        batch["bigbird_key_idx"], L=L, R=register_tokens)

    raw_model = _get_model(model)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        enc_feat, vit_feat = raw_model.encode(
            leaf_centroid_norm=batch["leaf_centroid_norm"],
            leaf_stats=batch["leaf_stats"],
            leaf_sdf=batch["leaf_sdf"],
            leaf_sdf_grad=batch["leaf_sdf_grad"],
            leaf_curvature_mean=batch["leaf_curvature_mean"],
            leaf_curvature_gauss=batch["leaf_curvature_gauss"],
            leaf_pbd=batch["leaf_pbd"],
            transient1=batch["transient1"],
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            flex_mask=flex_mask,
        )

    return enc_feat, vit_feat


# ---------------------------------------------------------------------------
# Single-case full evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_one_case(model, pt: dict, norm_stats: dict,
                   device: torch.device, cfg: dict,
                   chunk_size: int = 4_000_000,
                   keep_arrays: bool = False) -> dict[str, Any]:
    """Full-point inference for one test case.

    Returns metrics dict with per-field relative L2 and optionally raw arrays.
    """
    COORD_MAX = float(cfg.get("physical_constants", {}).get("COORD_MAX", 550.0))

    # Encoder + ViT (once)
    enc_feat, vit_feat = _encode_case(model, pt, device, cfg)

    N_vol = pt["vol_reorder_idx"].shape[0]
    N_surf = pt["surf_reorder_idx"].shape[0]

    # --- Volume chunks ---
    vol_preds = []
    for lo in range(0, N_vol, chunk_size):
        hi = min(lo + chunk_size, N_vol)
        pred_chunk = _decode_chunk(
            model, enc_feat, vit_feat, pt, lo, hi,
            is_volume=True, device=device)
        vol_preds.append(pred_chunk.cpu())
    pred_vol_z = torch.cat(vol_preds, dim=0)  # (N_vol, 5) z-scored

    # --- Surface chunks ---
    surf_preds = []
    for lo in range(0, N_surf, chunk_size):
        hi = min(lo + chunk_size, N_surf)
        pred_chunk = _decode_chunk(
            model, enc_feat, vit_feat, pt, lo, hi,
            is_volume=False, device=device)
        surf_preds.append(pred_chunk.cpu())
    pred_surf_z = torch.cat(surf_preds, dim=0)  # (N_surf, 1) z-scored

    # --- Denormalize predictions ---
    # Physical z coordinate from pos_norm
    vol_reorder = pt["vol_reorder_idx"]
    surf_reorder = pt["surf_reorder_idx"]
    if isinstance(vol_reorder, torch.Tensor):
        vol_reorder = vol_reorder.long()
    else:
        vol_reorder = torch.as_tensor(vol_reorder, dtype=torch.long)
    if isinstance(surf_reorder, torch.Tensor):
        surf_reorder = surf_reorder.long()
    else:
        surf_reorder = torch.as_tensor(surf_reorder, dtype=torch.long)

    pos_norm = pt["point_pos_norm"]  # (N_total, 3)
    vol_z_phys = pos_norm[vol_reorder, 2].float() * COORD_MAX   # metres
    surf_z_phys = pos_norm[surf_reorder, 2].float() * COORD_MAX

    pred_vol_phys = denormalize_volume(pred_vol_z, norm_stats, vol_z_phys)
    pred_surf_phys = denormalize_surface(pred_surf_z, norm_stats, surf_z_phys)

    # --- Denormalize ground truth (same procedure) ---
    gt_vol_z = pt["point_y_volume"].float()   # (N_vol, 5) z-scored
    gt_surf_z = pt["point_y_surface"].float()  # (N_surf, 1) z-scored
    gt_vol_phys = denormalize_volume(gt_vol_z, norm_stats, vol_z_phys)
    gt_surf_phys = denormalize_surface(gt_surf_z, norm_stats, surf_z_phys)

    # --- Per-field relative L2 ---
    metrics: dict[str, Any] = {
        "Ux_rel_L2": relative_l2_scalar(pred_vol_phys[:, 0], gt_vol_phys[:, 0]),
        "Uy_rel_L2": relative_l2_scalar(pred_vol_phys[:, 1], gt_vol_phys[:, 1]),
        "Uz_rel_L2": relative_l2_scalar(pred_vol_phys[:, 2], gt_vol_phys[:, 2]),
        "p_vol_rel_L2": relative_l2_scalar(pred_vol_phys[:, 3], gt_vol_phys[:, 3]),
        "nut_rel_L2": relative_l2_scalar(pred_vol_phys[:, 4], gt_vol_phys[:, 4]),
        "p_surf_rel_L2": relative_l2_scalar(pred_surf_phys[:, 0], gt_surf_phys[:, 0]),
    }

    # Mean relative L2 across all fields (for median-case selection)
    all_fields = [metrics[k] for k in [
        "Ux_rel_L2", "Uy_rel_L2", "Uz_rel_L2",
        "p_vol_rel_L2", "nut_rel_L2", "p_surf_rel_L2"]]
    metrics["mean_rel_L2"] = float(np.mean(all_fields))

    if keep_arrays:
        metrics["_pred_vol_phys"] = pred_vol_phys
        metrics["_pred_surf_phys"] = pred_surf_phys
        metrics["_gt_vol_phys"] = gt_vol_phys
        metrics["_gt_surf_phys"] = gt_surf_phys
        # Volume point positions for visualization
        metrics["_vol_pos"] = pos_norm[vol_reorder].float() * COORD_MAX
        metrics["_surf_pos"] = pos_norm[surf_reorder].float() * COORD_MAX

    return metrics


# ---------------------------------------------------------------------------
# DDP gather
# ---------------------------------------------------------------------------

def _allgather_metrics(local_metrics: dict[str, dict[str, Any]],
                       rank: int, world: int,
                       device: torch.device) -> dict[str, dict[str, Any]]:
    """Gather scalar per-case metrics from all ranks to rank 0."""
    if world <= 1 or not is_distributed():
        return local_metrics

    # Serialize only scalar metrics
    serializable = {}
    for cid, m in local_metrics.items():
        serializable[cid] = {k: v for k, v in m.items()
                             if not k.startswith("_")}
    data = pickle.dumps(serializable)
    size_t = torch.tensor([len(data)], dtype=torch.long, device=device)
    sizes = [torch.zeros(1, dtype=torch.long, device=device)
             for _ in range(world)]
    dist.all_gather(sizes, size_t)
    max_size = max(s.item() for s in sizes)

    buf = torch.zeros(max_size, dtype=torch.uint8, device=device)
    buf[:len(data)] = torch.frombuffer(
        bytearray(data), dtype=torch.uint8).to(device)
    all_bufs = [torch.zeros(max_size, dtype=torch.uint8, device=device)
                for _ in range(world)]
    dist.all_gather(all_bufs, buf)

    merged: dict[str, dict[str, Any]] = {}
    for i in range(world):
        sz = int(sizes[i].item())
        remote_data = bytes(all_bufs[i][:sz].cpu().numpy().tobytes())
        remote_metrics = pickle.loads(remote_data)
        merged.update(remote_metrics)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_full_inference(model, test_case_pts: dict[str, dict],
                        norm_stats: dict, device: torch.device,
                        cfg: dict, chunk_size: int = 4_000_000,
                        ) -> dict[str, Any]:
    """Run full-point test inference across DDP ranks.

    Two-pass evaluation:
      Pass 1: compute per-field metrics for all cases
      Pass 2: re-run median-error case with keep_arrays=True for viz

    Parameters
    ----------
    model : HDB3DModel (possibly DDP-wrapped), must be in eval mode.
    test_case_pts : dict mapping case_id -> pre-loaded PT dict.
    norm_stats : loaded norm_stats.json dict.
    device : CUDA device for this rank.
    cfg : full config dict.
    chunk_size : points per decoder chunk.

    Returns
    -------
    On rank 0: dict with 'per_case' (50 entries), 'summary' (mean/std),
               'median_case_id', and optionally arrays for viz.
    On other ranks: empty dict.
    """
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    model.eval()

    # --- Pass 1: metrics for all assigned cases ---
    my_per_case: dict[str, dict[str, Any]] = {}
    for i, (case_id, pt) in enumerate(test_case_pts.items()):
        if rank == 0:
            print(f"  [test pass 1] case {i + 1}/{len(test_case_pts)}: {case_id}",
                  flush=True)
        my_per_case[case_id] = _eval_one_case(
            model, pt, norm_stats, device, cfg,
            chunk_size=chunk_size, keep_arrays=False)
        torch.cuda.empty_cache()

    # Gather to rank 0
    per_case = _allgather_metrics(my_per_case, rank, world, device)

    result: dict[str, Any] = {}

    if rank == 0:
        # --- Select median-error case ---
        case_ids_sorted = sorted(per_case.keys(),
                                 key=lambda c: per_case[c]["mean_rel_L2"])
        median_idx = len(case_ids_sorted) // 2
        median_case_id = case_ids_sorted[median_idx]

        # --- Compute summary statistics ---
        field_keys = ["Ux_rel_L2", "Uy_rel_L2", "Uz_rel_L2",
                      "p_vol_rel_L2", "nut_rel_L2", "p_surf_rel_L2"]
        summary: dict[str, dict[str, float]] = {}
        for fk in field_keys:
            vals = [per_case[c][fk] for c in per_case]
            summary[fk] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        # --- Pass 2: re-run median case with arrays for viz ---
        if median_case_id in test_case_pts:
            print(f"  [test pass 2] re-running median case {median_case_id} "
                  f"for visualization", flush=True)
            per_case[median_case_id] = _eval_one_case(
                model, test_case_pts[median_case_id], norm_stats, device, cfg,
                chunk_size=chunk_size, keep_arrays=True)
            torch.cuda.empty_cache()

        result = {
            "per_case": per_case,
            "summary": summary,
            "median_case_id": median_case_id,
        }

    if is_distributed():
        dist.barrier()

    return result
