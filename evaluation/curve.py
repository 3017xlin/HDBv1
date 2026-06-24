"""Train-eval and validation curve evaluation.

For train_eval (40 cases) and val (50 cases):
  - 500K sampling per case (same Gumbel-top-k as training transient2)
  - Compute z-score MSE (same space as training loss)
  - Return per-field MSE: {vol_mse, surf_mse, total_mse}
  - No gradient computation

Cases are processed one-at-a-time to handle variable L across sub-bins.
"""
from __future__ import annotations

import json
import math
import os
import os.path as osp
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from hdb.models.encoder import build_leaf_aggregate
from hdb.models.idw import gpu_idw
from hdb.models.rope import precompute_rope_3d
from hdb.training.ddp import is_distributed
from hdb.utils.seed import per_case_epoch_seed


# ---------------------------------------------------------------------------
# Sampling helpers (mirror training transient2)
# ---------------------------------------------------------------------------

def _gumbel_topk(log_weights: np.ndarray, k: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Gumbel-top-k sampling with pre-computed log weights."""
    u = rng.random(len(log_weights)).astype(np.float64)
    gumbel = -np.log(-np.log(np.clip(u, 1e-10, 1.0 - 1e-10)))
    perturbed = log_weights.astype(np.float64) + gumbel
    idx = np.argpartition(-perturbed, k)[:k]
    return idx


def _build_transient1(pt: dict, epoch: int, encoder_k: int = 32
                      ) -> torch.Tensor:
    """Sample 32 neighbours from the precomputed encoder_pool (L, 256, 10).

    Returns (L, 32, 10) float32.
    """
    case_id = pt.get("_case_id", 0)
    seed = per_case_epoch_seed(case_id, epoch)
    rng = np.random.default_rng(seed)

    L = int(pt["L"]) if isinstance(pt.get("L"), (int, np.integer)) else pt["encoder_pool"].shape[0]
    pool = pt["encoder_pool"]  # (L, 256, 10) bf16
    sampled_idx = rng.integers(0, 256, size=(L, encoder_k))  # (L, 32)
    t1 = pool[np.arange(L)[:, None], sampled_idx]             # (L, 32, 10)
    if isinstance(t1, torch.Tensor):
        return t1.float()
    return torch.from_numpy(np.asarray(t1, dtype=np.float32))


def _build_transient2(pt: dict, epoch: int, n_query: int = 500_000,
                      surf_ratio: float = 2.0
                      ) -> dict[str, Any]:
    """Build query sample (same logic as training transient2).

    Returns dict with query tensors and target tensors for MSE computation.
    """
    case_id = pt.get("_case_id", 0)
    seed = per_case_epoch_seed(case_id, epoch) ^ 0xA5A5_A5A5
    rng = np.random.default_rng(seed)

    vol_reorder = pt["vol_reorder_idx"]
    surf_reorder = pt["surf_reorder_idx"]
    N_vol = vol_reorder.shape[0] if isinstance(vol_reorder, torch.Tensor) else len(vol_reorder)
    N_surf = surf_reorder.shape[0] if isinstance(surf_reorder, torch.Tensor) else len(surf_reorder)

    # Compute surface/volume split
    n_query_surf = min(N_surf, int(n_query * N_surf / (N_surf + N_vol / surf_ratio)))
    n_query_vol = n_query - n_query_surf

    # Surface: uniform sampling
    surf_choice = rng.choice(N_surf, size=n_query_surf, replace=False)
    surf_choice = torch.as_tensor(surf_choice, dtype=torch.long)

    # Volume: Gumbel-top-k with precomputed log weights
    log_w = pt["vol_log_sample_weight"]
    if isinstance(log_w, torch.Tensor):
        log_w = log_w.numpy()
    vol_choice = _gumbel_topk(log_w, n_query_vol, rng)
    vol_choice = torch.as_tensor(vol_choice, dtype=torch.long)

    # Global indices
    vol_reorder_t = torch.as_tensor(vol_reorder, dtype=torch.long)
    surf_reorder_t = torch.as_tensor(surf_reorder, dtype=torch.long)
    vol_global_idx = vol_reorder_t[vol_choice]
    surf_global_idx = surf_reorder_t[surf_choice]
    query_idx = torch.cat([vol_global_idx, surf_global_idx])

    # Gather query features from cache (all already z-scored)
    query_pos = pt["point_pos_norm"][query_idx]
    query_sdf = pt["point_sdf"][query_idx]
    query_sdf_grad = pt["point_sdf_grad"][query_idx]
    query_leaf_id = pt["point_leaf_id"][query_idx]

    # Targets (z-scored)
    target_vol = pt["point_y_volume"][vol_choice]
    target_surf = pt["point_y_surface"][surf_choice]

    return {
        "query_pos_norm": query_pos,
        "query_sdf": query_sdf,
        "query_sdf_grad": query_sdf_grad,
        "query_leaf_id": query_leaf_id,
        "target_vol": target_vol,
        "target_surf": target_surf,
        "n_query_vol": int(n_query_vol),
    }


# ---------------------------------------------------------------------------
# BigBird index builder (fast path using precomputed bigbird_fixed)
# ---------------------------------------------------------------------------

def _build_bigbird_index(pt: dict, L: int, n_random: int = 32,
                         seed: int = 42) -> torch.Tensor:
    """Build BigBird key index: precomputed (64 local + 16 register) + 32 random.

    Returns (L, 112) int32.
    """
    fixed = pt["bigbird_fixed"]  # (L, 80) int32
    if isinstance(fixed, torch.Tensor):
        fixed_np = fixed.numpy()
    else:
        fixed_np = np.asarray(fixed)
    rng = np.random.RandomState(seed)
    random_keys = rng.randint(0, L, size=(L, n_random)).astype(np.int32)
    result = np.concatenate([fixed_np, random_keys], axis=1)
    return torch.from_numpy(result).to(torch.int32)


# ---------------------------------------------------------------------------
# Single-case forward pass (no gradient)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_one_case(model, pt: dict, norm_stats: dict, epoch: int,
                   device: torch.device, cfg: dict, n_query: int = 500_000
                   ) -> dict[str, float]:
    """Evaluate one case with 500K sampling, returning z-score MSE metrics."""
    encoder_k = int(cfg["model"].get("encoder_k", 32))
    n_random = int(cfg["model"].get("bigbird_random", 32))
    register_tokens = int(cfg["model"].get("register_tokens", 16))

    # --- Build transient inputs ---
    transient1 = _build_transient1(pt, epoch, encoder_k)
    t2 = _build_transient2(pt, epoch, n_query)

    # --- Leaf-level tensors ---
    L = pt["leaf_centroid_norm"].shape[0]
    leaf_keys = [
        "leaf_centroid_norm", "leaf_stats", "leaf_sdf",
        "leaf_sdf_grad", "leaf_curvature_mean", "leaf_curvature_gauss",
        "leaf_pbd",
    ]
    batch: dict[str, torch.Tensor] = {}
    for k in leaf_keys:
        batch[k] = pt[k].unsqueeze(0).to(device, non_blocking=True)
    batch["transient1"] = transient1.unsqueeze(0).to(device, non_blocking=True)

    # --- BigBird index ---
    bb_idx = _build_bigbird_index(pt, L, n_random=n_random, seed=42 + epoch)
    batch["bigbird_key_idx"] = bb_idx.unsqueeze(0).to(device, non_blocking=True)

    # --- RoPE (precomputed in PT) ---
    rope_cos = pt["rope_cos"].unsqueeze(0).to(device, non_blocking=True)
    rope_sin = pt["rope_sin"].unsqueeze(0).to(device, non_blocking=True)

    # --- FlexAttention BlockMask ---
    from hdb.models.bigbird import build_block_mask_direct
    flex_mask = build_block_mask_direct(
        batch["bigbird_key_idx"], L=L, R=register_tokens)

    # --- Encoder + ViT ---
    leaf_aggr = build_leaf_aggregate(
        leaf_stats=batch["leaf_stats"],
        leaf_sdf=batch["leaf_sdf"],
        leaf_sdf_grad=batch["leaf_sdf_grad"],
        leaf_curvature_mean=batch["leaf_curvature_mean"],
        leaf_curvature_gauss=batch["leaf_curvature_gauss"],
        leaf_centroid_norm=batch["leaf_centroid_norm"],
        leaf_pbd=batch["leaf_pbd"],
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        enc_feat, vit_feat = model.encode(
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

    # --- IDW for query points ---
    query_pos = t2["query_pos_norm"].unsqueeze(0).to(device, non_blocking=True)
    query_sdf = t2["query_sdf"].unsqueeze(0).to(device, non_blocking=True)
    query_sdf_grad = t2["query_sdf_grad"].unsqueeze(0).to(device, non_blocking=True)
    query_leaf_id = t2["query_leaf_id"].long()  # (N_q,)

    # Build per-query neighbor candidates from latent_neighbor_top64
    neighbor_top64 = pt["latent_neighbor_top64"]  # (L, 64) int32
    if isinstance(neighbor_top64, torch.Tensor):
        nbr = neighbor_top64.to(device)
    else:
        nbr = torch.from_numpy(np.asarray(neighbor_top64)).to(device)

    leaf_centroid_on_dev = batch["leaf_centroid_norm"].squeeze(0)  # (L, 3)

    query_leaf_id_dev = query_leaf_id.to(device)

    idw_idx, idw_w = gpu_idw(
        query_pos.squeeze(0),
        leaf_centroid_on_dev,
        nbr,
        query_leaf_id_dev,
        idw_k=8,
    )

    # --- Decoder ---
    n_qv = t2["n_query_vol"]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_vol, pred_surf = model.module.decoder(
            enc_feat, vit_feat,
            query_pos,
            query_sdf.to(torch.bfloat16),
            query_sdf_grad.to(torch.bfloat16),
            idw_idx.unsqueeze(0),
            idw_w.unsqueeze(0).to(torch.bfloat16),
            n_query_vol=n_qv,
        ) if hasattr(model, "module") else model.decoder(
            enc_feat, vit_feat,
            query_pos,
            query_sdf.to(torch.bfloat16),
            query_sdf_grad.to(torch.bfloat16),
            idw_idx.unsqueeze(0),
            idw_w.unsqueeze(0).to(torch.bfloat16),
            n_query_vol=n_qv,
        )

    # --- Z-score MSE (same as training loss) ---
    target_vol = t2["target_vol"].to(device).float()
    target_surf = t2["target_surf"].to(device).float()

    mse_vol = F.mse_loss(pred_vol.squeeze(0).float(), target_vol).item()
    mse_surf = F.mse_loss(pred_surf.squeeze(0).float(), target_surf).item()
    total_mse = mse_vol + mse_surf

    return {"vol_mse": mse_vol, "surf_mse": mse_surf, "total_mse": total_mse}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def eval_curve(model, case_pts: dict[str, dict], norm_stats: dict,
               epoch: int, device: torch.device, cfg: dict,
               n_query: int = 500_000) -> dict[str, float]:
    """Evaluate a set of cases (train_eval or val) at a given epoch.

    Parameters
    ----------
    model : HDB3DModel (possibly DDP-wrapped)
        Must be in eval mode.
    case_pts : dict mapping case_id -> pre-loaded PT dict (pinned).
    norm_stats : loaded norm_stats.json dict.
    epoch : current epoch number (affects transient sampling seeds).
    device : CUDA device for this rank.
    cfg : full config dict.
    n_query : number of query points per case.

    Returns
    -------
    dict with aggregated metrics:
        vol_mse, surf_mse, total_mse (mean across cases in this shard).
    """
    model.eval()
    vol_acc = 0.0
    surf_acc = 0.0
    n_cases = 0

    for case_id, pt in case_pts.items():
        metrics = _eval_one_case(
            model, pt, norm_stats, epoch, device, cfg, n_query)
        vol_acc += metrics["vol_mse"]
        surf_acc += metrics["surf_mse"]
        n_cases += 1
        torch.cuda.empty_cache()

    if n_cases == 0:
        return {"vol_mse": 0.0, "surf_mse": 0.0, "total_mse": 0.0}

    return {
        "vol_mse": vol_acc / n_cases,
        "surf_mse": surf_acc / n_cases,
        "total_mse": (vol_acc + surf_acc) / n_cases,
    }


def gather_curve_metrics(local_metrics: dict[str, float],
                         n_local: int,
                         device: torch.device) -> dict[str, float]:
    """All-reduce curve metrics across DDP ranks.

    Each rank provides its local mean metrics and count. Returns global mean
    on all ranks.
    """
    if not is_distributed():
        return local_metrics

    tensor = torch.tensor(
        [local_metrics["vol_mse"] * n_local,
         local_metrics["surf_mse"] * n_local,
         float(n_local)],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    vals = tensor.cpu().tolist()
    denom = max(vals[2], 1.0)
    return {
        "vol_mse": vals[0] / denom,
        "surf_mse": vals[1] / denom,
        "total_mse": (vals[0] + vals[1]) / denom,
    }
