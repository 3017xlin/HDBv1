"""GPU IDW (Inverse Distance Weighting) via torch.topk over neighbor candidates.

Each query point belongs to a leaf (query_leaf_id). That leaf's top-64
latent neighbors (latent_neighbor_top64) become the candidate set.
IDW picks the closest idw_k from those 64.

Kept outside torch.compile boundary (dynamic shapes).
"""
from __future__ import annotations

import torch


def gpu_idw(query_pos_norm: torch.Tensor,
            leaf_centroid_norm: torch.Tensor,
            latent_neighbor_top64: torch.Tensor,
            query_leaf_id: torch.Tensor,
            idw_k: int = 8
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU IDW=k via torch.topk over each query's candidate leaves.

    Args:
        query_pos_norm:        (B, N_q, 3) or (N_q, 3)
        leaf_centroid_norm:    (B, L, 3) or (L, 3)
        latent_neighbor_top64: (B, L, 64) or (L, 64) int — per-leaf neighbor indices
        query_leaf_id:         (B, N_q) or (N_q,) int — which leaf each query belongs to
        idw_k:                 number of nearest leaves to keep

    Returns (idw_indices, idw_weights) with matching leading dims.
    """
    if query_pos_norm.ndim == 3:
        return _gpu_idw_batched(query_pos_norm, leaf_centroid_norm,
                                latent_neighbor_top64, query_leaf_id, idw_k)
    return _gpu_idw_single(query_pos_norm, leaf_centroid_norm,
                           latent_neighbor_top64, query_leaf_id, idw_k)


def _gpu_idw_single(query_pos_norm: torch.Tensor,
                    leaf_centroid_norm: torch.Tensor,
                    latent_neighbor_top64: torch.Tensor,
                    query_leaf_id: torch.Tensor,
                    idw_k: int = 8
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-sample GPU IDW (no batch dim).

    query_pos_norm:        (N_q, 3)
    leaf_centroid_norm:    (L, 3)
    latent_neighbor_top64: (L, 64)
    query_leaf_id:         (N_q,) int
    """
    cands = latent_neighbor_top64[query_leaf_id.long()]                    # (N_q, 64)
    valid = cands != -1
    safe = cands.clamp(min=0).long()
    cand_c = leaf_centroid_norm[safe]                                      # (N_q, 64, 3)
    diff = query_pos_norm[:, None, :] - cand_c
    d2 = diff.pow(2).sum(-1)
    d2 = torch.where(valid, d2, torch.full_like(d2, float('inf')))
    top_d2, top = torch.topk(d2, k=idw_k, dim=1, largest=False)
    idw_idx = torch.gather(cands, 1, top).to(torch.int32)
    w = 1.0 / (top_d2.sqrt() + 1e-8)
    w = w / w.sum(dim=1, keepdim=True)
    return idw_idx, w.to(torch.float32)


def _gpu_idw_batched(query_pos_norm: torch.Tensor,
                     leaf_centroid_norm: torch.Tensor,
                     latent_neighbor_top64: torch.Tensor,
                     query_leaf_id: torch.Tensor,
                     idw_k: int = 8
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched GPU IDW.

    query_pos_norm:        (B, N_q, 3)
    leaf_centroid_norm:    (B, L, 3)
    latent_neighbor_top64: (B, L, 64)
    query_leaf_id:         (B, N_q) int
    """
    B, N_q, _ = query_pos_norm.shape
    batch_ar = torch.arange(B, device=query_leaf_id.device)[:, None]
    cands = latent_neighbor_top64[batch_ar, query_leaf_id.long()]          # (B, N_q, 64)
    valid = cands != -1
    safe = cands.clamp(min=0).long()
    batch_ar3 = torch.arange(B, device=cands.device)[:, None, None]
    cand_c = leaf_centroid_norm[batch_ar3, safe]                           # (B, N_q, 64, 3)
    diff = query_pos_norm[:, :, None, :] - cand_c
    d2 = diff.pow(2).sum(-1)                                              # (B, N_q, 64)
    d2 = torch.where(valid, d2, torch.full_like(d2, float('inf')))
    top_d2, top = torch.topk(d2, k=idw_k, dim=-1, largest=False)
    idw_idx = torch.gather(cands, -1, top).to(torch.int32)
    w = 1.0 / (top_d2.sqrt() + 1e-8)
    w = w / w.sum(dim=-1, keepdim=True)
    return idw_idx, w.to(torch.float32)
