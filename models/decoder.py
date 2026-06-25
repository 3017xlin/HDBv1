"""Decoder: IDW=8 + Fourier MLP + dual FiLM + dual heads (adapted for HDB).

Changes from DrivAerML:
- vol_out=5 (was 8), surf_out=1 (was 4)
- n_query_vol passed as runtime parameter (not self.n_query_vol)
- decode_chunk method for test inference
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEncoding(nn.Module):
    """sin/cos encoding with 10 frequencies on a 7-dim input.

    Output dimensionality: 7 + 7 * 2 * 10 = 147.
    """

    def __init__(self, n_freqs: int = 10, in_dim: int = 7):
        super().__init__()
        # freqs = (2^k) * pi for k = 0..n_freqs-1
        freqs = (2.0 ** torch.arange(n_freqs).float()) * math.pi
        self.register_buffer('freqs', freqs)                               # (n_freqs,)
        self.out_dim = in_dim + 2 * n_freqs * in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)
        theta = x[..., None] * self.freqs                                  # (B, N, in_dim, n_freqs)
        s = torch.sin(theta)
        c = torch.cos(theta)
        feats = torch.cat([s, c], dim=-1).reshape(
            *x.shape[:-1], -1)                                             # (B, N, 2*nfreq*in_dim)
        return torch.cat([x, feats], dim=-1)


class Decoder(nn.Module):
    """IDW gather + Fourier + pos MLP + dual FiLM + dual heads."""

    def __init__(self, dim: int = 192, idw_k: int = 8,
                 fourier_freqs: int = 10,
                 pos_hidden: int = 256, pos_out: int = 512,
                 vol_out: int = 5, surf_out: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.idw_k = idw_k
        self.fourier = FourierEncoding(n_freqs=fourier_freqs, in_dim=7)
        self.pos_lin1 = nn.Linear(self.fourier.out_dim, pos_hidden)
        self.pos_lin2 = nn.Linear(pos_hidden, pos_out)
        self.film_enc = nn.Linear(dim, 2 * pos_out)
        self.film_vit = nn.Linear(dim, 2 * pos_out)
        self.volume_head = nn.Linear(pos_out, vol_out)
        self.surface_head = nn.Linear(pos_out, surf_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, enc_feat: torch.Tensor, vit_feat: torch.Tensor,
                query_pos_norm: torch.Tensor, query_sdf: torch.Tensor,
                query_sdf_grad: torch.Tensor,
                idw_indices: torch.Tensor, idw_weights: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        enc_feat, vit_feat: (B, L, dim)
        query_pos_norm:     (B, N_q, 3)   N_q is the SAME (= cfg.training.n_query)
                                          for every case in the batch.  Volume,
                                          surface, and padding slots are
                                          distinguished by the caller's
                                          `query_is_surf` / `query_valid_mask`
                                          and applied via the loss, not here.
        query_sdf:          (B, N_q)
        query_sdf_grad:     (B, N_q, 3)
        idw_indices:        (B, N_q, idw_k) int32 leaf ids
        idw_weights:        (B, N_q, idw_k) fp32

        Returns (pred_vol [B, N_q, 5], pred_surf [B, N_q, 1]).
        Both heads predict on every slot; only the relevant slots feed
        the loss (the other head's prediction at that slot is discarded).
        """
        B, L, D = enc_feat.shape
        N_q = query_pos_norm.shape[1]
        k = self.idw_k
        idx = idw_indices.long().clamp(min=0)
        batch_ar = torch.arange(B, device=enc_feat.device)[:, None, None]
        enc_gather = enc_feat[batch_ar, idx]                               # (B, N_q, k, D)
        vit_gather = vit_feat[batch_ar, idx]
        w = idw_weights.unsqueeze(-1).to(enc_gather.dtype)                  # (B, N_q, k, 1)
        enc_interp = (enc_gather * w).sum(dim=-2)                          # (B, N_q, D)
        vit_interp = (vit_gather * w).sum(dim=-2)

        fourier_in = torch.cat([
            query_pos_norm.to(enc_feat.dtype),
            query_sdf.to(enc_feat.dtype).unsqueeze(-1),
            query_sdf_grad.to(enc_feat.dtype),
        ], dim=-1)                                                         # (B, N_q, 7)
        feats = self.fourier(fourier_in)                                   # (B, N_q, 147)
        pos = self.pos_lin1(feats)
        pos = F.relu(pos)
        pos = self.drop(pos)
        pos = self.pos_lin2(pos)                                           # (B, N_q, pos_out)

        g1, b1 = self.film_enc(enc_interp).chunk(2, dim=-1)
        pos = pos * (1.0 + g1) + b1
        g2, b2 = self.film_vit(vit_interp).chunk(2, dim=-1)
        pos = pos * (1.0 + g2) + b2

        pred_vol = self.volume_head(pos)                                   # (B, N_q, 5)
        pred_surf = self.surface_head(pos)                                 # (B, N_q, 1)
        return pred_vol, pred_surf
