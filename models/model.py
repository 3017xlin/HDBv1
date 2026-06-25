"""Top-level HDB3DModel: Encoder -> ViT -> Decoder (adapted from DrivAerML).

forward() and encode() take explicit kwargs (not a batch dict) so dynamo
can compile cleanly without dict membership / get / mutate boundaries.
The caller (training/loop.py, evaluation/curve.py, evaluation/test_eval.py)
is responsible for:
  * IDW (gpu_idw)
  * RoPE cos/sin precompute (prefetcher)
  * BigBird BlockMask (build_block_mask_direct)
and passes them in as kwargs.

Key changes from DrivAerML:
- Encoder q_in_dim=46 (adds PBD 16-dim), kv_hidden=64
- leaf_pbd as explicit forward parameter
- Decoder vol_out=5, surf_out=1
- n_query_vol passed as runtime arg (not self.n_query_vol)
- drop_path_rate=0.0 always (CUDA graph compatible)
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import EncoderCrossAttention, build_leaf_aggregate
from .vit import ViT


class HDB3DModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        m = cfg['model']
        reg = cfg.get('regularization', {})
        self.cfg = cfg
        self.encoder = EncoderCrossAttention(
            q_in_dim=46, kv_in_dim=10,
            dim=int(m['latent_dim']),
            num_heads=int(m['num_heads']),
            head_dim=int(m['head_dim']),
            q_hidden=64,
            kv_hidden=64,
        )
        self.vit = ViT(
            dim=int(m['latent_dim']),
            num_layers=int(m['num_layers']),
            num_heads=int(m['num_heads']),
            head_dim=int(m['head_dim']),
            ffn_hidden=int(m['ffn_hidden']),
            register_tokens=int(m['register_tokens']),
            ffn_dropout=float(reg.get('ffn_dropout', 0.0)),
            attn_dropout=float(reg.get('attn_dropout', 0.0)),
            drop_path_rate=0.0,
            rope_base=float(m['rope_base']),
            rope_dims=tuple(m['rope_dims']),
        )
        self.decoder = Decoder(
            dim=int(m['latent_dim']),
            idw_k=int(m['decoder_idw_k']),
            fourier_freqs=int(m['decoder_fourier_freqs']),
            pos_hidden=int(m['decoder_pos_hidden']),
            pos_out=int(m['decoder_pos_out']),
            vol_out=5, surf_out=1,
            dropout=float(reg.get('decoder_dropout', 0.0)),
        )

    # ------------------------------------------------------------------
    # Forward (training / curve)
    # ------------------------------------------------------------------

    def forward(self,
                leaf_centroid_norm: torch.Tensor,    # (B, L, 3)
                leaf_stats: torch.Tensor,            # (B, L, 21)
                leaf_sdf: torch.Tensor,              # (B, L)
                leaf_sdf_grad: torch.Tensor,         # (B, L, 3)
                leaf_curvature_mean: torch.Tensor,   # (B, L)
                leaf_curvature_gauss: torch.Tensor,  # (B, L)
                leaf_pbd: torch.Tensor,              # (B, L, 16)
                transient1: torch.Tensor,            # (B, L, 32, 10)
                query_pos_norm: torch.Tensor,        # (B, N_q, 3)
                query_sdf: torch.Tensor,             # (B, N_q)
                query_sdf_grad: torch.Tensor,        # (B, N_q, 3)
                idw_indices: torch.Tensor,           # (B, N_q, 8)
                idw_weights: torch.Tensor,           # (B, N_q, 8)
                rope_cos: torch.Tensor,              # (B, L+R, 32)
                rope_sin: torch.Tensor,              # (B, L+R, 32)
                flex_mask,                           # BlockMask
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both heads predict on every query slot (B, N_q, *).  The
        caller is responsible for masking the loss by query_is_surf /
        query_valid_mask — see training/loop.py."""
        leaf_aggr = build_leaf_aggregate(
            leaf_stats=leaf_stats,
            leaf_sdf=leaf_sdf,
            leaf_sdf_grad=leaf_sdf_grad,
            leaf_curvature_mean=leaf_curvature_mean,
            leaf_curvature_gauss=leaf_curvature_gauss,
            leaf_centroid_norm=leaf_centroid_norm,
            leaf_pbd=leaf_pbd,
        )
        leaf_token = self.encoder(leaf_aggr, transient1)
        vit_feat = self.vit(leaf_token, rope_cos, rope_sin, flex_mask)
        pred_vol, pred_surf = self.decoder(
            leaf_token, vit_feat,
            query_pos_norm, query_sdf, query_sdf_grad,
            idw_indices, idw_weights,
        )
        return pred_vol, pred_surf

    # ------------------------------------------------------------------
    # Test inference helpers (encoder run once; decoder chunked)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self,
               leaf_centroid_norm: torch.Tensor,    # (B, L, 3)
               leaf_stats: torch.Tensor,            # (B, L, 21)
               leaf_sdf: torch.Tensor,              # (B, L)
               leaf_sdf_grad: torch.Tensor,         # (B, L, 3)
               leaf_curvature_mean: torch.Tensor,   # (B, L)
               leaf_curvature_gauss: torch.Tensor,  # (B, L)
               leaf_pbd: torch.Tensor,              # (B, L, 16)
               transient1: torch.Tensor,            # (B, L, 32, 10)
               rope_cos: torch.Tensor,              # (B, L+R, 32)
               rope_sin: torch.Tensor,              # (B, L+R, 32)
               flex_mask,                           # BlockMask
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder + ViT once; return (enc_feat, vit_feat) (B, L, D)."""
        leaf_aggr = build_leaf_aggregate(
            leaf_stats=leaf_stats,
            leaf_sdf=leaf_sdf,
            leaf_sdf_grad=leaf_sdf_grad,
            leaf_curvature_mean=leaf_curvature_mean,
            leaf_curvature_gauss=leaf_curvature_gauss,
            leaf_centroid_norm=leaf_centroid_norm,
            leaf_pbd=leaf_pbd,
        )
        leaf_token = self.encoder(leaf_aggr, transient1)
        vit_feat = self.vit(leaf_token, rope_cos, rope_sin, flex_mask)
        return leaf_token, vit_feat

    @torch.no_grad()
    def decode_chunk(self, enc_feat: torch.Tensor, vit_feat: torch.Tensor,
                     query_pos_norm: torch.Tensor, query_sdf: torch.Tensor,
                     query_sdf_grad: torch.Tensor,
                     idw_indices: torch.Tensor, idw_weights: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """For test inference: decode a chunk of queries.  Both heads
        run on all slots; the caller selects pred_vol or pred_surf based
        on which kind of chunk was passed in."""
        return self.decoder(enc_feat, vit_feat, query_pos_norm, query_sdf,
                            query_sdf_grad, idw_indices, idw_weights)
