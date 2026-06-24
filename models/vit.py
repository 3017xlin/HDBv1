"""12-layer ViT with BigBird sparse attention via FlexAttention (adapted for HDB).

- Pre-norm RMSNorm + multi-head attention with QK-RMSNorm + 3D RoPE.
- BigBird sparsity comes in as a prebuilt BlockMask (built by the caller,
  see models.bigbird.build_block_mask_direct). No mask_mod, no gather.
- 6 U-Net skip pairs (layer 0<->11, 1<->10, 2<->9, 3<->8, 4<->7, 5<->6).
- 16 register tokens appended after Encoder.

Changes from DrivAerML: drop_path_rate defaults to 0.0 (CUDA graph
compatible). The parameter is kept for flexibility but no DropPath is
instantiated when rate is 0.0.

RoPE cos/sin are also precomputed by the caller (prefetcher) and passed
through forward as tensors, so dynamo never traces the trig math.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import flex_attention

from .encoder import RMSNorm
from .rope import apply_rope


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int = 192, hidden: int = 768,
                 dropout: float = 0.0):
        super().__init__()
        # Fused gate+up Linear: one GEMM instead of two. At B=1 kernel
        # launch overhead is non-negligible, so this saves ~10-15% of
        # FFN time per layer.
        self.w_gate_up = nn.Linear(dim, 2 * hidden)
        self.w_down = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.drop(self.w_down(F.silu(gate) * up))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with QK-RMSNorm + 3D RoPE + FlexAttention BlockMask."""

    def __init__(self, dim: int = 192, num_heads: int = 3,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        # attn_dropout is unused with flex_attention (dropout via mod_fn
        # would graph-break the Triton lowering); kept for API parity.
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                flex_mask) -> torch.Tensor:
        """
        x:        (B, T, dim)        T = L + R (register tokens included)
        cos, sin: (B, T, head_dim)
        flex_mask: prebuilt FlexAttention BlockMask (BigBird sparsity)
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim
        Q = self.q_norm(self.q_proj(x).view(B, T, H, D))
        K = self.k_norm(self.k_proj(x).view(B, T, H, D))
        V = self.v_proj(x).view(B, T, H, D)

        Q = Q.permute(0, 2, 1, 3)                                              # (B, H, T, D)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        Q, K = apply_rope(Q, K, cos, sin)

        # RMSNorm promotes Q/K to fp32 for stability; apply_rope keeps fp32.
        # V never sees a norm so it stays bf16 under autocast. Align all three
        # to the same dtype (bf16 under autocast, fp32 in eager) -- required
        # by flex_attention's input validator.
        target_dtype = V.dtype
        if Q.dtype != target_dtype:
            Q = Q.to(target_dtype)
        if K.dtype != target_dtype:
            K = K.to(target_dtype)

        attn = flex_attention(Q, K, V, block_mask=flex_mask)

        attn = attn.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        return self.o_proj(attn)


class ViTBlock(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 3,
                 ffn_hidden: int = 768, ffn_dropout: float = 0.0,
                 attn_dropout: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, attn_dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden, dropout=ffn_dropout)
        # drop_path_rate=0.0 always for CUDA graph compatibility.
        # Parameter kept for flexibility but no DropPath instantiated at 0.0.
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                flex_mask) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, flex_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class ViT(nn.Module):
    """12-layer ViT with 6 U-Net skip pairs."""

    def __init__(self, dim: int = 192, num_layers: int = 12,
                 num_heads: int = 3, ffn_hidden: int = 768,
                 register_tokens: int = 16,
                 head_dim: int = 64,
                 ffn_dropout: float = 0.0, attn_dropout: float = 0.0,
                 drop_path_rate: float = 0.0,
                 rope_base: float = 100.0,
                 rope_dims: tuple[int, int, int] = (22, 22, 20)):
        super().__init__()
        assert num_layers % 2 == 0
        self.num_layers = num_layers
        self.register_tokens = register_tokens
        self.register = nn.Parameter(torch.randn(register_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, ffn_hidden, ffn_dropout, attn_dropout,
                     drop_path_rate=drop_path_rate)
            for _ in range(num_layers)
        ])
        self.skip_proj = nn.ModuleList([
            nn.Linear(2 * dim, dim) for _ in range(num_layers // 2)
        ])
        # Kept only for backward compatibility with set_rope_scale callers
        # (preprocess/training code) -- the actual forward path takes cos/sin
        # from outside, so this module is never invoked at compile time.
        from .rope import RotaryEmbedding3D
        self.rope = RotaryEmbedding3D(head_dim=head_dim, base=rope_base,
                                      rope_dims=rope_dims,
                                      register_tokens=register_tokens)
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.rope_dims = rope_dims
        self.final_norm = RMSNorm(dim)

    def forward(self, leaf_token: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                flex_mask) -> torch.Tensor:
        """
        leaf_token: (B, L, dim) bf16 -- output of encoder cross-attention
        rope_cos/rope_sin: (B, L+R, head_dim) -- precomputed by caller
        flex_mask: FlexAttention BlockMask

        Returns vit_features: (B, L, dim) bf16
        """
        B, L, dim = leaf_token.shape
        R = self.register_tokens
        reg = self.register[None].expand(B, R, dim).to(leaf_token.dtype)
        x = torch.cat([leaf_token, reg], dim=1)                                # (B, L+R, dim)
        cos = rope_cos.to(leaf_token.dtype)
        sin = rope_sin.to(leaf_token.dtype)

        skips: list[torch.Tensor] = []
        half = self.num_layers // 2
        for i in range(half):
            x = self.blocks[i](x, cos, sin, flex_mask)
            skips.append(x)
        for j in range(half):
            i = half + j
            x = self.blocks[i](x, cos, sin, flex_mask)
            mirror = skips[half - 1 - j]
            x = self.skip_proj[j](torch.cat([x, mirror], dim=-1))
        x = self.final_norm(x)
        return x[:, :L].contiguous()
