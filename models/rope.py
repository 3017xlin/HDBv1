"""3D RoPE with per-axis Nyquist scale (v4 §8).

Positions are leaf_centroid_norm in [-1, 1]^3 (already normalized in
preprocess; no second-pass normalization). rope_scale_per_axis comes
from coef_norm.pt (derived from bounding-box extent / geo mean).

cos/sin are stored at HALF head_dim and consumed by apply_rope without
an interleave: rotation pairs adjacent dims, so each pair shares one
cos/sin value. Halves the precompute memory and skips a repeat_interleave
per axis (was ~5-8 ms CPU per case).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding3D(nn.Module):
    """Kept for backwards compatibility with set_rope_scale callers. The
    actual cos/sin are precomputed externally (see precompute_rope_3d);
    this module is never invoked at compile time.
    """

    def __init__(self, head_dim: int = 64, base: float = 100.0,
                 rope_dims: tuple[int, int, int] = (22, 22, 20),
                 register_tokens: int = 16):
        super().__init__()
        assert sum(rope_dims) == head_dim, (
            f'rope_dims {rope_dims} must sum to head_dim {head_dim}')
        self.head_dim = head_dim
        self.base = base
        self.dims = rope_dims
        self.register_tokens = register_tokens
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._scale_z = 1.0

    def set_rope_scale(self, rope_scale_per_axis) -> None:
        """Cache rope scale as plain floats. Accepts torch.Tensor or
        numpy.ndarray (preprocess writes numpy)."""
        flat = torch.as_tensor(rope_scale_per_axis).reshape(-1)
        self._scale_x = float(flat[0])
        self._scale_y = float(flat[1])
        self._scale_z = float(flat[2])


def precompute_rope_3d(leaf_centroid_norm: torch.Tensor,
                       head_dim: int,
                       base: float,
                       rope_dims: tuple[int, int, int],
                       scale: tuple[float, float, float],
                       register_tokens: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-function 3D RoPE precompute (no nn.Module state).

    Used by the prefetcher to compute (cos, sin) per case on the CPU so
    the compiled model.forward never has to trace the trig math.

    Returns cos/sin of shape (L + register_tokens, head_dim // 2) -- half
    width, since apply_rope pairs adjacent dims and each pair shares one
    cos/sin value. No repeat_interleave needed.
    """
    assert sum(rope_dims) == head_dim
    assert all(d % 2 == 0 for d in rope_dims), (
        f'rope_dims {rope_dims} must each be even')
    dx, dy, dz = rope_dims
    sx, sy, sz = scale
    half_dim = head_dim // 2

    def _axis(pos: torch.Tensor, dim: int, s: float
              ) -> tuple[torch.Tensor, torch.Tensor]:
        half = dim // 2
        idx = torch.arange(half, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (2.0 * idx / dim))                     # (half,)
        theta = pos[:, None] * inv_freq * s                              # (L, half)
        return torch.cos(theta), torch.sin(theta)                        # (L, half)

    cx, sx_ = _axis(leaf_centroid_norm[..., 0], dx, sx)
    cy, sy_ = _axis(leaf_centroid_norm[..., 1], dy, sy)
    cz, sz_ = _axis(leaf_centroid_norm[..., 2], dz, sz)
    cos = torch.cat([cx, cy, cz], dim=-1)                                # (L, half_dim)
    sin = torch.cat([sx_, sy_, sz_], dim=-1)
    # Register tokens get identity rotation: cos=1, sin=0.
    reg_cos = torch.ones(register_tokens, half_dim, dtype=cos.dtype)
    reg_sin = torch.zeros(register_tokens, half_dim, dtype=sin.dtype)
    cos = torch.cat([cos, reg_cos], dim=0)                               # (L+R, half_dim)
    sin = torch.cat([sin, reg_sin], dim=0)
    return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos_half: torch.Tensor, sin_half: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary to (B, H, T, D) Q/K with half-dim cos/sin (B, T, D//2)."""
    cos_half = cos_half[:, None]                                         # (B, 1, T, D//2)
    sin_half = sin_half[:, None]
    return _rotate(q, cos_half, sin_half), _rotate(k, cos_half, sin_half)


def _rotate(x: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor
            ) -> torch.Tensor:
    """Pair-wise rotate adjacent dims using half-dim cos/sin (one value per pair).

    (x0, x1) -> (x0 cos - x1 sin, x1 cos + x0 sin)
    """
    x_even = x[..., 0::2]                                                # (..., D//2)
    x_odd = x[..., 1::2]
    rot_e = x_even * cos_half - x_odd * sin_half
    rot_o = x_odd * cos_half + x_even * sin_half
    return torch.stack([rot_e, rot_o], dim=-1).flatten(-2)
