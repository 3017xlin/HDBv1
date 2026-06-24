"""BigBird sparse mask construction via FlexAttention (adapted for HDB).

Per query token: 64 local (from 1+2 order neighbor pool) + 16 register
(always) + 32 random (per-case-per-epoch) = 112 total keys.

Changes from DrivAerML: n_local default 64 (was 110).
"""
from __future__ import annotations

import numpy as np
import torch

# Flipped to True after the first build_block_mask_direct call dumps the
# BlockMask attribute names; keeps log noise minimal in steady state.
_BLOCK_MASK_INTROSPECTED = False


def build_bigbird_index(leaf_neighbor_idx: np.ndarray,
                        L: int, n_local: int = 64, n_register: int = 16,
                        n_random: int = 32, seed: int = 0
                        ) -> np.ndarray:
    """Return key index tensor of shape (L, n_local + n_register + n_random).

    Indices in [0, L+n_register). Register keys are L..L+n_register-1.
    Random keys are drawn from {0..L-1} minus the local set.
    """
    rng = np.random.default_rng(seed)
    K = n_local + n_register + n_random
    out = np.empty((L, K), dtype=np.int32)

    valid_mask = leaf_neighbor_idx != -1
    valid_counts = valid_mask.sum(axis=1)
    local_part = leaf_neighbor_idx[:, :n_local].copy()
    short = valid_counts < n_local
    if short.any():
        for q in np.where(short)[0]:
            vc = int(valid_counts[q])
            local_part[q, :vc] = leaf_neighbor_idx[q, valid_mask[q]][:vc]
            pool = np.setdiff1d(np.arange(L, dtype=np.int32), local_part[q, :vc])
            local_part[q, vc:] = rng.choice(pool, n_local - vc, replace=False)
    out[:, :n_local] = local_part

    out[:, n_local:n_local + n_register] = np.arange(L, L + n_register, dtype=np.int32)

    rand = rng.integers(0, L, size=(L, n_random + 32), dtype=np.int32)
    local_sorted = np.sort(local_part, axis=1)

    # Row-wise searchsorted: numpy's searchsorted only accepts 1D `a`, so
    # offset each row by i * M (M > any element) to make ranges disjoint,
    # flatten, do one global searchsorted, then subtract row bases.
    M = np.int64(local_sorted.max()) + np.int64(1)
    L_n, K_local = local_sorted.shape
    offsets = (np.arange(L_n, dtype=np.int64) * M)[:, None]
    flat_sorted = (local_sorted.astype(np.int64) + offsets).ravel()
    flat_rand = (rand.astype(np.int64) + offsets).ravel()
    pos_flat = np.searchsorted(flat_sorted, flat_rand)
    row_base = (np.arange(L_n, dtype=np.intp) * K_local)[:, None]
    pos = (pos_flat.reshape(rand.shape) - row_base).astype(np.intp)
    pos = np.clip(pos, 0, n_local - 1)

    hit = np.take_along_axis(local_sorted, pos, 1) == rand
    rand_clean = np.where(hit, L, rand)
    rand_clean.sort(axis=1)
    out[:, n_local + n_register:] = rand_clean[:, :n_random]

    return out


def build_flex_block_mask(key_idx: torch.Tensor, B: int, H: int,
                          L_with_reg: int, BLOCK_SIZE: int = 128):
    """[DEPRECATED -- kept for reference]

    Wraps key_idx as FlexAttention BlockMask via mask_mod + searchsorted.
    The searchsorted inside mask_mod cannot lower to Triton, so
    create_block_mask falls back to per-element materialization and OOMs
    on any non-trivial (B, L). Use build_block_mask_direct instead.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    L = key_idx.shape[-2]
    n_keys = key_idx.shape[-1]

    key_sorted = key_idx.sort(dim=-1).values.contiguous()

    def mask_mod(b, h, q_idx, kv_idx):
        is_reg_q = q_idx >= L
        is_reg_kv = kv_idx >= L
        q_safe = torch.where(q_idx < L, q_idx, torch.zeros_like(q_idx))
        row = key_sorted[b, q_safe]
        pos = torch.searchsorted(row, kv_idx.unsqueeze(-1)).squeeze(-1)
        pos_clamped = pos.clamp(max=n_keys - 1)
        found = row[pos_clamped] == kv_idx
        return is_reg_q | is_reg_kv | found

    block_mask = create_block_mask(
        mask_mod, B=B, H=H, Q_LEN=L_with_reg, KV_LEN=L_with_reg,
        BLOCK_SIZE=BLOCK_SIZE, device=key_idx.device,
    )
    return block_mask


@torch._dynamo.disable
def build_block_mask_direct(key_idx, L: int, R: int,
                            BLOCK_SIZE: int = 128,
                            device: torch.device | str | None = None):
    """Build a FlexAttention BlockMask without mask_mod.

    Why: create_block_mask + mask_mod compiles the predicate to a Triton
    kernel. Our predicate needs a per-query key-set lookup (searchsorted),
    which cannot lower to Triton and forces per-element materialization
    (OOM at 64 GiB). Instead we:

      1. Compute block-level (q_block, kv_block) adjacency on CPU with
         vectorized numpy (~10 ms / batch).
      2. Treat every adjacent pair as "full" -- every query in the q-block
         attends to every key in the kv-block. This over-attends relative
         to BigBird's exact 112-key pattern, but BigBird's sparsity is
         block-structured by design (local + random + register), and the
         block expansion costs <5x extra attention work vs ~400x for full
         dense attention.

    Wrapped in @torch._dynamo.disable so the numpy + BlockMask construction
    is treated as an opaque boundary; the surrounding model still compiles.

    Args:
        key_idx: (B, L, n_keys) int32/int64 -- usually on GPU
        L: number of leaves
        R: number of register tokens (their q/kv block is treated as
           attending all and attended-by-all)
        BLOCK_SIZE: FlexAttention block size

    Returns:
        BlockMask suitable for flex_attention(block_mask=...)
    """
    from torch.nn.attention.flex_attention import BlockMask

    if isinstance(key_idx, torch.Tensor):
        key_idx_np = key_idx.detach().cpu().numpy()
        if device is None:
            device = key_idx.device
    else:
        key_idx_np = np.asarray(key_idx)
        if device is None:
            device = torch.device('cpu')

    B = key_idx_np.shape[0]
    T = L + R
    n_blocks = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    reg_qb_start = L // BLOCK_SIZE                                       # first block touching register region

    # Per-leaf q-block id and per-key kv-block id.
    qb = (np.arange(L, dtype=np.int64) // BLOCK_SIZE)                    # (L,)
    kb = (np.clip(key_idx_np, 0, T - 1) // BLOCK_SIZE).astype(np.int64)  # (B, L, n_keys)

    # Set adj[b, q_block, kv_block] = True for every visited pair.
    adj = np.zeros((B, n_blocks, n_blocks), dtype=bool)
    bb = np.broadcast_to(np.arange(B, dtype=np.int64)[:, None, None], kb.shape)
    qq = np.broadcast_to(qb[None, :, None], kb.shape)
    adj[bb.ravel(), qq.ravel(), kb.ravel()] = True

    # Register q-blocks attend everywhere; register kv-blocks are
    # attended by everyone.
    adj[:, reg_qb_start:, :] = True
    adj[:, :, reg_qb_start:] = True

    num_blocks = adj.sum(axis=-1).astype(np.int32)                       # (B, n_blocks)
    # Pad indices to a fixed second-axis size so the resulting tensor
    # shape only depends on (B, n_blocks), never on the per-case max
    # visited-block count. Variable shape -> dynamo guard fail ->
    # recompile storm in the first few epochs. Trade ~50 KB extra GPU
    # memory per case for compile-time stability.
    max_kv = n_blocks
    indices = np.zeros((B, n_blocks, max_kv), dtype=np.int32)
    for b in range(B):
        for qbi in range(n_blocks):
            ks = np.flatnonzero(adj[b, qbi])
            indices[b, qbi, :ks.size] = ks

    # FlexAttention expects (B, H, Q_BLOCKS, ...). H=1 broadcasts.
    full_num_t = torch.from_numpy(num_blocks).unsqueeze(1).to(device)        # (B, 1, n_blocks)
    full_idx_t = torch.from_numpy(indices).unsqueeze(1).to(device)            # (B, 1, n_blocks, max_kv)

    # seq_lengths tells FlexAttention the actual (Q_LEN, KV_LEN); without
    # it the mask is assumed to cover n_blocks * BLOCK_SIZE positions
    # (513 * 128 = 65664), and flex_attention errors when the real Q/K
    # length is 65552.
    #
    # We pass our blocks via kv_num_blocks/kv_indices (the "partial-block"
    # slot) with NO mask_mod. PyTorch interprets missing mask_mod as
    # identity, so the effective semantics are the same as full_kv (every
    # position in an attended block is attended). full_kv_* hit a
    # backward-pass kernel bug in torch 2.6 that ILMA'd on certain
    # per-case adjacency patterns. The partial path takes a different
    # Triton kernel.
    try:
        bm = BlockMask.from_kv_blocks(
            kv_num_blocks=full_num_t,
            kv_indices=full_idx_t,
            BLOCK_SIZE=BLOCK_SIZE,
            seq_lengths=(T, T),
        )
    except TypeError:
        bm = BlockMask.from_kv_blocks(
            kv_num_blocks=full_num_t,
            kv_indices=full_idx_t,
            BLOCK_SIZE=BLOCK_SIZE,
            Q_LEN=T,
            KV_LEN=T,
        )

    # Defensive: some torch 2.6 patch releases accept seq_lengths kwarg but
    # don't actually write it onto the BlockMask object -- the kernel then
    # uses the inferred n_blocks*BLOCK_SIZE = 65664 instead of the real T
    # = 65552, walks past Q/K/V tensors during backward, and ILMAs.
    # On the first call, dump the BlockMask attribute names that look like
    # lengths so we can see what really got set; then force-write T into
    # any matching attribute.
    global _BLOCK_MASK_INTROSPECTED
    if not _BLOCK_MASK_INTROSPECTED:
        len_attrs = sorted(a for a in dir(bm)
                           if any(k in a.lower() for k in ('seq', 'len')))
        attr_vals = []
        for a in len_attrs:
            try:
                attr_vals.append(f'{a}={getattr(bm, a)!r}')
            except Exception:
                attr_vals.append(f'{a}=<unreadable>')
        print(f'[bigbird] BlockMask len/seq attrs: {", ".join(attr_vals)} '
              f'(expected T={T})', flush=True)
        _BLOCK_MASK_INTROSPECTED = True

    for q_attr, kv_attr in (('seq_lengths', None),       # tuple
                            ('Q_LEN', 'KV_LEN'),
                            ('q_len', 'kv_len'),
                            ('_q_length', '_kv_length')):
        if q_attr == 'seq_lengths' and hasattr(bm, 'seq_lengths'):
            try:
                object.__setattr__(bm, 'seq_lengths', (T, T))
            except Exception:
                pass
        elif kv_attr is not None and (hasattr(bm, q_attr) or hasattr(bm, kv_attr)):
            try:
                object.__setattr__(bm, q_attr, T)
                object.__setattr__(bm, kv_attr, T)
            except Exception:
                pass

    return bm


# ---------------------------------------------------------------------------
# Practical fallback used by ViT: gather K/V at the 112 selected positions
# per query, then run dense attention over that 112-key window. This is
# mathematically equivalent to FlexAttention BigBird, costs the same FLOPs,
# and is portable without PyTorch 2.5's experimental APIs.
# ---------------------------------------------------------------------------


def gather_kv_for_bigbird(K: torch.Tensor, V: torch.Tensor,
                          key_idx: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V at BigBird key positions.

    K, V         : (B, L_with_reg, H, head_dim)
    key_idx      : (B, L, n_keys) int32 -- for each query token q,
                   the L indices that q attends to (register & random).
    Returns:
      K_gather, V_gather : (B, L, n_keys, H, head_dim)
    """
    B, _, H, D = K.shape
    nq, nk = key_idx.shape[1], key_idx.shape[2]
    idx = key_idx.long().clamp(min=0)
    batch_ar = torch.arange(B, device=K.device)[:, None, None]
    K_g = K[batch_ar, idx]                                                 # (B, L, n_keys, H, D)
    V_g = V[batch_ar, idx]
    return K_g, V_g
