"""Async producer-consumer prefetcher for HDB training.

Each rank owns one AsyncPrefetcher. A background thread pool builds the
next batch (transient1 from encoder_pool, Gumbel-top-k query sampling,
BigBird key indices, precomputed RoPE). A bounded queue keeps several
batches ready so the GPU never stalls on CPU work.

All heavy precomputed fields (encoder_pool, bigbird_fixed, rope_cos/sin,
point_leaf_id, vol_log_sample_weight) are read directly from the pinned
PT dicts — the prefetcher only does lightweight index/sample operations.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any

import numpy as np
import torch

from models.bigbird import build_block_mask_direct
from training.transient import build_transient1, build_transient2
from utils.seed import per_case_epoch_seed


def _tensor_to_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            return x.to(torch.float32).numpy()
        return x.numpy()
    return x


def _ensure_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def stack_batch(batch_items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack a list of single-case dicts into a batched dict.

    Tensor and ndarray fields with identical shape are stacked along a new
    leading B dimension. Scalars become a 1-d int32 tensor.
    """
    out: dict[str, Any] = {}
    keys = batch_items[0].keys()
    for k in keys:
        v0 = batch_items[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([item[k] for item in batch_items], dim=0)
        elif isinstance(v0, np.ndarray):
            out[k] = torch.stack(
                [torch.from_numpy(item[k]) for item in batch_items], dim=0)
        elif isinstance(v0, (int, np.integer)):
            out[k] = torch.tensor(
                [int(item[k]) for item in batch_items], dtype=torch.int32)
        else:
            out[k] = v0
    return out


def build_bigbird_index_fast(bigbird_fixed: torch.Tensor | np.ndarray,
                             L: int, n_random: int = 32,
                             seed: int | None = None) -> np.ndarray:
    """Append 32 random keys to the precomputed 64-local + 16-register block.

    Args:
        bigbird_fixed: (L, 80) int32 — precomputed local + register indices.
        L: number of latent tokens (random keys drawn from [0, L)).
        n_random: number of random attention keys per query.
        seed: deterministic seed for the random keys.

    Returns:
        (L, 112) int32 array — full BigBird key index per query.
    """
    rng = np.random.RandomState(seed)
    random_keys = rng.randint(0, L, size=(L, n_random)).astype(np.int32)
    fixed_np = _tensor_to_np(bigbird_fixed).astype(np.int32)
    return np.concatenate([fixed_np, random_keys], axis=1)


# (Removed _trim_queries_to_nqv: build_transient2 now returns full
#  [n_query, ...] arrays padded to a fixed length, with `query_is_surf`
#  and `query_valid_mask` distinguishing vol / surf / padding.  No per-
#  batch trimming is needed — every per-case tensor already has shape
#  [n_query, *] and stacks cleanly into [B, n_query, *].)


def prepare_one_case(case_pt: dict[str, Any], case_id: int, epoch: int,
                     encoder_k: int = 32,
                     n_query: int = 500_000) -> dict[str, Any]:
    """Build all per-case CPU tensors for one training step.

    Leverages six precomputed fields in the PT dict so that the only real
    work is: (a) sampling 32-of-256 for transient1, (b) Gumbel-top-k for
    volume queries, (c) generating 32 random BigBird keys.
    """
    case_pt['_case_id'] = case_id

    t1 = build_transient1(case_pt, epoch, encoder_k=encoder_k)
    t2 = build_transient2(case_pt, epoch, n_query=n_query)

    seed_bb = per_case_epoch_seed(case_id, epoch) ^ 0xBB17_BB17
    L = int(case_pt['L'])
    key_idx = build_bigbird_index_fast(
        case_pt['bigbird_fixed'], L, n_random=32, seed=seed_bb)

    out: dict[str, Any] = {
        'leaf_centroid_norm': case_pt['leaf_centroid_norm'],
        'leaf_stats': case_pt['leaf_stats'],
        'leaf_sdf': case_pt['leaf_sdf'],
        'leaf_sdf_grad': case_pt['leaf_sdf_grad'],
        'leaf_curvature_mean': case_pt['leaf_curvature_mean'],
        'leaf_curvature_gauss': case_pt['leaf_curvature_gauss'],
        'leaf_pbd': case_pt['leaf_pbd'],
        'latent_neighbor_top64': case_pt['latent_neighbor_top64'],
        'transient1': torch.from_numpy(t1).to(torch.bfloat16),
        'query_pos_norm': torch.from_numpy(t2['query_pos_norm']),
        'query_sdf': torch.from_numpy(t2['query_sdf']).to(torch.bfloat16),
        'query_sdf_grad': torch.from_numpy(
            t2['query_sdf_grad']).to(torch.bfloat16),
        'query_leaf_id': torch.from_numpy(t2['query_leaf_id']),
        'query_target_volume': torch.from_numpy(
            t2['query_target_volume']).to(torch.bfloat16),
        'query_target_surface': torch.from_numpy(
            t2['query_target_surface']).to(torch.bfloat16),
        'query_is_surf': torch.from_numpy(t2['query_is_surf']),       # (n_query,) bool
        'query_valid_mask': torch.from_numpy(t2['query_valid_mask']),  # (n_query,) bool
        'bigbird_key_idx': torch.from_numpy(key_idx),
        'rope_cos': _ensure_tensor(case_pt['rope_cos']),
        'rope_sin': _ensure_tensor(case_pt['rope_sin']),
        'n_query_vol': int(t2['n_query_vol']),
        'n_query_surf': int(t2['n_query_surf']),
        'L': L,
    }
    return out


class AsyncPrefetcher:
    """Producer-consumer pipeline owned by one DDP rank.

    A daemon thread prepares batches in the background using a small
    ``ThreadPoolExecutor`` (cases within a batch are built concurrently).
    The main training thread consumes ready batches from a bounded queue.

    Args:
        batches: ordered list of ``(batch_case_ids, sub_bin, L, B)`` tuples
            produced by :func:`hdb.training.shard.build_grouped_shard`.
        all_pt_data: rank-local dict mapping case_id → pinned PT dict.
        epoch: current epoch (for deterministic RNG seeds).
        encoder_k: number of encoder neighbours to sample (default 32).
        n_query: total query points per case (default 500 000).
        num_workers: thread pool size for parallel per-case preparation.
        queue_size: max batches buffered ahead of the consumer.
    """

    def __init__(self, batches: list[tuple], all_pt_data: dict,
                 epoch: int, *,
                 encoder_k: int = 32,
                 n_query: int = 500_000,
                 num_workers: int = 4,
                 queue_size: int = 4,
                 register_tokens: int = 16,
                 mask_device: torch.device | str = 'cuda'):
        self.batches = list(batches)
        self.all_pt_data = all_pt_data
        self.epoch = epoch
        self.encoder_k = encoder_k
        self.n_query = n_query
        self.num_workers = num_workers
        self.register_tokens = register_tokens
        self.mask_device = mask_device
        self.queue: Queue[dict[str, Any] | None] = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._bg = threading.Thread(target=self._run, daemon=True)
        self._bg.start()

    def _build_one(self, case_id: int) -> dict[str, Any]:
        pt = self.all_pt_data[case_id]
        return prepare_one_case(
            pt, case_id, self.epoch,
            encoder_k=self.encoder_k,
            n_query=self.n_query,
        )

    def _run(self) -> None:
        try:
            pool = ThreadPoolExecutor(
                max_workers=min(self.num_workers, 8))
            for batch_case_ids, sub_bin, L, B in self.batches:
                if self._stop.is_set():
                    break

                if len(batch_case_ids) > 1:
                    items = list(pool.map(self._build_one, batch_case_ids))
                else:
                    items = [self._build_one(batch_case_ids[0])]

                # All per-case query tensors are already shaped
                # [n_query, *] (see training/transient.py build_transient2),
                # so stacking is just torch.stack — no per-batch trim.
                for item in items:
                    item['L'] = L
                    item['sub_bin'] = sub_bin

                batch = stack_batch(items)
                batch['L'] = L
                batch['sub_bin'] = sub_bin

                R = self.register_tokens
                flex_mask = build_block_mask_direct(
                    batch['bigbird_key_idx'], L=L, R=R,
                    device=self.mask_device)
                T = L + R
                assert flex_mask is not None, (
                    f"BlockMask construction returned None for L={L}")
                batch['flex_mask'] = flex_mask

                self.queue.put(batch)
            pool.shutdown(wait=False)
        finally:
            self.queue.put(None)

    def __iter__(self) -> AsyncPrefetcher:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch

    def __len__(self) -> int:
        return len(self.batches)

    def close(self) -> None:
        self._stop.set()
