"""Main training loop for HDB 3D urban wind CFD surrogate model.

Entry point:
    torchrun --nproc_per_node=4 hdb/training/loop.py --config hdb/config.yaml

Per-rank flow:
    init_ddp → load+pin train/val cases → build model → DDP wrap →
    torch.compile(reduce-overhead) → for epoch:
        curriculum sampling → build_grouped_shard → AsyncPrefetcher →
        for batch: BlockMask(CPU) → H2D → gpu_idw → forward(bf16) →
            loss → backward → clip → step
        val_curve(50 cases, 500K, z-score MSE)
        train_eval_curve(40 cases, same)
        SWA snapshot (epochs 300-399)
        checkpoint
    finalize: save swa_model.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from hdb.dataset.loaders import load_cases_pinned
from hdb.dataset.prefetcher import AsyncPrefetcher, prepare_one_case, stack_batch
from hdb.models import HDB3DModel
from hdb.models.bigbird import build_block_mask_direct
from hdb.models.idw import gpu_idw
from hdb.training.checkpoint import save_checkpoint, load_checkpoint
from hdb.training.curriculum import CurriculumScheduler
from hdb.training.ddp import init_ddp, cleanup_ddp, is_distributed
from hdb.training.shard import build_grouped_shard, BATCH_SIZES
from hdb.training.swa import SWAManager
from hdb.utils.memory import cpu_rss_gib, gpu_peak_gib
from hdb.utils.seed import seed_everything, per_case_epoch_seed

SUB_BIN_ORDER = [
    '0-19_easy', '0-19_hard', '20-39_easy', '20-39_hard',
    '40-59_easy', '40-59_hard', '60-79_easy', '60-79_hard',
    '80-123_easy', '80-123_hard',
]


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _move_batch_to_gpu(batch: dict[str, Any],
                       device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _build_optimizer_and_scheduler(
    model: torch.nn.Module, cfg: dict, world: int,
    steps_per_epoch_est: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    base_lr = float(cfg['training']['lr'])
    wd = float(cfg['training']['weight_decay'])
    effective_lr = base_lr * math.sqrt(world * 4)

    opt = AdamW(model.parameters(), lr=effective_lr,
                weight_decay=wd, fused=True)

    total_epochs = int(cfg['training']['num_epochs'])
    total_steps = max(1, total_epochs * steps_per_epoch_est)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def _estimate_steps_per_epoch(cases_per_bin: dict[str, list]) -> int:
    total = 0
    for sb, cids in cases_per_bin.items():
        B = BATCH_SIZES.get(sb, 1)
        total += (len(cids) + B - 1) // B
    return max(1, total)


@torch.no_grad()
def evaluate_split(
    compiled_model: Any,
    pt_data: dict,
    case_ids: list,
    epoch: int,
    device: torch.device,
) -> float:
    """500K sampled z-score MSE evaluation on a set of cases."""
    if not case_ids:
        return 0.0

    total_loss = 0.0
    n_cases = 0

    for cid in case_ids:
        pt = pt_data[cid]
        item = prepare_one_case(pt, cid, epoch)
        batch_cpu = stack_batch([item])

        L = item['L']
        R = 16
        n_qv = item['n_query_vol']

        flex_mask = build_block_mask_direct(
            batch_cpu['bigbird_key_idx'], L=L, R=R, device=device)
        batch = _move_batch_to_gpu(batch_cpu, device)

        idw_idx, idw_w = gpu_idw(
            batch['query_pos_norm'],
            batch['leaf_centroid_norm'],
            batch['latent_neighbor_top64'],
            batch['query_leaf_id'],
            idw_k=8)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_vol, pred_surf = compiled_model(
                leaf_centroid_norm=batch['leaf_centroid_norm'],
                leaf_stats=batch['leaf_stats'],
                leaf_sdf=batch['leaf_sdf'],
                leaf_sdf_grad=batch['leaf_sdf_grad'],
                leaf_curvature_mean=batch['leaf_curvature_mean'],
                leaf_curvature_gauss=batch['leaf_curvature_gauss'],
                leaf_pbd=batch['leaf_pbd'],
                transient1=batch['transient1'],
                query_pos_norm=batch['query_pos_norm'],
                query_sdf=batch['query_sdf'],
                query_sdf_grad=batch['query_sdf_grad'],
                idw_indices=idw_idx,
                idw_weights=idw_w,
                rope_cos=batch['rope_cos'],
                rope_sin=batch['rope_sin'],
                flex_mask=flex_mask,
                n_query_vol=n_qv,
            )
            loss_v = F.mse_loss(
                pred_vol, batch['query_target_volume'][:, :n_qv])
            loss_s = F.mse_loss(
                pred_surf, batch['query_target_surface'])
            total_loss += (loss_v + loss_s).item()
            n_cases += 1

    if is_distributed():
        t = torch.tensor([total_loss, float(n_cases)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t[0] / max(t[1], 1))
    return total_loss / max(n_cases, 1)


def train(cfg: dict) -> None:
    seed_everything(cfg.get('seed', 42))
    rank, world, local = init_ddp()
    device = torch.device('cuda', local)
    total_epochs = int(cfg['training']['num_epochs'])
    cache_dir = cfg['data']['cache_dir']

    if rank == 0:
        print(f'[init] world={world}, total_epochs={total_epochs}', flush=True)

    # ------------------------------------------------------------------ data
    manifest_path = cfg.get('data', {}).get('manifest_path')
    if manifest_path:
        manifest_path = str(Path(manifest_path).expanduser().resolve())
    else:
        manifest_path = os.path.join(cache_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_train_ids = manifest['splits']['train']
    my_train_ids = sorted(all_train_ids[rank::world])

    if rank == 0:
        print(f'[data] loading {len(my_train_ids)} train cases ...', flush=True)
    all_pt_data = load_cases_pinned(
        cache_dir, my_train_ids,
        num_workers=int(cfg['training'].get('num_workers', 30)))

    sub_bin_map: dict[str, str] = {}
    cases_per_bin: dict[str, list] = {}
    for cid in my_train_ids:
        sb = all_pt_data[cid]['sub_bin']
        sub_bin_map[cid] = sb
        cases_per_bin.setdefault(sb, []).append(cid)

    val_ids = manifest['splits']['val']
    my_val_ids = sorted(val_ids[rank::world])
    if rank == 0:
        print(f'[data] loading {len(my_val_ids)} val cases ...', flush=True)
    val_pt_data = load_cases_pinned(
        cache_dir, my_val_ids,
        num_workers=int(cfg['training'].get('num_workers', 30)))

    if rank == 0:
        print(f'[data] train={len(my_train_ids)} val={len(my_val_ids)} '
              f'(this rank)', flush=True)

    # ----------------------------------------------------------------- model
    model = HDB3DModel(cfg).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local])

    torch._dynamo.config.cache_size_limit = int(
        cfg['training'].get('cache_size_limit', 16))
    compile_mode = cfg['training'].get('compile_mode', 'reduce-overhead')
    try:
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
    except Exception as e:
        if rank == 0:
            print(f'[warn] torch.compile failed ({e}); using eager', flush=True)
        compiled = model

    # ------------------------------------------------------------- optimizer
    steps_per_epoch_est = _estimate_steps_per_epoch(cases_per_bin)
    opt, sched = _build_optimizer_and_scheduler(
        model, cfg, world, steps_per_epoch_est)

    # ------------------------------------------------------------ curriculum
    curriculum = CurriculumScheduler(
        SUB_BIN_ORDER, cases_per_bin,
        T_start=3.0, T_end=1.0, total_epochs=total_epochs)

    # ------------------------------------------------------------------ swa
    swa_window = int(cfg['training'].get('swa_window', 100))
    swa = SWAManager(swa_window=swa_window, num_epochs=total_epochs)

    # ------------------------------------------------------------ run dir
    run_dir = cfg.get('run_dir', 'runs/default')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ----------------------------------------------------------- resume
    start_epoch = 0
    resume_path = cfg.get('resume_checkpoint')
    if resume_path and os.path.exists(resume_path):
        start_epoch = load_checkpoint(resume_path, model, opt, sched)
        if rank == 0:
            print(f'[resume] epoch {start_epoch}', flush=True)

    if rank == 0:
        print(f'[train] steps_per_epoch≈{steps_per_epoch_est} '
              f'lr={opt.param_groups[0]["lr"]:.3e}', flush=True)

    # ============================================================ EPOCH LOOP
    for epoch in range(start_epoch, total_epochs):
        torch.cuda.reset_peak_memory_stats(device)
        rng_epoch = np.random.default_rng(cfg.get('seed', 42) + epoch)

        # 1. Curriculum sampling
        sampled_ids = curriculum.get_epoch_samples(epoch, rng_epoch)

        # 2. Build grouped shard (batches of same sub-bin)
        batches = build_grouped_shard(
            sampled_ids, sub_bin_map, epoch, rng_epoch)

        # 3. Prefetcher (builds BlockMask in background thread)
        prefetcher = AsyncPrefetcher(
            batches, all_pt_data, epoch,
            encoder_k=int(cfg['model'].get('encoder_k', 32)),
            n_query=int(cfg['training'].get('n_query', 125_000)),
            queue_size=int(cfg['training'].get('prefetch_queue_size', 4)),
            register_tokens=int(cfg['model'].get('register_tokens', 16)),
            mask_device=device,
        )

        epoch_loss_vol = 0.0
        epoch_loss_surf = 0.0
        n_steps = 0
        t_epoch = time.time()

        # ====================================================== STEP LOOP
        for batch_cpu in prefetcher:
            flex_mask = batch_cpu.pop('flex_mask')

            batch = _move_batch_to_gpu(batch_cpu, device)

            idw_idx, idw_w = gpu_idw(
                batch['query_pos_norm'],
                batch['leaf_centroid_norm'],
                batch['latent_neighbor_top64'],
                batch['query_leaf_id'],
                idw_k=int(cfg['model'].get('decoder_idw_k', 8)))

            n_qv = batch_cpu['n_query_vol']

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_vol, pred_surf = compiled(
                    leaf_centroid_norm=batch['leaf_centroid_norm'],
                    leaf_stats=batch['leaf_stats'],
                    leaf_sdf=batch['leaf_sdf'],
                    leaf_sdf_grad=batch['leaf_sdf_grad'],
                    leaf_curvature_mean=batch['leaf_curvature_mean'],
                    leaf_curvature_gauss=batch['leaf_curvature_gauss'],
                    leaf_pbd=batch['leaf_pbd'],
                    transient1=batch['transient1'],
                    query_pos_norm=batch['query_pos_norm'],
                    query_sdf=batch['query_sdf'],
                    query_sdf_grad=batch['query_sdf_grad'],
                    idw_indices=idw_idx,
                    idw_weights=idw_w,
                    rope_cos=batch['rope_cos'],
                    rope_sin=batch['rope_sin'],
                    flex_mask=flex_mask,
                    n_query_vol=n_qv,
                )

                loss_vol = F.mse_loss(
                    pred_vol, batch['query_target_volume'][:, :n_qv])
                loss_surf = F.mse_loss(
                    pred_surf, batch['query_target_surface'])
                loss = loss_vol + loss_surf

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg['training'].get('max_grad_norm', 1.0)))
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

            epoch_loss_vol += loss_vol.item()
            epoch_loss_surf += loss_surf.item()
            n_steps += 1

        prefetcher.close()

        # ================================================= EVAL CURVES
        val_loss = evaluate_split(
            compiled, val_pt_data, my_val_ids, epoch, device)

        # ======================================================== SWA
        swa.accumulate(model, epoch)

        # ====================================================== LOG
        if rank == 0:
            avg_total = (epoch_loss_vol + epoch_loss_surf) / max(n_steps, 1)
            avg_vol = epoch_loss_vol / max(n_steps, 1)
            avg_surf = epoch_loss_surf / max(n_steps, 1)
            T_curr = curriculum.temperature(epoch)
            print(
                f'[epoch {epoch:03d}] '
                f'loss={avg_total:.5f} (vol={avg_vol:.5f} surf={avg_surf:.5f}) '
                f'val={val_loss:.5f} '
                f'lr={sched.get_last_lr()[0]:.3e} T={T_curr:.2f} '
                f'steps={n_steps} '
                f'gpu={gpu_peak_gib(local):.1f}GiB '
                f'cpu={cpu_rss_gib():.1f}GiB '
                f't={time.time() - t_epoch:.1f}s',
                flush=True,
            )

        # ================================================= CHECKPOINT
        ckpt_every_pre = int(cfg['checkpoint'].get('every_epochs_pre_swa', 10))
        ckpt_every_swa = int(cfg['checkpoint'].get('every_epochs_swa', 5))
        swa_start = total_epochs - swa_window
        if rank == 0:
            do_ckpt = False
            if epoch < swa_start:
                do_ckpt = (epoch + 1) % ckpt_every_pre == 0
            else:
                do_ckpt = (epoch - swa_start) % ckpt_every_swa == 0
            if do_ckpt:
                save_checkpoint(
                    model, opt, sched, epoch,
                    os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pt'),
                    swa_manager=swa)

        if is_distributed():
            dist.barrier()

    # ========================================================= FINALIZE
    if rank == 0 and swa.has_snapshots():
        avg_sd = swa.get_averaged()
        swa_path = os.path.join(run_dir, 'swa_model.pt')
        torch.save(avg_sd, swa_path)
        print(f'[done] SWA model → {swa_path}', flush=True)

    if is_distributed():
        dist.barrier()
    cleanup_ddp()


def main(cfg: dict | None = None, resume_path: str | None = None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser(description='HDB training loop')
        parser.add_argument('--config', type=str, required=True,
                            help='Path to config.yaml')
        parser.add_argument('--run-dir', type=str, default=None,
                            help='Override run directory')
        parser.add_argument('--resume', type=str, default=None,
                            help='Checkpoint to resume from')
        args = parser.parse_args()
        cfg = _load_config(args.config)
        if args.run_dir is not None:
            cfg['run_dir'] = args.run_dir
        resume_path = args.resume

    if resume_path is not None:
        cfg['resume_checkpoint'] = resume_path

    train(cfg)


if __name__ == '__main__':
    main()
