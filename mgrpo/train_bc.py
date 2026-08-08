"""BC 预训练：人类牌谱监督学习，为 GRPO 提供策略起点"""
import argparse
import glob
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mgrpo.prelude  # noqa: E402  注入 mortal/ 路径，使 libriichi.pyd 可导入

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from mgrpo.config import ENV, MODEL, PATHS
from mgrpo.data.mjson import iter_human_games
from mgrpo.model.brain import PolicyNet

log = logging.getLogger(__name__)


def iter_batches(files: list[Path], batch_size: int):
    """流式展开整局为 per-step 样本，攒满一个 batch 即产出 CPU tensor"""
    obs_buf, act_buf, mask_buf = [], [], []
    for game in iter_human_games(files):
        for i in range(len(game.actions)):
            obs_buf.append(game.obs[i])
            act_buf.append(game.actions[i])
            mask_buf.append(game.masks[i])
            if len(obs_buf) == batch_size:
                yield (
                    torch.as_tensor(np.stack(obs_buf)),
                    torch.as_tensor(np.stack(act_buf)),
                    torch.as_tensor(np.stack(mask_buf)),
                )
                obs_buf, act_buf, mask_buf = [], [], []


class FileIter(IterableDataset):
    """按 worker 分片文件列表，无限产出 batch；shuffle 在每轮开头"""
    def __init__(self, files: list[Path], batch_size: int):
        self.files = files
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = (
            self.files[worker_info.id :: worker_info.num_workers]
            if worker_info is not None
            else self.files
        )
        while True:
            random.shuffle(files)
            yield from iter_batches(files, self.batch_size)


def _worker_init(_):
    torch.set_num_threads(1)  # 多 worker 各占全核会互相拖慢


def _identity_collate(batch):
    return batch  # batch_size=None 时 collate 直接收到 dataset 产出，原样返回


def _save_ckpt(net, opt, scaler, step):
    """原子写 checkpoint：先写 tmp 再替换，防中断损坏"""
    PATHS.ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = PATHS.ckpt_dir / 'bc.tmp.pth'
    torch.save(
        {
            'model': net.state_dict(),
            'optimizer': opt.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': step,
        },
        tmp,
    )
    os.replace(tmp, PATHS.ckpt_dir / 'bc.pth')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=100_000)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--save_every', type=int, default=2_000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit_files', type=int, default=None, help='仅取前 N 个牌谱文件')
    ap.add_argument('--resume', default=None, help='从已有 bc.pth 续训')
    ap.add_argument('--num_workers', type=int, default=3, help='CPU 预加载进程数')
    ap.add_argument('--prefetch_factor', type=int, default=2, help='每 worker 预取批数')
    args = ap.parse_args()

    device = torch.device(args.device)
    net = PolicyNet(version=ENV.version, **asdict(MODEL)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, weights_only=True, map_location=device)
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])  # 恢复 AMP 缩放因子，避免重新爬升
        start_step = state['steps']
        log.info('从 step %d 续训', start_step)

    files = [p for g in PATHS.human_globs for p in glob.glob(g, recursive=True)]
    if args.limit_files:
        files = files[: args.limit_files]
    log.info('牌谱文件数: %d', len(files))

    PATHS.ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(PATHS.ckpt_dir.parent / 'tb_bc')
    writer.add_text('model/params', f'{sum(p.numel() for p in net.parameters()):,}')
    net.train()
    step = start_step

    loader = DataLoader(
        FileIter(files, args.batch_size),
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        collate_fn=_identity_collate,
        worker_init_fn=_worker_init,
        persistent_workers=True,
    )
    pbar = tqdm(total=args.steps, initial=start_step, unit='step', dynamic_ncols=True, ascii=True)
    last_t = time.perf_counter()
    for obs, actions, masks in loader:
        obs = obs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            logits = net(obs)
            loss = F.cross_entropy(logits, actions)
        opt.zero_grad()
        scaler.scale(loss).backward()
        if step % 100 == 0:
            scaler.unscale_(opt)
            grad_norm = sum(
                (p.grad.norm() ** 2).item() for p in net.parameters() if p.grad is not None
            ) ** 0.5
        scaler.step(opt)
        scaler.update()
            now = time.perf_counter()
            samples_per_s = 100 * args.batch_size / (now - last_t)
            writer.add_scalar('bc/loss', loss.item(), step)
            writer.add_scalar('bc/lr', opt.param_groups[0]['lr'], step)
            writer.add_scalar('bc/grad_norm', grad_norm, step)
            writer.add_scalar('bc/throughput', samples_per_s, step)
            pbar.set_postfix(loss=f'{loss.item():.3f}', grad=f'{grad_norm:.2f}', sps=f'{samples_per_s:.0f}')
            last_t = now
        if step % args.save_every == 0:
            _save_ckpt(net, opt, scaler, step)
            log.info('step %d loss %.4f', step, loss.item())
        pbar.update(1)
        step += 1
        if step >= args.steps:
            break
    pbar.close()
    _save_ckpt(net, opt, scaler, step)


if __name__ == '__main__':
    main()
