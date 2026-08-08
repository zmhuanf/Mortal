"""GRPO 主训练循环：rollout → 组内归一化 → 策略更新 → 定期存档，支持断点续训"""
import argparse
import ctypes
import logging
import secrets
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mgrpo.prelude  # noqa: E402  注入 mortal/ 路径，使 libriichi.pyd 可导入

import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

from mgrpo.agent.grpo import GRPO
from mgrpo.agent.opponent import OpponentPool
from mgrpo.agent.rollout import RolloutPool
from mgrpo.config import ENV, MODEL, PATHS, REWARD, ROLLOUT
from mgrpo.env.reward import game_reward
from mgrpo.model.brain import PolicyNet

log = logging.getLogger(__name__)


def _trim_heap():
    """归还 UCRT 空闲堆段给 OS，防大块数组释放后堆碎片残留累积"""
    try:
        ctypes.CDLL('ucrtbase')._heapmin()
    except OSError:
        pass


def _on_batch(step: int, received: list[int], n: int):
    """rollout 回传进度：每 4 批（一个模拟批）输出一行"""
    received[0] += n
    if received[0] % 4 == 0:
        log.info('step %d 回传中：%d 批', step, received[0])


def _mem_line(pool) -> str:
    """主进程与各 worker 的 RSS 诊断行，单位 MB"""
    procs = [psutil.Process(), *(psutil.Process(p.pid) for p in pool.procs)]
    return ' | '.join(f'{p.pid}:{p.memory_info().rss / 2 ** 20:.0f}MB' for p in procs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bc', default=str(PATHS.ckpt_dir / 'bc.pth'), help='BC 预训练权重')
    ap.add_argument('--resume', default=None, help='从 grpo_*.pth 续训，恢复权重/优化器/seed/步数')
    ap.add_argument('--n_games', type=int, default=32, help='每轮 rollout 种子数（每种子四座位变体，轨迹 = 4×n_games）')
    ap.add_argument('--n_workers', type=int, default=4)
    ap.add_argument('--rollout_device', default='cuda', help='rollout 推理设备（worker 进程内）')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--save_every', type=int, default=5)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--clip_eps', type=float, default=0.2)
    ap.add_argument('--kl_beta', type=float, default=0.01)
    ap.add_argument('--chunk_games', type=int, default=4, help='梯度累积每块局数，控制显存')
    ap.add_argument('--keep_ckpts', type=int, default=32, help='对手池保留的最近 checkpoint 数')
    ap.add_argument('--eval_games', type=int, default=64, help='评估种子数（每种子四座位变体）')
    args = ap.parse_args()

    device = torch.device('cuda')
    bc = torch.load(args.bc, weights_only=True, map_location='cpu')
    policy = PolicyNet(version=ENV.version, **asdict(MODEL)).to(device)
    ref_policy = PolicyNet(version=ENV.version, **asdict(MODEL)).to(device)
    ref_policy.load_state_dict(bc['model'])
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    grpo = GRPO(policy, ref_policy, opt, clip_eps=args.clip_eps, kl_beta=args.kl_beta, chunk_games=args.chunk_games)

    PATHS.ckpt_dir.mkdir(parents=True, exist_ok=True)
    pool = RolloutPool(args.n_workers, ENV.version, args.rollout_device, args.chunk_games)
    opponents = OpponentPool(Path(args.bc), PATHS.ckpt_dir)
    writer = SummaryWriter(PATHS.ckpt_dir.parent / 'tb_grpo', flush_secs=5)
    key = 0x600D
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, weights_only=True, map_location=device)
        policy.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        seed = state['seed']
        eval_seed = state['eval_seed']
        start_step = state['steps'] + 1
        log.info('从 step %d 续训，seed=%d', start_step, seed)
    else:
        # 牌山种子随机初始化，避免重训重复消费同一批开局
        policy.load_state_dict(bc['model'])
        seed = secrets.randbits(32)
        eval_seed = secrets.randbits(32)
        log.info('全新训练，seed=%d', seed)

    base_opp = opponents.baseline_opponent()
    best_avg_pt = float('-inf')

    for step in range(start_step, args.steps):
        t0 = time.perf_counter()
        opp = opponents.sample()
        seeds = [(seed + i, key) for i in range(args.n_games)]
        seed += args.n_games

        log.info('step %d 开始 rollout：%d 种子 × 4 变体 → %d worker 模拟中', step, args.n_games, args.n_workers)
        received = [0]
        gen = pool.rollout(
            policy.state_dict(), opp.state, opp.name, seeds, ROLLOUT.log_dir,
            progress=lambda n: _on_batch(step, received, n),
        )
        total_batches = next(gen)  # 首项：总批数
        advantages = next(gen)  # 阶段 1 收齐
        log.info('step %d 回传完成（%d 批，%d 局），归一化 advantage，开始训练', step, total_batches, 4 * args.n_games)
        grpo.begin(advantages)
        ranks, rewards_list, n_games = [], [], 0
        for traj_batch, _ in gen:
            ranks.extend(t.rank for t in traj_batch)
            rewards_list.extend(
                game_reward(t.scores, 0, REWARD.pts, REWARD.score_weight, REWARD.init_score, REWARD.score_scale)
                for t in traj_batch
            )
            obs = torch.as_tensor(np.concatenate([t.obs for t in traj_batch]))
            actions = torch.as_tensor(np.concatenate([t.actions for t in traj_batch]))
            masks = torch.as_tensor(np.concatenate([t.masks for t in traj_batch]))
            old_lp = torch.as_tensor(np.concatenate([t.log_probs for t in traj_batch]))
            lengths = torch.as_tensor([len(t.actions) for t in traj_batch])
            game_ids = torch.repeat_interleave(torch.arange(n_games, n_games + len(traj_batch)), lengths)
            grpo.feed(obs, actions, masks, old_lp, game_ids)
            n_games += len(traj_batch)
            del obs, actions, masks, old_lp, game_ids  # 本批随即释放，峰值仅与 chunk_games 成正比
        stats = grpo.end()
        _trim_heap()  # traj_store 已释放，趁此归还堆段
        rollout_s = time.perf_counter() - t0
        avg_rank = float(np.mean(ranks))
        rewards = torch.tensor(rewards_list)

        for k, v in stats.items():
            writer.add_scalar(f'grpo/{k}', v, step)
        writer.add_scalar('grpo/avg_rank', avg_rank, step)
        writer.add_scalar('grpo/reward_mean', rewards.mean().item(), step)
        writer.add_scalar('grpo/rollout_s', rollout_s, step)
        writer.add_scalar('grpo/samples_per_s', n_games / rollout_s, step)
        writer.flush()
        log.info(
            'step %d loss=%.4f clip=%.3f kl=%.4f ratio=%.3f adv_std=%.3f rank=%.2f reward=%.1f rollout=%.0fs opp=%s | %s',
            step, stats['loss'], stats['clip_frac'], stats['kl'], stats['ratio'], stats['adv_std'],
            avg_rank, rewards.mean().item(), rollout_s, opp.name, _mem_line(pool),
        )

        if step % args.save_every == 0:
            ckpt_file = PATHS.ckpt_dir / f'grpo_{step:06d}.pth'
            torch.save(
                {'model': policy.state_dict(), 'optimizer': opt.state_dict(), 'seed': seed, 'eval_seed': eval_seed, 'steps': step, 'config': asdict(MODEL)},
                ckpt_file,
            )
            log.info('已保存 %s', ckpt_file)
            # 只留最近 keep_ckpts 个，防磁盘膨胀且避免对手池被远古权重稀释
            for f in sorted(PATHS.ckpt_dir.glob('grpo_*.pth'))[:-args.keep_ckpts]:
                f.unlink()

            # 对固定参照（BC baseline）评估，顺位质量优于历史最优则覆盖 best.pth
            eval_seeds = [(eval_seed + i, key) for i in range(args.eval_games)]
            eval_seed += args.eval_games
            log.info('eval 开始：%d 种子 → 模拟评估中', args.eval_games)
            eval_gen = pool.rollout(policy.state_dict(), base_opp.state, base_opp.name, eval_seeds, ROLLOUT.log_dir / 'eval')
            next(eval_gen)  # 总批数
            next(eval_gen)  # 丢弃 advantage，仅统计顺位
            eval_ranks = [t.rank for batch, _ in eval_gen for t in batch]
            log.info('eval 完成：%d 局', 4 * args.eval_games)
            eval_avg_pt = float(np.mean([REWARD.pts[r - 1] for r in eval_ranks]))
            writer.add_scalar('grpo/eval_avg_pt', eval_avg_pt, step)
            if eval_avg_pt > best_avg_pt:
                best_avg_pt = eval_avg_pt
                torch.save(
                    {'model': policy.state_dict(), 'optimizer': opt.state_dict(), 'seed': seed, 'eval_seed': eval_seed, 'steps': step, 'avg_pt': eval_avg_pt, 'config': asdict(MODEL)},
                    PATHS.ckpt_dir / 'best.pth',
                )
                log.info('顺位新高，已覆盖 best.pth（avg_pt=%.2f）', eval_avg_pt)
            log.info('eval avg_pt=%.2f best=%.2f opp=%s', eval_avg_pt, best_avg_pt, base_opp.name)
            _trim_heap()  # eval 的 traj_store 同样释放后归还

    pool.close()


if __name__ == '__main__':
    main()
