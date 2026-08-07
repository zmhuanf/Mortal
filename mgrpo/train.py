"""GRPO 主训练循环：rollout → 组内归一化 → 策略更新 → 定期存档"""
import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mgrpo.agent.grpo import GRPO
from mgrpo.agent.opponent import OpponentPool
from mgrpo.agent.rollout import RolloutPool
from mgrpo.config import ENV, MODEL, PATHS, REWARD, ROLLOUT
from mgrpo.env.reward import game_reward
from mgrpo.model.brain import PolicyNet

log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bc', default=str(PATHS.ckpt_dir / 'bc.pth'), help='BC 预训练权重')
    ap.add_argument('--n_games', type=int, default=64, help='每轮 rollout 局数（GRPO 组大小）')
    ap.add_argument('--n_workers', type=int, default=6)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--save_every', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--clip_eps', type=float, default=0.2)
    ap.add_argument('--kl_beta', type=float, default=0.01)
    ap.add_argument('--chunk_games', type=int, default=4, help='梯度累积每块局数，控制显存')
    ap.add_argument('--keep_ckpts', type=int, default=32, help='对手池保留的最近 checkpoint 数')
    args = ap.parse_args()

    device = torch.device('cuda')
    bc = torch.load(args.bc, weights_only=True, map_location='cpu')
    policy = PolicyNet(version=ENV.version, **asdict(MODEL)).to(device)
    policy.load_state_dict(bc['model'])
    ref_policy = PolicyNet(version=ENV.version, **asdict(MODEL)).to(device)
    ref_policy.load_state_dict(bc['model'])
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    grpo = GRPO(policy, ref_policy, opt, clip_eps=args.clip_eps, kl_beta=args.kl_beta, chunk_games=args.chunk_games)

    PATHS.ckpt_dir.mkdir(parents=True, exist_ok=True)
    pool = RolloutPool(args.n_workers, ENV.version)
    opponents = OpponentPool(Path(args.bc), PATHS.ckpt_dir)
    writer = SummaryWriter(PATHS.ckpt_dir.parent / 'tb_grpo')
    key = 0x600D
    seed = 20000

    for step in range(args.steps):
        opp = opponents.sample()
        seeds = [(seed + i, key) for i in range(args.n_games)]
        seed += args.n_games
        trajs = pool.rollout(policy.state_dict(), opp.state, opp.name, seeds, ROLLOUT.log_dir)

        rewards = torch.tensor([
            game_reward(t.scores, 0, REWARD.pts, REWARD.score_weight, REWARD.init_score, REWARD.score_scale)
            for t in trajs
        ])
        advantages = GRPO.compute_advantages(rewards)

        obs = torch.as_tensor(np.concatenate([t.obs for t in trajs]))
        actions = torch.as_tensor(np.concatenate([t.actions for t in trajs]))
        masks = torch.as_tensor(np.concatenate([t.masks for t in trajs]))
        old_lp = torch.as_tensor(np.concatenate([t.log_probs for t in trajs]))
        lengths = torch.as_tensor([len(t.actions) for t in trajs])
        game_ids = torch.repeat_interleave(torch.arange(args.n_games), lengths)

        stats = grpo.update(obs, actions, masks, old_lp, advantages, game_ids)
        avg_rank = float(np.mean([t.rank for t in trajs]))

        if step % 10 == 0:
            for k, v in stats.items():
                writer.add_scalar(f'grpo/{k}', v, step)
            writer.add_scalar('grpo/avg_rank', avg_rank, step)
            writer.add_scalar('grpo/reward_mean', rewards.mean().item(), step)
            log.info(
                'step %d loss=%.4f clip=%.3f kl=%.4f avg_rank=%.3f opp=%s',
                step, stats['loss'], stats['clip_frac'], stats['kl'], avg_rank, opp.name,
            )

        if step % args.save_every == 0:
            torch.save(
                {'model': policy.state_dict(), 'steps': step, 'config': asdict(MODEL)},
                PATHS.ckpt_dir / f'grpo_{step:06d}.pth',
            )
            # 只留最近 keep_ckpts 个，防磁盘膨胀且避免对手池被远古权重稀释
            for f in sorted(PATHS.ckpt_dir.glob('grpo_*.pth'))[:-args.keep_ckpts]:
                f.unlink()

    pool.close()


if __name__ == '__main__':
    main()
