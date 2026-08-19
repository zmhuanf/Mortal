"""uncertainty_scale 定标：量 base/v1 的 q_std 与 policy_logits 尺度比值

纯只读：一次只加载一个模型，forward 算 q_std / logits 跨动作 std，不写文件
用法: python script/probe_uncertainty.py --ckpt mortal_base/out/mortal.pth --files 3
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / 'mortal_base'
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from libriichi.dataset import GameplayLoader  # noqa: E402
from evaluate import V4Brain, V4DQN  # noqa: E402
from model import Brain, DQN  # noqa: E402

INDEX = BASE_DIR / 'out' / 'file_index.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True, help='base/v1 checkpoint 路径')
    ap.add_argument('--files', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    s = torch.load(args.ckpt, weights_only=False, map_location='cpu')
    cfg = s['config']
    if 'model' in cfg:
        brain = Brain(version=cfg['control']['version'], **cfg['model']).to(device).eval()
        dqn = DQN(version=cfg['control']['version'], **cfg['dqn']).to(device).eval()
        brain.load_state_dict(s['mortal'])
        dqn.load_state_dict(s['current_dqn'])
    else:
        brain = V4Brain(**cfg['resnet']).to(device).eval()
        dqn = V4DQN(num_heads=cfg['dqn'].get('num_heads', 1)).to(device).eval()
        brain.load_state_dict(s['mortal'], strict=False)
        dqn.load_state_dict(s['current_dqn'], strict=False)

    q_stds, logit_stds = [], []
    idx = torch.load(INDEX, weights_only=True)
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(idx['file_list'][:args.files]):
        for game in file:
            game.take_player_id()
            g = game.take_grp()  # 消费式，仅调一次
            g.take_rank_by_player()
            obs = np.asarray(game.take_obs(), dtype=np.float32)
            if len(obs) == 0:
                continue
            obs_t = torch.from_numpy(obs).to(device)
            masks = torch.ones(len(obs), 46, dtype=torch.bool, device=device)
            with torch.no_grad():
                phi = brain(obs_t)
                pl = brain.policy_logits(phi)
                qk = dqn(phi, masks)  # (N, K, A)
            q_std = qk.std(1)  # (N, A)
            q_stds.append(q_std.cpu().numpy())
            logit_stds.append(pl.cpu().numpy().std(-1))
    q_std = np.concatenate(q_stds)
    logit_std = np.concatenate(logit_stds)
    print(f'samples: {len(logit_std)}  模型: {args.ckpt.name}')
    print(f'q_std(跨head Q分歧): mean={q_std.mean():.3f} p50={np.median(q_std):.3f} p95={np.percentile(q_std, 95):.3f}')
    print(f'logits 跨动作std    : mean={logit_std.mean():.3f} p50={np.median(logit_std):.3f}')
    print(f'比值 q_std/logits_std = {q_std.mean() / logit_std.mean():.3f}  (uc 的合理量级参考)')
    print(f'若让扰动项占 logits 尺度 15%%: uc ≈ {0.15 * logit_std.mean() / q_std.mean():.3f}')


if __name__ == '__main__':
    main()