"""网络组件级体检：Policy 头 / Q 头（V、A、ensemble）逐项指标

纯只读：一次一个模型，--ckpt 指定 base 或 v1
用法: python script/probe_components.py --ckpt <path> --files 3
输出: policy 决策熵与动作分布 / V 标度 / A 幅度 / ensemble 分歧 / 两决策源分歧率
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
ACT = 46


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
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
    print(f'模型: {args.ckpt.name} steps={s["steps"]}')

    ents, v_all, a_spread, q_kdiv, pol_arg, q_arg = [], [], [], [], [], []
    idx = torch.load(INDEX, weights_only=True)
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(idx['file_list'][:args.files]):
        for game in file:
            game.take_player_id()
            g = game.take_grp()  # 消费式，仅一次
            g.take_rank_by_player()
            obs = np.asarray(game.take_obs(), dtype=np.float32)
            if len(obs) == 0:
                continue
            obs_t = torch.from_numpy(obs).to(device)
            masks = torch.ones(len(obs), ACT, dtype=torch.bool, device=device)
            with torch.no_grad():
                phi = brain(obs_t)
                pl = brain.policy_logits(phi)
                qk = dqn(phi, masks)  # (N, K, A)
                if hasattr(dqn, 'value'):
                    v = dqn.value(phi)  # (N, K)
                else:
                    v = dqn.net(phi).split((dqn.num_heads, dqn.num_heads * ACT), -1)[0].view(-1, dqn.num_heads, 1)
            prob = pl.softmax(-1).cpu().numpy()
            qm = qk.mean(1).cpu().numpy()
            ents.append(-(prob * np.log(prob + 1e-12)).sum(-1))
            pol_arg.append(prob.argmax(-1))
            q_arg.append(qm.argmax(-1))
            v_all.append(v.mean(-1).cpu().numpy())
            a_spread.append((qm - qm.mean(-1, keepdims=True)).std(-1))
            q_kdiv.append(qk.std(1).mean(-1).cpu().numpy())

    ents = np.concatenate(ents)
    pol_arg = np.concatenate(pol_arg)
    q_arg = np.concatenate(q_arg)
    v_all = np.concatenate(v_all)
    a_spread = np.concatenate(a_spread)
    q_kdiv = np.concatenate(q_kdiv)
    print(f'samples: {len(ents)}')
    print()
    print('[Policy 头]')
    print(f'  决策熵 mean={ents.mean():.3f} p95={np.percentile(ents, 95):.3f} (v1 需对照)')
    top, cnt = np.unique(pol_arg, return_counts=True)
    top_idx = np.argsort(-cnt)[:8]
    print(f'  argmax 动作TOP8: {[(int(top[i]), int(cnt[i])) for i in top_idx]}')
    print(f'  policy vs Q 决策分歧率: {(pol_arg != q_arg).mean()*100:.1f}%')
    print('[Q 头]')
    print(f'  V 均值 {v_all.mean():+.3f}  V std {v_all.std():.3f}')
    print(f'  A 幅度(跨动作std) mean={a_spread.mean():.4f}')
    print(f'  ensemble 分歧(跨head std) mean={q_kdiv.mean():.4f} p95={np.percentile(q_kdiv, 95):.4f}')


if __name__ == '__main__':
    main()