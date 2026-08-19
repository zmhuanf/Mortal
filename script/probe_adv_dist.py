"""adv 分布统计：AWR 的 adv=q_target-v 的 std → β 理论最优值 (β≈σ(adv))

纯只读：加载 mortal.pth，前向算 q_target/v，不写任何文件
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
BASE_CKPT = None  # 由 --ckpt 参数指定
PTS = np.array([10.0, 4.0, -1.0, -5.0])
GAMMA_N = 0.99 ** 3  # n_step=3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, default=None)
    ap.add_argument('--files', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    s = torch.load(args.ckpt, weights_only=False, map_location='cpu')
    conf = s['config']
    if 'model' in conf:
        brain = Brain(version=4, **conf['model']).to(device).eval()
        dqn = DQN(version=4, **conf['dqn']).to(device).eval()
        brain.load_state_dict(s['mortal'])
        dqn.load_state_dict(s['current_dqn'])
    else:
        brain = V4Brain(**conf['resnet']).to(device).eval()
        dqn = V4DQN(num_heads=conf['dqn'].get('num_heads', 1)).to(device).eval()
        brain.load_state_dict(s['mortal'], strict=False)
        dqn.load_state_dict(s['current_dqn'], strict=False)

    all_adv, end_adv = [], []
    idx = torch.load(INDEX, weights_only=True)
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(idx['file_list'][:args.files]):
        for game in file:
            player_id = game.take_player_id()
            g = game.take_grp()  # 消费式，仅调一次
            rank = np.frombuffer(g.take_rank_by_player(), dtype=np.uint8)
            obs = np.asarray(game.take_obs(), dtype=np.float32)
            nt = len(obs)
            if nt == 0:
                continue
            obs_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                phi = brain(obs_t)
                next_phi = brain(torch.from_numpy(obs[1:nt]).to(device))
                if hasattr(dqn, 'value'):
                    v = dqn.value(phi).cpu().numpy()
                    next_v = dqn.value(next_phi).cpu().numpy()
                else:
                    v = dqn.net(phi).split((dqn.num_heads, dqn.num_heads * 46), -1)[0].view(-1, dqn.num_heads, 1).cpu().numpy()
                    next_v = dqn.net(next_phi).split((dqn.num_heads, dqn.num_heads * 46), -1)[0].view(-1, dqn.num_heads, 1).cpu().numpy()
            q_target = np.zeros_like(v)
            q_target[:-1] = GAMMA_N * next_v
            q_target[-1] = PTS[rank[player_id]]
            is_end = np.zeros(nt, dtype=bool)
            is_end[-1] = True
            all_adv.append((q_target - v).mean(-1))
            end_adv.append((q_target - v).mean(-1)[is_end])
    adv = np.concatenate(all_adv)
    adv_end = np.concatenate(end_adv)
    print(f'samples: {len(adv)} (局末 {len(adv_end)})')
    for name, a in (('全体', adv), ('局末', adv_end)):
        print(f'{name}: mean={a.mean():+.3f} std={a.std():.3f} '
              f'p5={np.percentile(a, 5):+.3f} p25={np.percentile(a, 25):+.3f} '
              f'p50={np.median(a):+.3f} p75={np.percentile(a, 75):+.3f} p95={np.percentile(a, 95):+.3f}')
    print(f'理论最优 beta = std(adv) = {adv.std():.3f}')


if __name__ == '__main__':
    main()