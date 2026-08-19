"""V 校准诊断：模型 V 输出 vs 该玩家真实最终 pts，按终局排名分桶

纯只读：加载 checkpoint、前向推理、比对，不写任何文件
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
from model import Brain, DQN  # noqa: E402
from evaluate import V4Brain, V4DQN  # noqa: E402

INDEX = BASE_DIR / 'out' / 'file_index.pth'
BASE_CKPT = BASE_DIR / 'out' / 'mortal.pth'
V1_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'
PTS = np.array([10.0, 4.0, -1.0, -5.0])  # 校准基准用当前 base 的 pts 排序档


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    idx = torch.load(INDEX, weights_only=True)
    files = idx['file_list'][:args.files]

    obs_all, mask_all, fin_pts_all = [], [], []
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(files):
        for game in file:
            player_id = game.take_player_id()
            g = game.take_grp()  # take_grp 消费式，只能取一次
            rank = np.frombuffer(g.take_rank_by_player(), dtype=np.uint8)  # rank[player]=名次 0..3
            obs_g = np.asarray(game.take_obs(), dtype=np.float32)  # take_obs 消费式，只能取一次
            nt = len(obs_g)
            if nt == 0:
                continue
            obs_all.append(obs_g)
            fin_pts_all.append(np.full(nt, PTS[rank[player_id]]))  # 该玩家真实最终 pts
    obs = np.concatenate(obs_all)
    fin = np.concatenate(fin_pts_all)
    print(f'samples: {len(obs)}')

    sb = torch.load(BASE_CKPT, weights_only=False, map_location='cpu')
    sv = torch.load(V1_CKPT, weights_only=True, map_location='cpu')
    brain_b = Brain(version=4, **sb['config']['model']).to(device).eval()
    brain_b.load_state_dict(sb['mortal'])
    dqn_b = DQN(version=4, **sb['config']['dqn']).to(device).eval()
    dqn_b.load_state_dict(sb['current_dqn'])
    brain_v = V4Brain(version=4, **sv['config']['resnet']).to(device).eval()
    brain_v.load_state_dict(sv['mortal'])
    dqn_v = V4DQN(num_heads=sv['config']['dqn']['num_heads']).to(device).eval()
    dqn_v.load_state_dict(sv['current_dqn'])

    def calc_v(predict, v_head):
        out = np.empty(len(obs))
        for s in range(0, len(obs), 512):
            e = min(s + 512, len(obs))
            o = torch.from_numpy(obs[s:e]).to(device)
            with torch.no_grad():
                v = v_head(predict(o))
            out[s:e] = v.mean(-1).cpu().numpy()
        return out

    v_b = calc_v(brain_b, dqn_b.value)
    v_v = calc_v(brain_v, lambda phi: dqn_v.net(phi).split((dqn_v.num_heads, dqn_v.num_heads * 46), dim=-1)[0])

    print(f'\n{"":<12}{"n":>6}{"真实pts":>9}{"baseV":>9}{"v1V":>9}{"base偏差":>10}{"v1偏差":>9}')
    for rank in range(4):
        sel = fin == PTS[rank]
        n = sel.sum()
        if n == 0:
            continue
        rb, rv = v_b[sel], v_v[sel]
        print(f'终局rank{rank}  {n:>6}{PTS[rank]:>9.0f}{rb.mean():>9.2f}{rv.mean():>9.2f}'
              f'{rb.mean()-PTS[rank]:>+10.2f}{rv.mean()-PTS[rank]:>+9.2f}')
    print(f'\n全体        {len(fin):>6}{fin.mean():>9.2f}{v_b.mean():>9.2f}{v_v.mean():>9.2f}'
          f'{v_b.mean()-fin.mean():>+10.2f}{v_v.mean()-fin.mean():>+9.2f}')
    print(f'base V 标准差 {v_b.std():.2f}   v1 V 标准差 {v_v.std():.2f}')


if __name__ == '__main__':
    main()