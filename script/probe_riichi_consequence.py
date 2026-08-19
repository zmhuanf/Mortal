"""立直分歧后果验证：base 多立直的场景，真实玩家事后结果如何

对每个"一方立直一方不立直"的分歧 turn，用该玩家本半庄最终排名 + 本局后续窗口的和/放炮做裁判
"""

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

RIICHI, PASS = 37, 45
INDEX = BASE_DIR / 'out' / 'file_index.pth'
BASE_CKPT = BASE_DIR / 'out' / 'mortal.pth'
V1_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'


def load_pair(device):
    sb = torch.load(BASE_CKPT, weights_only=False, map_location='cpu')
    sv = torch.load(V1_CKPT, weights_only=True, map_location='cpu')
    brain_b = Brain(version=4, **sb['config']['model']).to(device).eval()
    brain_b.load_state_dict(sb['mortal'])
    brain_v = V4Brain(version=4, **sv['config']['resnet']).to(device).eval()
    brain_v.load_state_dict(sv['mortal'])
    return brain_b, brain_v


def main():
    import argparse

    ap = argparse.ArgumentParser(description='立直分歧后果验证')
    ap.add_argument('--files', type=int, default=60)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    idx = torch.load(INDEX, weights_only=True)
    files = idx['file_list'][:args.files]

    games = []
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(files):
        for game in file:
            grp = game.take_grp()
            if grp.take_feature().shape[0] > 12:  # 与训练一致丢弃
                continue
            final_scores = np.asarray(grp.take_final_scores())
            player_id = game.take_player_id()
            rank = int((-final_scores).argsort(kind='stable').argsort(kind='stable')[player_id])
            dones = game.take_dones()
            games.append({
                'obs': np.asarray(game.take_obs(), dtype=np.float32),
                'masks': np.asarray(game.take_masks()),
                'rank': rank,
                'dones': dones,
                'is_agari': np.array(game.take_is_agari_turn(), dtype=bool),
                'is_houjuu': np.array(game.take_is_houjuu_turn(), dtype=bool),
            })

    brain_b, brain_v = load_pair(device)
    rows = []  # (rank, in_window_agari, in_window_houjuu, type)
    chunk = 2048
    for g in games:
        T = len(g['obs'])
        a_b = np.zeros(T, np.int64)
        a_v = np.zeros(T, np.int64)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            o = torch.from_numpy(g['obs'][s:e]).to(device)
            m = torch.from_numpy(g['masks'][s:e]).to(device)
            with torch.no_grad():
                lb = brain_b.policy_logits(brain_b(o)).masked_fill(~m, -1e9)
                lv = brain_v.policy_logits(brain_v(o)).masked_fill(~m, -1e9)
            a_b[s:e] = lb.argmax(-1).cpu().numpy()
            a_v[s:e] = lv.argmax(-1).cpu().numpy()

        kyoku_end = np.flatnonzero(g['dones'])
        for i in range(T):
            rb, rv = a_b[i] == RIICHI, a_v[i] == RIICHI
            if rb == rv:
                continue
            end = kyoku_end[kyoku_end > i]
            end = int(end[0]) + 1 if len(end) else T
            agari = bool(g['is_agari'][i + 1:end].any())
            houjuu = bool(g['is_houjuu'][i + 1:end].any())
            rows.append((g['rank'], agari, houjuu, 'base立直' if rb else 'v1立直'))

    if not rows:
        print('无分歧样本')
        return
    arr = np.array([[r[0], r[1], r[2]] for r in rows])
    print(f'分歧立直样本 {len(rows)}（60 文件）\n')
    for label in ('base立直', 'v1立直'):
        sel = arr[[r[3] == label for r in rows]]
        n = len(sel)
        if n == 0:
            continue
        print(f'{label:<10} n={n:>4}  该玩家最终排名均值 {sel[:, 0].mean() + 1:.2f}'
              f' (1位{(sel[:, 0] == 0).mean() * 100:.0f}%/2位{(sel[:, 0] == 1).mean() * 100:.0f}%'
              f'/3位{(sel[:, 0] == 2).mean() * 100:.0f}%/4位{(sel[:, 0] == 3).mean() * 100:.0f}%)'
              f'  局内后续和牌率 {sel[:, 1].mean() * 100:.1f}%  放炮率 {sel[:, 2].mean() * 100:.1f}%')


if __name__ == '__main__':
    main()