"""GRP 是否已给立直/和牌/放炮定价？

统计各事件 turn 的 n_step 窗口内 delta(GRP 排名信号)与 shaping 的分布
若 delta 在事件 turn 明显非零 → GRP 自带定价，shaping 冗余；否则 shaping 是唯一信号源
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
import config_base  # noqa: E402
from dataset import RewardCalculator  # noqa: E402
from model import GRP  # noqa: E402
from config import config  # noqa: E402

INDEX = BASE_DIR / 'out' / 'file_index.pth'
PTS = [10.0, 4.0, -1.0, -5.0]
SHAPING = {'riichi': 0.3, 'agari': 1.5, 'houjuu': -2.0}
N_STEP, GAMMA = 5, 0.99
GAMMA_POW = np.float32(GAMMA) ** np.arange(N_STEP + 1, dtype=np.float32)


def main():
    import argparse

    ap = argparse.ArgumentParser(description='GRP 定价检查')
    ap.add_argument('--files', type=int, default=60)
    args = ap.parse_args()

    grp = GRP(**config['grp']['network']).eval()
    grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True)['model'])
    calc = RewardCalculator(grp, PTS)

    stats = {k: [] for k in ('riichi', 'agari', 'houjuu', 'plain')}
    idx = torch.load(INDEX, weights_only=True)
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(idx['file_list'][:args.files]):
        for game in file:
            obs = np.asarray(game.take_obs())
            T = len(obs)
            if T == 0:
                continue
            g = game.take_grp()  # take_grp 消费式，只能取一次
            grp_feature = g.take_feature()
            if grp_feature.shape[0] > 12:
                continue
            player_id = game.take_player_id()
            rank_by_player = g.take_rank_by_player()
            kyoku_rewards = calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
            apply_gamma = game.take_apply_gamma()
            dones = game.take_dones()
            at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
            is_riichi = np.array(game.take_is_riichi_turn(), dtype=bool)
            is_agari = np.array(game.take_is_agari_turn(), dtype=bool)
            is_houjuu = np.array(game.take_is_houjuu_turn(), dtype=bool)
            gamma_prefix = np.concatenate(([0], np.cumsum(np.asarray(apply_gamma, dtype=np.int64))))
            kyoku_end_turns = np.flatnonzero(dones)

            for i in range(T):
                end = min(T, int(np.searchsorted(gamma_prefix, gamma_prefix[i] + N_STEP, side='left')))
                discount = gamma_prefix[i:end] - gamma_prefix[i]
                turn_part = np.dot(GAMMA_POW[discount],
                                   (is_riichi[i:end] * SHAPING['riichi']
                                    + is_agari[i:end] * SHAPING['agari']
                                    + is_houjuu[i:end] * SHAPING['houjuu']).astype(np.float32))
                delta_part = np.float32(0.0)
                for j in kyoku_end_turns[np.searchsorted(kyoku_end_turns, i, side='left'):]:
                    d = int(gamma_prefix[j + 1] - gamma_prefix[i])
                    if d >= N_STEP:
                        break
                    delta_part += np.float32(GAMMA_POW[d] * kyoku_rewards[at_kyoku[j]])
                key = 'riichi' if is_riichi[i] else ('agari' if is_agari[i] else ('houjuu' if is_houjuu[i] else 'plain'))
                stats[key].append((float(turn_part), float(delta_part)))

    print('事件 turn 的 n_step 窗口成分（当前配置 pts=[10,4,-1,-5] shaping=0.3/1.5/-2.0）')
    print(f'{"turn类型":<8}{"n":>6}{"|turn|均值":>12}{"|delta|均值":>12}{"delta均值":>12}{"delta非零率":>12}')
    for k, v in stats.items():
        v = np.array(v)
        print(f'{k:<8}{len(v):>6}{np.abs(v[:, 0]).mean():>12.3f}{np.abs(v[:, 1]).mean():>12.3f}'
              f'{v[:, 1].mean():>12.3f}{(v[:, 1] != 0).mean() * 100:>11.1f}%')


if __name__ == '__main__':
    main()