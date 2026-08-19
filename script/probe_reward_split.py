"""turn-level shaping vs 排名信号(GRP delta_pt) 在 n_step reward 中的占比

两套配置同批牌谱同窗口对比：
  A. mortal_base 当前：pts [10,4,-1,-5], reward 0.3/1.5/-2.0
  B. baseline_v1 验证过：pts [6,4,2,0],  reward 0.1/0.3/-0.3
输出每 transition 的 n_step 窗口内 |turn| 与 |delta| 的贡献占比
"""

import logging
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / 'mortal_base'
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from libriichi.dataset import GameplayLoader  # noqa: E402
import config_base  # noqa: E402  # 注册 sys.modules['config']
from dataset import RewardCalculator  # noqa: E402
from model import GRP  # noqa: E402
from config import config  # noqa: E402

INDEX = BASE_DIR / 'out' / 'file_index.pth'

CONFIGS = {
    'mortal_base 当前': {
        'pts': [10.0, 4.0, -1.0, -5.0],
        'reward': {'riichi': 0.3, 'agari': 1.5, 'houjuu': -2.0},
    },
    'baseline_v1 验证': {
        'pts': [6.0, 4.0, 2.0, 0.0],
        'reward': {'riichi': 0.1, 'agari': 0.3, 'houjuu': -0.3},
    },
}

N_STEP = 5
GAMMA = 0.99
GAMMA_POW = np.float32(GAMMA) ** np.arange(N_STEP + 1, dtype=np.float32)


def run():
    import argparse

    ap = argparse.ArgumentParser(description='reward 占比分解')
    ap.add_argument('--files', type=int, default=30)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    idx = torch.load(INDEX, weights_only=True)
    files = idx['file_list'][:args.files]

    grp = GRP(**config['grp']['network']).eval()
    grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True)['model'])
    calc = RewardCalculator(grp, CONFIGS['mortal_base 当前']['pts'], uniform_init=False)

    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    rows = {k: [] for k in CONFIGS}
    for file in loader.load_gz_log_files(files):
        for game in file:
            obs = np.asarray(game.take_obs())
            T = len(obs)
            if T == 0:
                continue
            apply_gamma = game.take_apply_gamma()
            dones = game.take_dones()
            at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
            is_riichi = np.array(game.take_is_riichi_turn(), dtype=bool)
            is_agari = np.array(game.take_is_agari_turn(), dtype=bool)
            is_houjuu = np.array(game.take_is_houjuu_turn(), dtype=bool)
            g = game.take_grp()  # take_grp 消费式，只能取一次
            grp_feature = g.take_feature()
            if grp_feature.shape[0] > 12:
                continue
            rank_by_player = g.take_rank_by_player()
            player_id = game.take_player_id()

            gamma_prefix = np.concatenate(([0], np.cumsum(np.asarray(apply_gamma, dtype=np.int64))))
            kyoku_end_turns = np.flatnonzero(dones)

            for name, cfg in CONFIGS.items():
                kyoku_rewards = calc.calc_delta_pt(player_id, grp_feature, rank_by_player) \
                    if name == 'mortal_base 当前' else RewardCalculator(grp, cfg['pts']).calc_delta_pt(
                        player_id, grp_feature, rank_by_player)
                turn_rewards = (
                    is_riichi * cfg['reward']['riichi']
                    + is_agari * cfg['reward']['agari']
                    + is_houjuu * cfg['reward']['houjuu']
                ).astype(np.float32)

                for i in range(T):
                    end = min(T, int(np.searchsorted(gamma_prefix, gamma_prefix[i] + N_STEP, side='left')))
                    discount = gamma_prefix[i:end] - gamma_prefix[i]
                    turn_part = np.dot(GAMMA_POW[discount], turn_rewards[i:end])
                    delta_part = np.float32(0.0)
                    for j in kyoku_end_turns[np.searchsorted(kyoku_end_turns, i, side='left'):]:
                        d = int(gamma_prefix[j + 1] - gamma_prefix[i])
                        if d >= N_STEP:
                            break
                        delta_part += np.float32(GAMMA_POW[d] * kyoku_rewards[at_kyoku[j]])
                    rows[name].append((abs(turn_part), abs(delta_part), turn_part + delta_part))

    print(f'每 transition n_step 窗口（{args.files} 文件, T={sum(len(r) for r in rows.values()):,}）')
    for name, r in rows.items():
        turn = np.array([x[0] for x in r])
        delta = np.array([x[1] for x in r])
        total = np.array([x[2] for x in r])
        frac = turn / (turn + delta + 1e-9)
        print(f'\n{name}')
        print(f'  pts={CONFIGS[name]["pts"]}  reward={CONFIGS[name]["reward"]}')
        print(f'  |turn| 均值 {turn.mean():.3f}  占比 {frac.mean() * 100:.1f}%')
        print(f'  |delta|均值 {delta.mean():.3f}')
        print(f'  净 n_step_r 均值 {total.mean():+.3f}  正负比 {(total > 0).mean() * 100:.0f}%')
        print(f'  窗口全为 0(turn=delta=0) 占比 {((turn + delta) == 0).mean() * 100:.1f}%')


if __name__ == '__main__':
    run()