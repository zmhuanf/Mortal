"""统计真实对局中各事件的平均排名期望差分（delta_pt），标定 shaping 奖励

事件类别（与 event 标签一致）：0=无事 1=立直 2=和牌 3=放铳 4=流局 5=被自摸
输出每类事件的 delta_pt 均值/中位/标准差，均值即该事件对排名期望的边际影响，可直接当 shaping 值
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config_base  # noqa: E402
from config import config  # noqa: E402
from model import GRP  # noqa: E402
from dataset import RewardCalculator  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402

N_FILES = 100
NAMES = ('无事', '立直', '和牌', '放铳', '流局', '被自摸')


def main():
    grp = GRP(**config['grp']['network'])
    grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')['model'])
    reward_calc = RewardCalculator(grp, config['env']['pts'])

    idx = torch.load(config['dataset']['file_index'], weights_only=True)
    rng = np.random.default_rng(7)
    files = rng.choice(idx['file_list'], N_FILES, replace=False)

    buckets = {k: [] for k in range(6)}
    n_kyoku_tot = 0
    loader = GameplayLoader(version=config['control']['version'], oracle=False)
    for file in loader.load_gz_log_files(files):
        for game in file:
            obs = np.asarray(game.take_obs(), dtype=np.float32)
            if len(obs) == 0:
                continue
            at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
            dones = np.asarray(game.take_dones(), dtype=bool)
            is_riichi = np.array(game.take_is_riichi_turn(), dtype=bool)
            is_agari = np.array(game.take_is_agari_turn(), dtype=bool)
            is_houjuu = np.array(game.take_is_houjuu_turn(), dtype=bool)
            grp_obj = game.take_grp()
            grp_feature = grp_obj.take_feature()
            if grp_feature.shape[0] > 12:
                continue
            rank_by_player = grp_obj.take_rank_by_player()
            final_scores = grp_obj.take_final_scores()
            scores_seq = np.concatenate((grp_feature[:, 3:7] * 1e4, [final_scores]))
            player_id = game.take_player_id()
            kyoku_rewards = reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
            n_kyoku_tot += len(kyoku_rewards)
            # 事件分层：与 dataset.py 完全一致
            for i in range(len(obs)):
                if is_riichi[i]:
                    buckets[1].append(kyoku_rewards[at_kyoku[i]])
                elif is_agari[i]:
                    buckets[2].append(kyoku_rewards[at_kyoku[i]])
                elif is_houjuu[i]:
                    buckets[3].append(kyoku_rewards[at_kyoku[i]])
                elif dones[i]:
                    diff = scores_seq[at_kyoku[i] + 1, player_id] - scores_seq[at_kyoku[i], player_id]
                    buckets[5 if diff < 0 else 4].append(kyoku_rewards[at_kyoku[i]])

    print(f'文件 {N_FILES} | 局数 {n_kyoku_tot}')
    print(f'{"事件":<6} {"n":>6} {"mean":>8} {"median":>8} {"std":>8}  <- 均值即建议 shaping 值')
    for k in range(6):
        arr = np.array(buckets[k], dtype=np.float64)
        if len(arr) == 0:
            continue
        print(f'{NAMES[k]:<6} {len(arr):>6} {arr.mean():>8.4f} {np.median(arr):>8.4f} {arr.std():>8.4f}')


if __name__ == '__main__':
    main()