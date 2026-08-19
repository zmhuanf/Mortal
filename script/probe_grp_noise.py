"""GRP 误差对 delta_pt 的污染量化

口径：逐 turn 对 4 玩家预测最终排名(4类)，top1 命中=acc
关键测量：终局 turn（信息完整时）GRP 期望点数 vs 真实点数偏差 → 污染量级 vs delta 信号量级
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
from model import GRP  # noqa: E402
from config import config  # noqa: E402

INDEX = BASE_DIR / 'out' / 'file_index.pth'
PTS = np.asarray([10.0, 4.0, -1.0, -5.0])


def main():
    import argparse

    ap = argparse.ArgumentParser(description='GRP 污染量化')
    ap.add_argument('--files', type=int, default=60)
    args = ap.parse_args()

    grp = GRP(**config['grp']['network']).eval()
    grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True)['model'])

    tops, errs = [], []
    idx = torch.load(INDEX, weights_only=True)
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(idx['file_list'][:args.files]):
        for game in file:
            g = game.take_grp()  # take_grp 是消费式，只能取一次
            feature = g.take_feature()
            T = feature.shape[0]
            if T == 0 or T > 12:
                continue
            rank = np.frombuffer(g.take_rank_by_player(), dtype=np.uint8).astype(np.int64)  # 0..3
            with torch.inference_mode():
                logits = grp.forward([torch.as_tensor(feature, dtype=torch.float32)])  # (1, 16)
            matrix = grp.calc_matrix(logits).numpy()[0]  # (player, rank_prob)，Sinkhorn 双随机
            pred = matrix.argmax(-1)
            tops.append(pred == rank)
            errs.append(np.abs(matrix @ PTS - PTS[rank]))
            if len(tops) <= 3:
                print(f'  rank={rank.tolist()} pred={pred.tolist()} matrix_row0={np.round(matrix[0], 2).tolist()}')

    tops = np.concatenate(tops)
    errs = np.concatenate(errs)
    print(f'文件 {args.files}，终局样本 {len(tops)}')
    print(f'终局 top1 命中率(4类,随机25%): {tops.mean() * 100:.1f}%')
    print(f'终局期望点数偏差 |E[pts]-真实pts| 均值: {errs.mean():.2f}  分位: '
          f'p50 {np.percentile(errs, 50):.2f} p75 {np.percentile(errs, 75):.2f} p90 {np.percentile(errs, 90):.2f}')
    print(f'pts 尺度参考: 一顺位差=14(10->-5) 或 6(10->4) 相邻位差均值 {np.diff(PTS).mean():.2f}')


if __name__ == '__main__':
    main()