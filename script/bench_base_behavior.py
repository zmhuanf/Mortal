"""mortal_base vs baseline_v1 2v2 行为诊断：位次/顺位/pt + 行为统计对比

加载复用 mortal_base/evaluate.py（load_model / load_opponent），与 bench_base_vs_baseline_v1.py 同源
"""

import importlib.util
import logging
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / 'mortal_base'
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # libriichi.pyd / model.py / config_base.py 优先取自 mortal_base

from libriichi.arena import TwoVsTwo  # noqa: E402
from libriichi.stat import Stat  # noqa: E402

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')

PTS = [90, 45, 0, -135]  # 半庄顺位赏
KEY = 0x2A44
BASE_CKPT = BASE_DIR / 'out' / 'best.pth'
V1_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'
BASE_NAME = 'mortal_base'
V1_NAME = 'baseline_v1'

BEHAVIOR = [
    ('agari_rate', '和牌率'),
    ('houjuu_rate', '放炮率'),
    ('riichi_rate', '立直率'),
    ('fuuro_rate', '副露率'),
    ('ryukyoku_rate', '流局听牌率'),
    ('tobi_rate', '击飞率'),
    ('avg_agari_jun', '平均和牌巡目'),
    ('avg_point_per_agari', '平均和牌点'),
    ('avg_point_per_houjuu', '平均放炮点'),
    ('avg_point_per_round', '局平均得分'),
    ('avg_riichi_jun', '平均立直巡目'),
    ('agari_rate_after_riichi', '立直后和牌率'),
    ('houjuu_rate_after_riichi', '立直后放炮率'),
    ('chasing_riichi_rate', '追立直率'),
    ('avg_fuuro_num', '平均副露数'),
]


@lru_cache(maxsize=1)
def load_base_evaluate():
    """独立命名空间加载 mortal_base/evaluate.py，内部注册 sys.modules['config']"""
    spec = importlib.util.spec_from_file_location('mortal_base_evaluate', BASE_DIR / 'evaluate.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description='mortal_base vs baseline_v1 2v2 行为诊断')
    ap.add_argument('--games', type=int, default=200, help='2v2 对局数，须为偶数')
    ap.add_argument('--seed', type=int, default=20260817, help='固定种子复现')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--base-ckpt', type=Path, default=BASE_CKPT, help='mortal_base checkpoint 路径')
    ap.add_argument('--log-dir', type=Path, default=None, help='保留对局日志的目录')
    args = ap.parse_args()
    if args.games % 2 != 0:
        ap.error('--games 必须为偶数')

    ev = load_base_evaluate()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    log_dir = args.log_dir or (ROOT / 'mgrpo' / '_bench_logs' / 'behavior_diag')
    if log_dir.exists():
        shutil.rmtree(log_dir)

    challenger = ev.load_model(args.base_ckpt, device)
    champion = ev.load_opponent(V1_CKPT, device, V1_NAME)
    print(f'2v2 {args.games} 局: {BASE_NAME} vs {V1_NAME} (seed={args.seed})')
    env = TwoVsTwo(disable_progress_bar=False, log_dir=str(log_dir))
    env.py_vs_py(
        challenger=challenger,
        champion=champion,
        seed_start=(args.seed, KEY),
        seed_count=args.games // 2,
    )

    for name in (BASE_NAME, V1_NAME):
        s = Stat.from_dir(str(log_dir), name, disable_progress_bar=True)
        ranks = [s.rank_1, s.rank_2, s.rank_3, s.rank_4]
        print(f'\n=== {name} ({s.game} 席次) ===')
        print(f'  位次 {ranks}  平均顺位 {s.avg_rank:.3f}  平均pt {s.avg_pt(PTS):+.2f}')
        for attr, label in BEHAVIOR:
            v = getattr(s, attr)
            print(f'  {label:<12} {(v() if callable(v) else v):.3f}')

    if args.log_dir is None:
        shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == '__main__':
    main()