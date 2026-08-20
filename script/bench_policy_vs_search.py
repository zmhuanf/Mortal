"""policy 直出 vs search（事件模型 rollout）2v2 同台对局

同一 checkpoint 的两个推理分支各占 2 席对打，直接比谁赢得多
每 seed 跑 a/b 两局轮换座位消除座位偏差，输出双方位次/顺位/pt

用法: python script/bench_policy_vs_search.py [--games 1000] [--swap] [--search-alpha 0.0]
"""

import argparse
import importlib.util
import logging
import secrets
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / 'mortal_base'
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from libriichi.arena import TwoVsTwo  # noqa: E402
from libriichi.stat import Stat  # noqa: E402

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
)
log = logging.getLogger(__name__)

PTS = [90, 45, 0, -135]  # 半庄顺位赏
KEY = 0x2A44
BASE_CKPT = BASE_DIR / 'out' / 'mortal.pth'


def load_evaluate():
    """mortal_base/evaluate.py 独立加载：内部 import config_base 注册 sys.modules['config']"""
    spec = importlib.util.spec_from_file_location('mortal_base_evaluate', BASE_DIR / 'evaluate.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_engines(mod, device, ckpt, mode_a, mode_b, search_k, search_alpha):
    """同一 checkpoint 构造两个引擎：按 mode 指定 action_source（policy/search/vrisk）"""
    import model as M  # noqa: F401

    state = torch.load(ckpt, weights_only=False, map_location=device)
    cfg = state['config']
    mortal = M.Brain(version=cfg['control']['version'], **cfg['model']).to(device).eval()
    mortal.load_state_dict(state['mortal'])
    dqn = M.DQN(version=cfg['control']['version'], **cfg['dqn']).to(device).eval()
    dqn.load_state_dict(state['current_dqn'])
    event_model = None
    if 'event_model' in state:
        event_model = M.EventModel(
            phi_dim=cfg['model'].get('phi_dim', 1024),
            **{k: v for k, v in cfg.get('event', {}).items() if k != 'weight'},
        ).to(device).eval()
        event_model.load_state_dict(state['event_model'])

    def build(name, mode):
        if mode == 'search' and event_model is None:
            raise SystemExit('checkpoint 不含 event_model，无法构造 search 引擎')
        return mod.make_engine(
            mortal, dqn, device, cfg['control']['version'],
            name=name, event_model=event_model, action_mode=mode,
            search_k=search_k, search_alpha=search_alpha,
        )
    return build('mode_a', mode_a), build('mode_b', mode_b)


def run_games(challenger, champion, seed, games, log_dir):
    env = TwoVsTwo(disable_progress_bar=False, log_dir=str(log_dir))
    env.py_vs_py(
        challenger=challenger,
        champion=champion,
        seed_start=(seed, KEY),
        seed_count=games // 2,
    )


def summarize(log_dir, name):
    stat = Stat.from_dir(str(log_dir), name, disable_progress_bar=True)
    return {
        'games': stat.game,
        'ranks': [stat.rank_1, stat.rank_2, stat.rank_3, stat.rank_4],
        'avg_rank': stat.avg_rank,
        'avg_pt': stat.avg_pt(PTS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='同一模型两种推理模式 2v2 同台对局')
    ap.add_argument('--ckpt', type=Path, default=BASE_CKPT, help='mortal_base checkpoint 路径')
    ap.add_argument('--games', type=int, default=1000, help='2v2 对局数，须为偶数')
    ap.add_argument('--mode-a', default='policy', choices=['policy', 'search', 'vrisk'])
    ap.add_argument('--mode-b', default='search', choices=['policy', 'search', 'vrisk'])
    ap.add_argument('--search-k', type=int, default=8)
    ap.add_argument('--search-alpha', type=float, default=0.0, help='Q 混合权重，0=纯事件 rollout')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=None, help='固定种子复现；默认随机')
    ap.add_argument('--log-dir', default=None, help='对局日志目录；显式指定则保留')
    ap.add_argument('--swap', action='store_true', help='角色对调：mode-b 为 challenger，mode-a 为 champion')
    args = ap.parse_args()

    if args.games % 2 != 0:
        ap.error('--games 必须为偶数（每 seed 跑 a/b 两局）')
    device = torch.device(args.device)
    mod = load_evaluate()
    torch.backends.cudnn.benchmark = False
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else ROOT / 'mgrpo' / '_bench_logs' / f'2v2_ps_{datetime.now():%Y%m%d_%H%M%S}'
    )

    eng_a, eng_b = build_engines(mod, device, args.ckpt, args.mode_a, args.mode_b, args.search_k, args.search_alpha)
    challenger, champion = eng_a, eng_b
    if args.swap:
        challenger, champion = champion, challenger
    names = (challenger.name, champion.name)
    log.info('2v2 %d 场（%s vs %s），seed=%d, k=%d alpha=%.2f',
             args.games, *names, seed, args.search_k, args.search_alpha)
    run_games(challenger, champion, seed, args.games, log_dir)

    try:
        for name in names:
            s = summarize(log_dir, name)
            ranks = ', '.join(str(x) for x in s['ranks'])
            print(f'{name:<8} 位次 [{ranks}] 平均顺位 {s["avg_rank"]:.3f} 平均pt {s["avg_pt"]:+.2f}  ({s["games"]} 席次)')
    finally:
        if not args.log_dir:
            shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == '__main__':
    main()