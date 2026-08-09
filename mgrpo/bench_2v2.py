"""mortal_v4_offline_baseline vs baseline_v1 2v2 benchmark：N 场半庄对局，输出双方位次/顺位/pt/打点统计"""
import argparse
import logging
import secrets
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mgrpo.prelude  # noqa: E402  加载 libriichi 与旧模型模块

from engine import MortalEngine  # noqa: E402
from libriichi.arena import TwoVsTwo  # noqa: E402
from libriichi.stat import Stat  # noqa: E402
from model import Brain, DQN  # noqa: E402

log = logging.getLogger(__name__)

PTS = [90, 45, 0, -135]  # 半庄顺位赏
KEY = 0x2A44  # 每局固定 second seed，与 --seed 组合保证可复现
# (checkpoint, 玩家名) 第一项为 challenger，a/b 两局轮换座位消除座位偏差
MORTAL_CKPTS: tuple[tuple[Path, str], ...] = (
    (ROOT / 'mortal' / 'mortal_v4_offline_baseline' / 'mortal.pth', 'mortal_v4'),
    (ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth', 'mortal_v1'),
)


def build_mortal_engine(device: torch.device, ckpt: Path, name: str) -> MortalEngine:
    """mortal.pth → Brain+DQN 旧架构引擎，action_source 按 checkpoint 是否含 policy_head 决定"""
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    cfg = state['config']
    version = cfg['control'].get('version', 4)
    brain = Brain(version=version, **cfg['resnet']).eval()
    dqn = DQN(version=version, num_heads=cfg.get('dqn', {}).get('num_heads', 1)).eval()
    brain.load_state_dict(state['mortal'], strict=False)
    dqn.load_state_dict(state['current_dqn'])
    return MortalEngine(
        brain,
        dqn,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=cfg['control'].get('enable_amp', False) and device.type == 'cuda',
        enable_rule_based_agari_guard=True,
        name=name,
        action_source='policy' if 'policy_head.weight' in state['mortal'] else 'q',
    )


def run_games(challenger: MortalEngine, champion: MortalEngine, seed: int, games: int, log_dir: Path) -> None:
    """TwoVsTwo 执行 games 场，每 seed 分 a/b 两局轮换双方座位消除座位偏差"""
    env = TwoVsTwo(disable_progress_bar=False, log_dir=str(log_dir))
    env.py_vs_py(
        challenger=challenger,
        champion=champion,
        seed_start=(seed, KEY),
        seed_count=games // 2,
    )


def summarize(log_dir: Path, name: str) -> dict:
    """按玩家名聚合同名两席的统计量"""
    stat = Stat.from_dir(str(log_dir), name, disable_progress_bar=True)
    return {
        'games': stat.game,
        'ranks': [stat.rank_1, stat.rank_2, stat.rank_3, stat.rank_4],
        'avg_rank': stat.avg_rank,
        'total_pt': stat.total_pt(PTS),
        'avg_pt': stat.avg_pt(PTS),
        'avg_point_per_agari': stat.avg_point_per_agari,
    }


def report(log_dir: Path, keep_logs: bool) -> None:
    """打印双方统计并清理对局日志"""
    try:
        for _, name in MORTAL_CKPTS:
            s = summarize(log_dir, name)
            ranks = ', '.join(str(x) for x in s['ranks'])
            print(
                f"{name:<10} 位次 [{ranks}]  "
                f"平均顺位 {s['avg_rank']:.3f}  "
                f"平均pt {s['avg_pt']:+.2f} (总 {s['total_pt']:+d})  "
                f"平均打点 {s['avg_point_per_agari']:.1f}  ({s['games']} 席次)"
            )
    finally:
        if not keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description='mortal_v4 vs baseline_v1 2v2 benchmark')
    ap.add_argument('--games', type=int, default=2000, help='2v2 对局数，须为偶数')
    ap.add_argument('--device', default='cuda:0', help='推理设备，如 cuda:0 / cpu')
    ap.add_argument('--seed', type=int, default=None, help='固定种子复现；默认随机')
    ap.add_argument('--log-dir', default=None, help='对局日志目录；显式指定则保留，否则统计后删除')
    args = ap.parse_args()

    if args.games % 2 != 0:
        ap.error('--games 必须为偶数（每 seed 跑 a/b 两局）')
    device = torch.device(args.device)
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else ROOT / 'mgrpo' / '_bench_logs' / f'2v2_{datetime.now():%Y%m%d_%H%M%S}'
    )

    (v4_ckpt, v4_name), (v1_ckpt, v1_name) = MORTAL_CKPTS
    challenger = build_mortal_engine(device, v4_ckpt, v4_name)
    champion = build_mortal_engine(device, v1_ckpt, v1_name)
    log.info('对局 %d 场 2v2（%s vs %s），seed=%d', args.games, v4_name, v1_name, seed)
    run_games(challenger, champion, seed, args.games, log_dir)
    report(log_dir, keep_logs=args.log_dir is not None)


if __name__ == '__main__':
    main()
