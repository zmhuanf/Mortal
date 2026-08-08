"""bc.pth vs mortal.pth 2v2 benchmark：N 场半庄对局，输出双方位次/顺位/pt/打点统计"""
import argparse
import logging
import secrets
import shutil
import sys
from dataclasses import asdict
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
from mgrpo.agent.engine import OpponentEngine  # noqa: E402
from mgrpo.config import ENV, MODEL  # noqa: E402
from mgrpo.model.brain import PolicyNet  # noqa: E402
from model import Brain, DQN  # noqa: E402

log = logging.getLogger(__name__)

BC_CKPT = ROOT / 'mgrpo' / 'ckpt' / 'bc.pth'
MORTAL_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'
PTS = [90, 45, 0, -135]  # 半庄顺位赏
KEY = 0x2A44  # 每局固定 second seed，与 --seed 组合保证可复现


def build_bc_engine(device: torch.device) -> OpponentEngine:
    """bc.pth → PolicyNet 贪心引擎，评估用确定性策略"""
    state = torch.load(BC_CKPT, weights_only=True, map_location='cpu')
    net = PolicyNet(version=ENV.version, **asdict(MODEL))
    net.load_state_dict(state['model'])
    return OpponentEngine(net, device=device, name='bc', version=ENV.version)


def build_mortal_engine(device: torch.device) -> MortalEngine:
    """mortal.pth → Brain+DQN 旧架构引擎，action_source 按 checkpoint 是否含 policy_head 决定"""
    state = torch.load(MORTAL_CKPT, weights_only=True, map_location='cpu')
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
        name='mortal',
        action_source='policy' if 'policy_head.weight' in state['mortal'] else 'q',
    )


def run_games(bc_engine, mortal_engine, seed: int, games: int, log_dir: Path) -> None:
    """TwoVsTwo 执行 games 场，每 seed 分 a/b 两局轮换双方座位消除座位偏差"""
    env = TwoVsTwo(disable_progress_bar=False, log_dir=str(log_dir))
    env.py_vs_py(
        challenger=bc_engine,
        champion=mortal_engine,
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
        for name in ('bc', 'mortal'):
            s = summarize(log_dir, name)
            ranks = ', '.join(str(x) for x in s['ranks'])
            print(
                f"{name:<8} 位次 [{ranks}]  "
                f"平均顺位 {s['avg_rank']:.3f}  "
                f"平均pt {s['avg_pt']:+.2f} (总 {s['total_pt']:+d})  "
                f"平均打点 {s['avg_point_per_agari']:.1f}  ({s['games']} 席次)"
            )
    finally:
        if not keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description='bc.pth vs mortal.pth 2v2 benchmark')
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

    bc_engine = build_bc_engine(device)
    mortal_engine = build_mortal_engine(device)
    log.info('对局 %d 场 2v2（bc vs mortal），seed=%d', args.games, seed)
    run_games(bc_engine, mortal_engine, seed, args.games, log_dir)
    report(log_dir, keep_logs=args.log_dir is not None)


if __name__ == '__main__':
    main()
