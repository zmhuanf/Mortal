"""mortal_base+baseline_v1 混合架构 vs baseline_v1 2v2 benchmark

混合引擎 = base 三阶段 ConvNeXt+注意力 与 v1 单阶段 ConvNeXt 的 policy logits
按 (1-w):w 加权（默认 0.5/0.5），challenger 与 champion 动作源均为 policy
网络复用 mortal_base/evaluate.py 的 load_model / load_opponent，禁止混用两侧加载方式
libriichi 取 mortal_base 目录的新版 pyd（react_batch 透传 indexes）
"""
import argparse
import importlib.util
import logging
import secrets
import shutil
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / 'mortal_base'
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # libriichi.pyd / model.py / config_base.py 优先取自 mortal_base

from libriichi.arena import TwoVsTwo  # noqa: E402
from libriichi.stat import Stat  # noqa: E402

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
)
log = logging.getLogger(__name__)

PTS = [90, 45, 0, -135]  # 半庄顺位赏
KEY = 0x2A44  # 每局固定 second seed，与 --seed 组合保证可复现
BASE_CKPT = BASE_DIR / 'out' / 'mortal.pth'
V1_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'
V1_NAME = 'baseline_v1'
MIX_NAME = 'mix_base_v1'


class BlendBrain(nn.Module):
    """base 与 v1 双网络混合策略：各自编码得 phi，policy logits 按权重线性加权"""

    def __init__(self, base: nn.Module, v1: nn.Module, *, v1_weight: float = 0.5):
        super().__init__()
        self.base = base
        self.v1 = v1
        self.v1_weight = v1_weight

    def forward(self, obs, invisible_obs=None):
        # v1 网络不消费 invisible_obs，仅 base 侧透传保持签名一致
        return self.base(obs, invisible_obs), self.v1(obs)

    def policy_logits(self, phi):
        base_logits = self.base.policy_logits(phi[0])
        v1_logits = self.v1.policy_logits(phi[1])
        return (1 - self.v1_weight) * base_logits + self.v1_weight * v1_logits


def _load_module(name: str, path: Path):
    """以独立命名空间加载文件模块：先注册 sys.modules，供模块内部 sys.modules[__name__] 自引用"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def load_base_evaluate():
    """mortal_base/evaluate.py 独立加载：其内部 import config_base 注册 sys.modules['config']"""
    return _load_module('mortal_base_evaluate', BASE_DIR / 'evaluate.py')


def build_mix_engine(device: torch.device, base_ckpt: Path, v1_ckpt: Path, *,
                     v1_weight: float, name: str) -> 'MortalEngine':
    """base+v1 混合策略引擎：两网各自按自身结构加载，policy logits 加权后 policy 模式直出"""
    evaluate = load_base_evaluate()
    base_engine = evaluate.load_model(base_ckpt, device)
    v1_engine = evaluate.load_opponent(v1_ckpt, device, V1_NAME)
    blend = BlendBrain(base_engine.brain, v1_engine.brain, v1_weight=v1_weight)
    return evaluate.MortalEngine(
        blend, base_engine.dqn, is_oracle=False, version=base_engine.version, device=device,
        enable_rule_based_agari_guard=True, name=name, action_source='policy',
    )


def build_v1_engine(device: torch.device, ckpt: Path, name: str) -> 'MortalEngine':
    """baseline_v1 checkpoint → 对手引擎，load_opponent 按 config.resnet 单阶段 ConvNeXt 加载"""
    return load_base_evaluate().load_opponent(ckpt, device, name)


def run_games(challenger, champion, seed: int, games: int, log_dir: Path) -> None:
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


def report(log_dir: Path, keep_logs: bool, names: tuple[str, str]) -> None:
    """打印双方统计并清理对局日志"""
    try:
        for name in names:
            s = summarize(log_dir, name)
            ranks = ', '.join(str(x) for x in s['ranks'])
            print(
                f"{name:<12} 位次 [{ranks}]  "
                f"平均顺位 {s['avg_rank']:.3f}  "
                f"平均pt {s['avg_pt']:+.2f} (总 {s['total_pt']:+d})  "
                f"平均打点 {s['avg_point_per_agari']:.1f}  ({s['games']} 席次)"
            )
    finally:
        if not keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description='base+v1 混合架构 vs baseline_v1 2v2 benchmark')
    ap.add_argument('--games', type=int, default=1000, help='2v2 对局数，须为偶数')
    ap.add_argument('--device', default='cuda:0', help='推理设备，如 cuda:0 / cpu')
    ap.add_argument('--seed', type=int, default=None, help='固定种子复现；默认随机')
    ap.add_argument('--log-dir', default=None, help='对局日志目录；显式指定则保留，否则统计后删除')
    ap.add_argument('--base-ckpt', type=Path, default=BASE_CKPT, help='mortal_base checkpoint 路径')
    ap.add_argument('--v1-ckpt', type=Path, default=V1_CKPT, help='baseline_v1 checkpoint 路径')
    ap.add_argument('--v1-weight', type=float, default=0.5, help='混合引擎中 baseline_v1 权重，base 为 1-w')
    ap.add_argument('--swap', action='store_true', help='角色对调：baseline_v1 为 challenger，混合引擎为 champion')
    args = ap.parse_args()

    if args.games % 2 != 0:
        ap.error('--games 必须为偶数（每 seed 跑 a/b 两局）')
    if not 0 <= args.v1_weight <= 1:
        ap.error('--v1-weight 必须在 [0, 1] 区间')
    for ckpt in (args.base_ckpt, args.v1_ckpt):
        if not ckpt.is_file():
            ap.error(f'checkpoint 不存在: {ckpt}')
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False  # 单步固定形状推理，关闭 benchmark 减少抖动
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else ROOT / 'mgrpo' / '_bench_logs' / f'2v2_{datetime.now():%Y%m%d_%H%M%S}'
    )

    challenger = build_mix_engine(device, args.base_ckpt, args.v1_ckpt, v1_weight=args.v1_weight, name=MIX_NAME)
    champion = build_v1_engine(device, args.v1_ckpt, V1_NAME)
    if args.swap:
        challenger, champion = champion, challenger
    names = (challenger.name, champion.name)
    log.info('对局 %d 场 2v2（%s vs %s，v1 权重 %.2f），seed=%d',
             args.games, *names, args.v1_weight, seed)
    run_games(challenger, champion, seed, args.games, log_dir)
    report(log_dir, keep_logs=args.log_dir is not None, names=names)


if __name__ == '__main__':
    main()