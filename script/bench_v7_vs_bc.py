"""mortal_v7 (DecisionTransformer) vs mortal_bc 2v2 benchmark：N 场半庄，输出双方位次/顺位/pt/打点统计

v7 引擎复用 mortal_v7/evaluate.py 的 DTEngine（按 index 维护局窗口，RTG 实时更新，含规则和牌/立直兜底）；
bc 为 v6 风格三阶段 ConvNeXt+Transformer 注意力池化纯 BC 模型，BcEngine 纯策略 argmax；
libriichi 取 mortal_v7 目录新版 pyd（react_batch 透传 indexes），BcEngine 签名不含 indexes 故包一层适配
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

ROOT = Path(__file__).resolve().parent.parent
V7_DIR = ROOT / 'mortal_v7'
BC_DIR = ROOT / 'mortal_bc'
if str(V7_DIR) not in sys.path:
    sys.path.insert(0, str(V7_DIR))  # 新版 libriichi.pyd（透传 indexes）与 v7 模块优先

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
V7_CKPT = V7_DIR / 'out' / 'mortal.pth'
BC_CKPT = BC_DIR / 'out' / 'mortal.pth'
V7_NAME = 'mortal_v7'
BC_NAME = 'mortal_bc'


def _load_module(name: str, path: Path):
    """以独立命名空间加载文件模块：先注册 sys.modules，供模块内部 sys.modules[__name__] 自引用"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def load_v7_evaluate():
    """mortal_v7/evaluate.py 独立加载：其内部按需 import config_v7 与 model，二者取自 v7 目录"""
    return _load_module('mortal_v7_evaluate', V7_DIR / 'evaluate.py')


@lru_cache(maxsize=1)
def load_bc_model():
    """mortal_bc/model.py 独立命名空间加载，避免与 v7 的 model 模块名冲突"""
    return _load_module('mortal_bc_model', BC_DIR / 'model.py')


@lru_cache(maxsize=1)
def load_bc_engine():
    """mortal_bc/engine.py 独立加载：仅依赖 numpy/torch，无模块名冲突"""
    return _load_module('mortal_bc_engine', BC_DIR / 'engine.py')


def build_v7_engine(device: torch.device, ckpt: Path, name: str):
    """v7 checkpoint → DTEngine：一次加载 state，模型与 RTG 窗口参数取自 config"""
    ev = load_v7_evaluate()
    state = torch.load(ckpt, weights_only=False, map_location='cpu')
    model = ev.DecisionTransformer(**state['config']['model'])
    model.load_state_dict(state['model'])
    rtg = state['config']['rtg']
    return ev.DTEngine(
        model, device,
        window=rtg['window'], score_scale=rtg['score_scale'], target_score=rtg['target_score'],
        name=name,
    )


def build_bc_engine(device: torch.device, ckpt: Path, name: str):
    """bc checkpoint → 策略直出引擎：纯 argmax，包一层适配新版 pyd 的 indexes 透传"""
    bc_model = load_bc_model()
    bc_engine = load_bc_engine()
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    cfg = state['config']
    brain = bc_model.Brain(version=cfg['control']['version'], **cfg['model']).to(device).eval()
    brain.load_state_dict(state['mortal'])

    class _IndexesAwareEngine(bc_engine.BcEngine):
        # 新版 pyd 固定传 4 个位置参数，BcEngine 每步独立编码无窗口，indexes 直接忽略
        def react_batch(self, obs, masks, invisible_obs, indexes=None):
            return super().react_batch(obs, masks, invisible_obs)

    return _IndexesAwareEngine(
        brain,
        version=cfg['control']['version'],
        device=device,
        enable_amp=False,  # fp32 数值最稳，与 v7 bench 的对手侧一致
        name=name,
    )


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
    ap = argparse.ArgumentParser(description='mortal_v7 vs mortal_bc 2v2 benchmark')
    ap.add_argument('--games', type=int, default=1000, help='2v2 对局数，须为偶数')
    ap.add_argument('--device', default='cuda:0', help='推理设备，如 cuda:0 / cpu')
    ap.add_argument('--seed', type=int, default=None, help='固定种子复现；默认随机')
    ap.add_argument('--log-dir', default=None, help='对局日志目录；显式指定则保留，否则统计后删除')
    ap.add_argument('--v7-ckpt', type=Path, default=V7_CKPT, help='mortal_v7 checkpoint 路径')
    ap.add_argument('--bc-ckpt', type=Path, default=BC_CKPT, help='mortal_bc checkpoint 路径')
    ap.add_argument('--swap', action='store_true', help='角色对调：bc 为 challenger，v7 为 champion')
    args = ap.parse_args()

    if args.games % 2 != 0:
        ap.error('--games 必须为偶数（每 seed 跑 a/b 两局）')
    for ckpt in (args.v7_ckpt, args.bc_ckpt):
        if not ckpt.is_file():
            ap.error(f'checkpoint 不存在: {ckpt}')
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False  # 窗口推理固定形状，关闭 benchmark 减少抖动
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else ROOT / 'mgrpo' / '_bench_logs' / f'2v2_{datetime.now():%Y%m%d_%H%M%S}'
    )

    challenger = build_v7_engine(device, args.v7_ckpt, V7_NAME)
    champion = build_bc_engine(device, args.bc_ckpt, BC_NAME)
    if args.swap:
        challenger, champion = champion, challenger
    names = (challenger.name, champion.name)
    log.info('对局 %d 场 2v2（%s vs %s），seed=%d', args.games, *names, seed)
    run_games(challenger, champion, seed, args.games, log_dir)
    report(log_dir, keep_logs=args.log_dir is not None, names=names)


if __name__ == '__main__':
    main()
