"""bc_16w (mortal_v6) vs baseline_v1 2v2 benchmark：N 场半庄，输出双方位次/顺位/pt/打点统计

bc_16w 为 v6 新架构（ConvNeXt+Transformer+QHead），baseline_v1 为 v4 旧架构（Brain+DQN）
v6 推理模式默认取 checkpoint 的 eval.action_mode（search/greedy/policy），与 evaluate.py 一致
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mgrpo.prelude  # noqa: E402  注入 libriichi 与旧 model 模块

from engine import MortalEngine  # noqa: E402
from libriichi.arena import TwoVsTwo  # noqa: E402
from libriichi.stat import Stat  # noqa: E402
from model import Brain, DQN  # noqa: E402  旧架构（v4 及更早）

log = logging.getLogger(__name__)

PTS = [90, 45, 0, -135]  # 半庄顺位赏
KEY = 0x2A44  # 每局固定 second seed，与 --seed 组合保证可复现
V6_CKPT = ROOT / 'mortal_v6' / 'out' / 'mortal.pth'
V1_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'
V6_NAME = 'bc_16w'
V1_NAME = 'mortal_v1'
ACTION_MODES = ('search', 'greedy', 'policy')


def _load_module(name: str, path: Path):
    """以独立命名空间加载文件模块：先注册 sys.modules，供模块内部 sys.modules[__name__] 自引用"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def load_v6_model():
    """mortal_v6/model.py 以独立命名空间加载，避免与 mgrpo 注入的旧 model 模块名冲突"""
    return _load_module('mortal_v6_model', ROOT / 'mortal_v6' / 'model.py')


@lru_cache(maxsize=1)
def load_v6_config():
    """config_v6.py 独立加载，其内部会把 sys.modules['config'] 注册为 v6 配置供 search.py 使用"""
    return _load_module('config_v6', ROOT / 'mortal_v6' / 'config_v6.py')


@lru_cache(maxsize=1)
def load_v6_search():
    """mortal_v6/search.py 独立命名空间加载，复用官方 search_action"""
    load_v6_config()
    return _load_module('mortal_v6_search', ROOT / 'mortal_v6' / 'search.py')


class SearchEngine(MortalEngine):
    """想象搜索引擎：策略 top-k 候选 + 事件 rollout + XQL Q 混合评分，逐局面搜索"""

    def __init__(self, brain, q_head, event_model, *, search_k: int, search_alpha: float,
                 gamma: float, rewards: list[float], **kwargs):
        super().__init__(brain, q_head, **kwargs)
        self.event_model = event_model.to(self.device).eval()
        self.search_action = load_v6_search().search_action
        self.search_k = search_k
        self.search_alpha = search_alpha
        self.gamma = gamma
        self.rewards = rewards

    def _react_batch(self, obs, masks, invisible_obs):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        with torch.inference_mode():
            phi = self.brain(obs, invisible_obs)
            actions = torch.stack([
                self.search_action(
                    phi[i:i + 1], masks[i:i + 1], self.brain, self.dqn, self.event_model,
                    k=self.search_k, alpha=self.search_alpha, gamma=self.gamma, rewards=self.rewards,
                )
                for i in range(phi.shape[0])
            ])
            q_values = self.dqn(phi, masks)  # (N, A) 每样本定长值向量，供 Rust 侧解析
        return actions.tolist(), q_values.tolist(), masks.tolist(), [True] * phi.shape[0]


class GreedyQEngine(MortalEngine):
    """直出精排引擎：policy top-k 候选 + Q 单步精排（无 rollout）"""

    def __init__(self, brain, q_head, *, top_k: int = 3, **kwargs):
        super().__init__(brain, q_head, **kwargs)
        self.top_k = top_k

    def _react_batch(self, obs, masks, invisible_obs):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        with torch.inference_mode():
            phi = self.brain(obs, invisible_obs)
            logits = self.brain.policy_logits(phi).masked_fill(~masks, -torch.inf)
            k = min(self.top_k, masks.sum(-1).max().item())
            candidates = logits.topk(k).indices  # (N, k)
            q = self.dqn(phi, masks)  # (N, A)
            best = candidates.gather(1, q.gather(1, candidates).argmax(-1, keepdim=True)).squeeze(-1)
        return best.tolist(), q.tolist(), masks.tolist(), [True] * obs.shape[0]


def build_v6_engine(device: torch.device, ckpt: Path, name: str, action_mode: str | None) -> MortalEngine:
    """v6 checkpoint → 引擎，action_mode 为 None 时取 checkpoint 的 eval.action_mode"""
    v6 = load_v6_model()
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    cfg = state['config']
    action_mode = action_mode or cfg['eval']['action_mode']
    brain = v6.Brain(version=cfg['control']['version'], **cfg['model']).eval()
    q_head = v6.QHead(phi_dim=cfg['model']['phi_dim'], **cfg['q_head']).eval()
    brain.load_state_dict(state['mortal'])
    q_head.load_state_dict(state['q_head'])
    common = dict(
        is_oracle=False, version=4, device=device,
        enable_amp=False,  # 评估用 fp32，数值最稳（同 evaluate.py）
        enable_rule_based_agari_guard=True, name=name,
    )
    if action_mode == 'search':
        event = v6.EventModel(phi_dim=cfg['model']['phi_dim'], **cfg['event']).eval()
        event.load_state_dict(state['event_model'])
        ev = cfg['eval']
        return SearchEngine(
            brain, q_head, event,
            search_k=ev['search_k'], search_alpha=ev['search_alpha'],
            gamma=float(cfg['env']['gamma']), rewards=cfg['event_loss']['rewards'],
            **common,
        )
    if action_mode == 'greedy':
        return GreedyQEngine(brain, q_head, top_k=cfg['eval']['greedy_top_k'], **common)
    return MortalEngine(brain, q_head, action_source='policy', **common)


def build_v1_engine(device: torch.device, ckpt: Path, name: str) -> MortalEngine:
    """旧架构 checkpoint → Brain+DQN 引擎，action_source 按是否含 policy_head 决定"""
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
        enable_amp=False,
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
    ap = argparse.ArgumentParser(description='bc_16w (v6) vs baseline_v1 2v2 benchmark')
    ap.add_argument('--games', type=int, default=1000, help='2v2 对局数，须为偶数')
    ap.add_argument('--device', default='cuda:0', help='推理设备，如 cuda:0 / cpu')
    ap.add_argument('--seed', type=int, default=None, help='固定种子复现；默认随机')
    ap.add_argument('--log-dir', default=None, help='对局日志目录；显式指定则保留，否则统计后删除')
    ap.add_argument('--v6-ckpt', type=Path, default=V6_CKPT, help='v6 checkpoint 路径')
    ap.add_argument('--v1-ckpt', type=Path, default=V1_CKPT, help='v1 checkpoint 路径')
    ap.add_argument('--action-mode', choices=ACTION_MODES, default=None,
                    help='v6 推理模式；默认取 checkpoint 的 eval.action_mode')
    ap.add_argument('--swap', action='store_true', help='角色对调：v1 为 challenger，v6 为 champion')
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

    challenger = build_v6_engine(device, args.v6_ckpt, V6_NAME, args.action_mode)
    champion = build_v1_engine(device, args.v1_ckpt, V1_NAME)
    if args.swap:
        challenger, champion = champion, challenger
    names = (challenger.name, champion.name)
    log.info('对局 %d 场 2v2（%s vs %s），seed=%d', args.games, *names, seed)
    run_games(challenger, champion, seed, args.games, log_dir)
    report(log_dir, keep_logs=args.log_dir is not None, names=names)


if __name__ == '__main__':
    main()
