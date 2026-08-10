"""mortal_v6 1v3 对战评估：challenger 为 v6 模型（直出或想象搜索），champion 为历史对手"""

import os
import sys
import shutil
import importlib.util
from os import path

ROOT = path.dirname(path.abspath(__file__))
MORTAL_DIR = path.join(ROOT, '..', 'mortal')
sys.path.append(MORTAL_DIR)

import torch
import numpy as np

import config_v6  # 注册 config 模块，必须先于 from config import config
from config import config
from engine import MortalEngine
from model import Brain, QHead, EventModel
from libriichi.arena import OneVsThree
from libriichi.stat import Stat
from search import search_action

_v4_module = None
_v5_module = None


def _v4_model():
    global _v4_module
    if _v4_module is None:
        spec = importlib.util.spec_from_file_location('mortal_model_v4', path.join(MORTAL_DIR, 'model.py'))
        _v4_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_v4_module)
    return _v4_module


def _v5_model():
    global _v5_module
    if _v5_module is None:
        spec = importlib.util.spec_from_file_location('mortal_model_v5', path.join(ROOT, '..', 'mortal_v5', 'model.py'))
        _v5_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_v5_module)
    return _v5_module


class SearchEngine(MortalEngine):
    """想象搜索引擎：策略 top-k 候选 + 事件 rollout + XQL Q 混合评分，逐局面搜索"""

    def __init__(self, brain, q_head, event_model, **kwargs):
        super().__init__(brain, q_head, **kwargs)
        self.event_model = event_model.to(self.device).eval()

    def _react_batch(self, obs, masks, invisible_obs):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        with torch.inference_mode():
            phi = self.brain(obs, invisible_obs)
            actions = torch.stack([
                search_action(phi[i:i + 1], masks[i:i + 1], self.brain, self.dqn, self.event_model)
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


def build_engine(brain, q_head, event_model, device, name, *, action_mode='policy', greedy_top_k=3):
    if action_mode == 'search':
        return SearchEngine(
            brain, q_head, event_model,
            is_oracle=False, version=4, device=device,
            enable_amp=False,  # 评估用 fp32 推理，数值最稳
            enable_rule_based_agari_guard=True,
            name=name,
        )
    if action_mode == 'greedy':
        return GreedyQEngine(
            brain, q_head, top_k=greedy_top_k,
            is_oracle=False, version=4, device=device,
            enable_amp=False,
            enable_rule_based_agari_guard=True,
            name=name,
        )
    return MortalEngine(
        brain, q_head,
        is_oracle=False, version=4, device=device,
        enable_amp=False,
        enable_rule_based_agari_guard=True,
        name=name,
        action_source='policy',
    )


def load_opponent(state_file, device, name):
    state = torch.load(state_file, weights_only=True, map_location='cpu')
    cfg = state['config']
    if 'q_head' in cfg:
        # v6 对手：结构一致，policy 直出
        brain = Brain(version=cfg['control']['version'], **cfg['model']).eval()
        dqn = QHead(phi_dim=cfg['model']['phi_dim'], **cfg['q_head']).eval()
        brain.load_state_dict(state['mortal'])
        if 'q_head' in state:
            dqn.load_state_dict(state['q_head'])
        return MortalEngine(
            brain, dqn,
            is_oracle=False, version=4, device=device, enable_amp=False,
            enable_rule_based_agari_guard=True,
            name=name, action_source='policy',
        )
    if 'model' in cfg:
        # v5 对手：结构不兼容 v6，用独立命名空间加载 v5 模型定义
        v5 = _v5_model()
        brain = v5.Brain(version=cfg['control']['version'], **cfg['model']).eval()
        dqn = v5.DQN(phi_dim=cfg['model']['phi_dim'], **cfg.get('dqn', {})).eval()
        brain.load_state_dict(state['mortal'])
        if 'dqn' in state:
            dqn.load_state_dict(state['dqn'])
        return MortalEngine(
            brain, dqn,
            is_oracle=False, version=4, device=device, enable_amp=False,
            enable_rule_based_agari_guard=True,
            name=name, action_source='policy' if 'policy_head.weight' in state['mortal'] else 'q',
        )
    # v4 及更早对手
    v4 = _v4_model()
    brain = v4.Brain(version=cfg['control']['version'], **cfg['resnet']).eval()
    dqn = v4.DQN(version=cfg['control']['version'], num_heads=cfg.get('dqn', {}).get('num_heads', 1)).eval()
    brain.load_state_dict(state['mortal'], strict=False)
    dqn.load_state_dict(state['current_dqn'])
    return MortalEngine(
        brain, dqn,
        is_oracle=False, version=4, device=device, enable_amp=False,
        enable_rule_based_agari_guard=True,
        name=name, action_source='policy' if 'policy_head.weight' in state['mortal'] else 'q',
    )


def run_eval(mortal, q_head=None, event_model=None, device=None, *, name='mortal_v6',
             games=None, opponents=None, log_dir=None, action_mode=None):
    device = device or torch.device(config['control']['device'])
    games = games or config['eval']['games']
    opponents = opponents or config['eval']['opponents']
    log_dir = log_dir or config['eval']['log_dir']
    action_mode = action_mode or config['eval']['action_mode']

    if q_head is None:
        q_head = QHead(phi_dim=config['model']['phi_dim'], **config['q_head']).eval()
    if event_model is None:
        event_model = EventModel(phi_dim=config['model']['phi_dim'], **config['event']).eval()
    engine_chal = build_engine(mortal, q_head, event_model, device, name, action_mode=action_mode)

    torch.backends.cudnn.benchmark = False
    if path.isdir(log_dir):
        shutil.rmtree(log_dir)
    seed_count = max(1, games // 4)
    per_opp = max(1, seed_count // len(opponents))
    results = []
    for i, op in enumerate(opponents):
        sub_dir = path.join(log_dir, f'op_{i:02d}')
        env = OneVsThree(disable_progress_bar=False, log_dir=sub_dir)
        rankings, _ = env.py_vs_py(
            challenger=engine_chal,
            champion=load_opponent(op['state_file'], device, op['name']),
            seed_start=(10000 + i * per_opp, 0x2000),
            seed_count=per_opp,
        )
        stat = Stat.from_dir(sub_dir, name)
        results.append((op['name'], rankings, stat))
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']

    totals = [0, 0, 0, 0]
    for _, rankings, _ in results:
        for k in range(4):
            totals[k] += rankings[k]
    total = max(1, sum(totals))
    avg_rank = sum((i + 1) * c for i, c in enumerate(totals)) / total
    avg_pt = sum(p * c for p, c in zip([90, 45, 0, -135], totals)) / total
    return {'avg_rank': avg_rank, 'avg_pt': avg_pt, 'results': results}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='mortal_v6 1v3 评估')
    parser.add_argument('--state-file', default=None, help='评估用 checkpoint，默认取 control.state_file')
    parser.add_argument('--games', type=int, default=None, help='覆盖 eval.games')
    parser.add_argument('--action-mode', default=None, choices=['search', 'greedy', 'policy'], help='覆盖 eval.action_mode')
    args = parser.parse_args()

    device = torch.device(config['control']['device'])
    state_file = args.state_file or config['control']['state_file']
    state = torch.load(state_file, weights_only=True, map_location=device)
    mortal = Brain(version=config['control']['version'], **config['model']).to(device).eval()
    mortal.load_state_dict(state['mortal'])
    q_head = QHead(phi_dim=config['model']['phi_dim'], **config['q_head']).to(device).eval()
    if 'q_head' in state:
        q_head.load_state_dict(state['q_head'])
    event_model = EventModel(phi_dim=config['model']['phi_dim'], **config['event']).to(device).eval()
    if 'event_model' in state:
        event_model.load_state_dict(state['event_model'])
    result = run_eval(mortal, q_head, event_model, device, games=args.games, action_mode=args.action_mode)
    print(f'avg rank: {result["avg_rank"]:.4}')
    print(f'avg pt: {result["avg_pt"]:.4}')
    for name, rankings, stat in result['results']:
        op_avg = sum((i + 1) * c for i, c in enumerate(rankings)) / max(1, sum(rankings))
        print(f'  vs {name}: {rankings} ({op_avg:.4})')
