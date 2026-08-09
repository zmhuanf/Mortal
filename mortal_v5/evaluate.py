"""mortal_v5 1v3 对战评估：challenger 为 v5 模型，champion 为历史对手"""

import os
import sys
import shutil
import importlib.util
from os import path

ROOT = path.dirname(path.abspath(__file__))
MORTAL_DIR = path.join(ROOT, '..', 'mortal')
# 追加而非前置：mortal_v5 同目录文件（model/dataloader）优先，libriichi 等仍可解析
sys.path.append(MORTAL_DIR)

import torch

import config_v5  # 注册 config 模块，必须先于 from config import config
from config import config
from engine import MortalEngine
from model import Brain, DQN
from libriichi.arena import OneVsThree
from libriichi.stat import Stat

_v4_module = None


def _v4_model():
    global _v4_module
    if _v4_module is None:
        # 独立命名空间加载 v4 model，避免与 v5 的 model 模块冲突
        spec = importlib.util.spec_from_file_location('mortal_model_v4', path.join(MORTAL_DIR, 'model.py'))
        _v4_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_v4_module)
    return _v4_module


def build_engine(brain, dqn, device, name, *, action_source='policy'):
    return MortalEngine(
        brain,
        dqn,
        is_oracle=False,
        version=4,
        device=device,
        enable_amp=False,  # 评估用 fp32 推理，数值最稳
        name=name,
        action_source=action_source,
    )


def load_opponent(state_file, device, name):
    state = torch.load(state_file, weights_only=True, map_location='cpu')
    cfg = state['config']
    action_source = 'policy' if 'policy_head.weight' in state['mortal'] else 'q'
    if 'model' in cfg:
        # v5 对手
        brain = Brain(version=cfg['control']['version'], **cfg['model']).eval()
        dqn = DQN(phi_dim=cfg['model']['phi_dim'], **cfg.get('dqn', {})).eval()
        brain.load_state_dict(state['mortal'])
        if 'dqn' in state:
            dqn.load_state_dict(state['dqn'])
    else:
        # v4 及更早对手
        v4 = _v4_model()
        brain = v4.Brain(version=cfg['control']['version'], **cfg['resnet']).eval()
        dqn = v4.DQN(version=cfg['control']['version'], num_heads=cfg.get('dqn', {}).get('num_heads', 1)).eval()
        brain.load_state_dict(state['mortal'], strict=False)
        dqn.load_state_dict(state['current_dqn'])
    return build_engine(brain, dqn, device, name, action_source=action_source)


def run_eval(mortal, dqn=None, device=None, *, name='mortal', games=None, opponents=None, log_dir=None):
    device = device or torch.device(config['control']['device'])
    games = games or config['eval']['games']
    opponents = opponents or config['eval']['opponents']
    log_dir = log_dir or config['eval']['log_dir']

    if dqn is None:
        dqn = DQN(phi_dim=config['model']['phi_dim'], **config['dqn']).eval()
    engine_chal = build_engine(mortal, dqn, device, name)

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

    parser = argparse.ArgumentParser(description='mortal_v5 1v3 评估')
    parser.add_argument('--state-file', default=None, help='评估用 checkpoint，默认取 control.state_file')
    parser.add_argument('--games', type=int, default=None, help='覆盖 eval.games')
    args = parser.parse_args()

    device = torch.device(config['control']['device'])
    state_file = args.state_file or config['control']['state_file']
    state = torch.load(state_file, weights_only=True, map_location=device)
    mortal = Brain(version=config['control']['version'], **config['model']).to(device).eval()
    mortal.load_state_dict(state['mortal'])
    dqn = DQN(phi_dim=config['model']['phi_dim'], **config['dqn']).to(device).eval()
    if 'dqn' in state:
        dqn.load_state_dict(state['dqn'])
    result = run_eval(mortal, dqn, device, games=args.games)
    print(f'avg rank: {result["avg_rank"]:.4}')
    print(f'avg pt: {result["avg_pt"]:.4}')
    for name, rankings, stat in result['results']:
        op_avg = sum((i + 1) * c for i, c in enumerate(rankings)) / max(1, sum(rankings))
        print(f'  vs {name}: {rankings} ({op_avg:.4})')
