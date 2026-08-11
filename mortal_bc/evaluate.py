"""1v3 评估：BC 模型 vs baseline_v1

challenger 用 BcEngine 纯策略直出，champion 用原版 MortalEngine+DQN 正确加载 baseline_v1
旧版模型用独立命名空间加载，避免与本目录 model 模块重名冲突"""
import importlib.util
import os
import sys
import shutil
from os import path

ROOT = path.dirname(path.abspath(__file__))
sys.path.insert(0, ROOT)
MORTAL_DIR = path.join(ROOT, '..', 'mortal')

import torch
from libriichi.arena import OneVsThree
from libriichi.stat import Stat
from config import config
from model import Brain
from engine import BcEngine

PTS = [90.0, 45.0, 0.0, -135.0]
CHAL_NAME = 'mortal_bc'
CHAMP_NAME = config['eval']['opponent_name']


def _load_legacy_modules():
    """独立命名空间加载 mortal/ 下 engine 与 model，避免与本目录 model 冲突"""
    spec_model = importlib.util.spec_from_file_location('mortal_legacy_model', path.join(MORTAL_DIR, 'model.py'))
    mod_model = importlib.util.module_from_spec(spec_model)
    sys.modules['mortal_legacy_model'] = mod_model
    spec_model.loader.exec_module(mod_model)

    # mortal/engine.py 顶部 from model import ...，临时把当前 mortal_legacy_model 暴露为 model
    saved = sys.modules.get('model')
    sys.modules['model'] = mod_model
    try:
        spec_engine = importlib.util.spec_from_file_location('mortal_legacy_engine', path.join(MORTAL_DIR, 'engine.py'))
        mod_engine = importlib.util.module_from_spec(spec_engine)
        spec_engine.loader.exec_module(mod_engine)
    finally:
        if saved is not None:
            sys.modules['model'] = saved
        else:
            sys.modules.pop('model', None)
    return mod_model, mod_engine


def load_opponent_engine(state_file, device, enable_amp, name):
    """加载 baseline checkpoint 为原版 MortalEngine

    action_source 按是否含 policy_head 决定，与 bench 脚本一致
    """
    legacy_model, legacy_engine = _load_legacy_modules()
    s = torch.load(state_file, map_location='cpu', weights_only=True)
    cfg = s['config']
    version = cfg['control'].get('version', 4)
    brain = legacy_model.Brain(version=version, **cfg['resnet']).eval()
    dqn = legacy_model.DQN(version=version, num_heads=cfg.get('dqn', {}).get('num_heads', 1)).eval()
    brain.load_state_dict(s['mortal'], strict=False)
    dqn.load_state_dict(s['current_dqn'])
    action_source = 'policy' if 'policy_head.weight' in s['mortal'] else 'q'
    return legacy_engine.MortalEngine(
        brain, dqn,
        is_oracle=False, version=version, device=device,
        enable_amp=enable_amp,
        enable_rule_based_agari_guard=True,
        name=name, action_source=action_source,
    )


def run_eval(mortal, device):
    """challenger=mortal 1v3 对战 baseline_v1，返回 avg_rank/avg_pt"""
    ctrl = config['control']
    eval_cfg = config['eval']
    log_dir = path.abspath(ctrl['eval_log_dir'])
    if path.isdir(log_dir):
        shutil.rmtree(log_dir)

    chal = BcEngine(mortal, version=ctrl['version'], device=device,
                    enable_amp=ctrl['enable_amp'], name=CHAL_NAME)
    champ = load_opponent_engine(eval_cfg['opponent_state_file'], device, ctrl['enable_amp'], CHAMP_NAME)

    torch.backends.cudnn.benchmark = False
    env = OneVsThree(disable_progress_bar=False, log_dir=log_dir)
    rankings, _ = env.py_vs_py(
        challenger=chal, champion=champ,
        seed_start=(10000, 0x2000),
        seed_count=eval_cfg['games'] // 4,
    )
    torch.backends.cudnn.benchmark = ctrl['enable_cudnn_benchmark']

    # rankings 为长度 4 的名次计数序列，rankings[i] = 第 i+1 名出现次数
    totals = [int(rankings[i]) for i in range(4)]
    total = sum(totals) or 1
    avg_rank = sum((i + 1) * c for i, c in enumerate(totals)) / total
    avg_pt = sum(p * c for p, c in zip(PTS, totals)) / total
    return {'avg_rank': avg_rank, 'avg_pt': avg_pt, 'totals': totals}


if __name__ == '__main__':
    import logging
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format='%(asctime)s %(levelname)8s %(message)s')
    ctrl = config['control']
    device = torch.device(ctrl['device'])
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ctrl['best_state_file']
    s = torch.load(ckpt, map_location=device, weights_only=True)
    mortal = Brain(version=ctrl['version'], **config['model']).to(device)
    mortal.load_state_dict(s['mortal'])
    r = run_eval(mortal, device)
    print(f"{CHAL_NAME}: avg_rank={r['avg_rank']:.4f} avg_pt={r['avg_pt']:.4f} totals={r['totals']}")
