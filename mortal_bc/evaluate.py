"""1v3 评估：BC 模型 vs baseline_v1"""
import os
import sys
import shutil
from os import path

ROOT = path.dirname(path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
from libriichi.arena import OneVsThree
from libriichi.stat import Stat
from config import config
from model import Brain, LegacyBrain
from engine import BcEngine

# 顺位点，与原版评估口径一致
PTS = [90.0, 45.0, 0.0, -135.0]


def load_opponent_engine(state_file, device, enable_amp):
    """加载 baseline checkpoint 为 BcEngine，用 LegacyBrain 匹配旧架构"""
    s = torch.load(state_file, map_location='cpu', weights_only=True)
    cfg = s['config']
    version = cfg['control'].get('version', 4)
    resnet = cfg['resnet']
    brain = LegacyBrain(
        version=version,
        conv_channels=resnet['conv_channels'],
        num_blocks=resnet['num_blocks'],
        layer_scale=resnet.get('layer_scale', 1e-6),
        drop_rate=resnet.get('drop_rate', 0.0),
    ).eval()
    brain.load_state_dict(s['mortal'], strict=False)
    return BcEngine(brain, version=version, device=device, enable_amp=enable_amp, name=config['eval']['opponent_name'])


def run_eval(mortal, device):
    """challenger=mortal 对战 baseline_v1，返回 avg_rank/avg_pt"""
    ctrl = config['control']
    eval_cfg = config['eval']
    log_dir = path.abspath(ctrl['eval_log_dir'])
    if path.isdir(log_dir):
        shutil.rmtree(log_dir)

    chal = BcEngine(mortal, version=ctrl['version'], device=device,
                    enable_amp=ctrl['enable_amp'], name='mortal')
    champ = load_opponent_engine(eval_cfg['opponent_state_file'], device, ctrl['enable_amp'])

    torch.backends.cudnn.benchmark = False
    env = OneVsThree(disable_progress_bar=False, log_dir=log_dir)
    rankings, _ = env.py_vs_py(
        challenger=chal, champion=champ,
        seed_start=(10000, 0x2000), seed_count=eval_cfg['games'] // 4,
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
    print(f"avg_rank={r['avg_rank']:.4f} avg_pt={r['avg_pt']:.4f} totals={r['totals']}")
