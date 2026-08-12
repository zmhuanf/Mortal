"""mortal_base 评估：1v3 对战 vs 固定 v4 对手（默认 baseline_v1）

challenger 与 opponent 均为 MortalEngine，动作源按有无 policy_head 判定
"""

import os
import shutil
from os import path

import torch
from torch import nn

import config_base  # 注册 config 模块，必须先于 from config import config
from config import config
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from libriichi.stat import Stat
from libriichi.consts import obs_shape, ACTION_SPACE


class _V4ConvNeXtBlock(nn.Module):
    """baseline_v1 训练架构的 ConvNeXtBlock：k7 单 DWConv（与 mortal/model.py 一致）"""

    def __init__(self, channels, *, layer_scale=1e-6, drop_rate=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.actv = nn.GELU()
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2))
        x = self.pwconv1(x)
        x = self.actv(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = self.drop(x)
        return residual + x.transpose(1, 2)


class _V4Encoder(nn.Module):
    """baseline_v1 编码器：单阶段 ConvNeXt，键名 encoder.net.* 与 checkpoint 严格一致"""

    def __init__(self, in_channels, conv_channels, num_blocks, *, layer_scale=1e-6, drop_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            *[_V4ConvNeXtBlock(conv_channels, layer_scale=layer_scale, drop_rate=drop_rate)
              for _ in range(num_blocks)],
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
        )

    def forward(self, x):
        return self.net(x)


class V4Brain(nn.Module):
    """baseline 等 v4 对手的策略网络：ConvNeXt 编码 + GELU + policy 头"""

    def __init__(self, *, conv_channels, num_blocks, layer_scale=1e-6, drop_rate=0.0, **kwargs):
        super().__init__()
        self.encoder = _V4Encoder(
            obs_shape(4)[0], conv_channels, num_blocks,
            layer_scale=layer_scale, drop_rate=drop_rate,
        )
        self.actv = nn.GELU()
        self.policy_head = nn.Linear(1024, ACTION_SPACE)

    def forward(self, obs, invisible_obs=None):
        return self.actv(self.encoder(obs))

    def policy_logits(self, phi):
        return self.policy_head(phi)


class V4DQN(nn.Module):
    """baseline 等 v4 对手的 Q 网络，version 4 单线性层"""

    def __init__(self, *, num_heads=1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.net = nn.Linear(1024, num_heads * (1 + ACTION_SPACE))
        nn.init.constant_(self.net.bias, 0)

    def forward(self, phi, masks):
        v, a = self.net(phi).split((self.num_heads, self.num_heads * ACTION_SPACE), dim=-1)
        v = v.view(-1, self.num_heads, 1)
        a = a.view(-1, self.num_heads, ACTION_SPACE)
        masks = masks.unsqueeze(1)
        a_sum = a.masked_fill(~masks, 0.).sum(-1, keepdim=True)
        mask_sum = masks.sum(-1, keepdim=True)
        a_mean = a_sum / mask_sum
        q = (v + a - a_mean).masked_fill(~masks, -torch.inf)
        return q


def make_engine(mortal, dqn, device, version, *, name='mortal_base', enable_amp=False):
    """用内存中的 mortal/dqn 构造挑战者引擎，train/eval 共用"""
    amp_dtype = torch.bfloat16 if config['control'].get('amp_dtype', 'float16') == 'bfloat16' else torch.float16
    return MortalEngine(mortal, dqn, is_oracle=False, version=version, device=device,
                        enable_amp=enable_amp, enable_rule_based_agari_guard=True,
                        name=name, action_source='policy', amp_dtype=amp_dtype)


def load_model(state_file, device):
    """加载 mortal_base 自身 checkpoint 为 challenger 引擎"""
    state = torch.load(state_file, weights_only=False, map_location=device)
    cfg = state['config']
    mortal = Brain(version=cfg['control']['version'], **cfg['model']).to(device).eval()
    mortal.load_state_dict(state['mortal'])
    dqn = DQN(version=cfg['control']['version'], **cfg['dqn']).to(device).eval()
    dqn.load_state_dict(state['current_dqn'])
    return make_engine(mortal, dqn, device, cfg['control']['version'])


def load_opponent(state_file, device, name):
    """baseline_v1 等 v4 对手，strict=False 容忍结构差异，action_source 按有无 policy_head 判定"""
    state = torch.load(state_file, weights_only=True, map_location='cpu')
    cfg = state['config']
    brain = V4Brain(version=cfg['control']['version'], **cfg['resnet']).to(device).eval()
    brain.load_state_dict(state['mortal'], strict=False)
    dqn = V4DQN(num_heads=cfg.get('dqn', {}).get('num_heads', 1)).to(device).eval()
    dqn.load_state_dict(state['current_dqn'], strict=False)
    action_source = 'policy' if 'policy_head.weight' in state['mortal'] else 'q'
    return MortalEngine(brain, dqn, is_oracle=False, version=4, device=device,
                        enable_rule_based_agari_guard=True, name=name, action_source=action_source)


def run_eval(model, device=None, *, games=None, opponents=None, log_dir=None):
    device = device or torch.device(config['control']['device'])
    games = games or config['eval']['games']
    opponents = opponents or config['eval']['opponents']
    log_dir = log_dir or config['eval']['log_dir']
    segment = config['eval']['segment_seeds']

    torch.backends.cudnn.benchmark = False
    if path.isdir(log_dir):
        shutil.rmtree(log_dir)
    seed_count = max(1, games // 4)
    per_opp = max(1, seed_count // len(opponents))
    results = []
    for i, op in enumerate(opponents):
        sub_dir = path.join(log_dir, f'op_{i:02d}')
        champ = load_opponent(op['state_file'], device, op['name'])
        totals = [0, 0, 0, 0]
        # 分段评估：窗口内存随 seed 数线性增长，段间重建引擎释放
        for seg_start in range(0, per_opp, segment):
            seg_count = min(segment, per_opp - seg_start)
            env = OneVsThree(disable_progress_bar=False, log_dir=sub_dir)
            rankings, _ = env.py_vs_py(
                challenger=model,
                champion=champ,
                seed_start=(10000 + i * per_opp + seg_start, 0x2000),
                seed_count=seg_count,
            )
            for k in range(4):
                totals[k] += rankings[k]
        stat = Stat.from_dir(sub_dir, 'mortal_base')
        results.append((op['name'], totals, stat))
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

    parser = argparse.ArgumentParser(description='mortal_base 1v3 评估')
    parser.add_argument('--state-file', default=None, help='评估用 checkpoint，默认取 control.state_file')
    parser.add_argument('--games', type=int, default=None, help='覆盖 eval.games')
    args = parser.parse_args()

    device = torch.device(config['control']['device'])
    state_file = args.state_file or config['control']['state_file']
    model = load_model(state_file, device)
    result = run_eval(model, device, games=args.games)
    print(f'avg rank: {result["avg_rank"]:.4}')
    print(f'avg pt: {result["avg_pt"]:.4}')
    for name, rankings, stat in result['results']:
        op_avg = sum((i + 1) * c for i, c in enumerate(rankings)) / max(1, sum(rankings))
        print(f'  vs {name}: {rankings} ({op_avg:.4})')
