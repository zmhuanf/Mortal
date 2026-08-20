"""mortal_base_v2 评估：1v3 对战 vs 固定 baseline_v1

challenger 为 v2 纯策略引擎；opponent 为 baseline_v1（V4 架构），load_opponent 复用其结构
"""

import os
import shutil
from os import path

import torch
from torch import nn

import config  # noqa: F401  注册 sys.modules['config']
from config import config
from model import Brain
from engine import PolicyEngine
from libriichi.arena import OneVsThree
from libriichi.stat import Stat
from libriichi.consts import obs_shape, ACTION_SPACE


class _V4ConvNeXtBlock(nn.Module):
    """baseline_v1 训练架构的 ConvNeXtBlock：k7 单 DWConv"""

    def __init__(self, channels, *, layer_scale=1e-6, drop_rate=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
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


class V4Brain(nn.Module):
    """baseline 等 v4 对手的策略网络：ConvNeXt 编码 + GELU + policy 头"""

    def __init__(self, *, conv_channels, num_blocks, layer_scale=1e-6, drop_rate=0.0, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(obs_shape(4)[0], conv_channels, 3, padding=1, bias=False),
            *[_V4ConvNeXtBlock(conv_channels, layer_scale=layer_scale, drop_rate=drop_rate)
              for _ in range(num_blocks)],
            nn.Conv1d(conv_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
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


def make_engine(brain, device, version, *, name='mortal_base_v2', enable_amp=False):
    amp_dtype = torch.bfloat16 if config['control'].get('amp_dtype', 'float16') == 'bfloat16' else torch.float16
    return PolicyEngine(brain, is_oracle=False, version=version, device=device,
                        enable_amp=enable_amp, enable_rule_based_agari_guard=True,
                        name=name, amp_dtype=amp_dtype)


def load_model(state_file, device):
    """加载 v2 checkpoint（仅 policy）为 challenger 引擎"""
    state = torch.load(state_file, weights_only=False, map_location=device)
    cfg = state['config']
    brain = Brain(version=cfg['control']['version'], **cfg['model']).to(device).eval()
    brain.load_state_dict(state['mortal'])
    return make_engine(brain, device, cfg['control']['version'])


def load_opponent(state_file, device, name):
    """baseline_v1 等 v4 对手，strict=False 容忍结构差异，policy 直出"""
    state = torch.load(state_file, weights_only=True, map_location='cpu')
    cfg = state['config']
    brain = V4Brain(version=cfg['control']['version'], **cfg['resnet']).to(device).eval()
    brain.load_state_dict(state['mortal'], strict=False)
    return PolicyEngine(brain, is_oracle=False, version=4, device=device,
                        enable_amp=True, enable_rule_based_agari_guard=True,
                        name=name, amp_dtype=torch.bfloat16)


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
        for seg_start in range(0, per_opp, segment):
            seg_count = min(segment, per_opp - seg_start)
            env = OneVsThree(disable_progress_bar=False, log_dir=sub_dir)
            rankings, _ = env.py_vs_py(
                challenger=model, champion=champ,
                seed_start=(10000 + i * per_opp + seg_start, 0x2000), seed_count=seg_count,
            )
            for k in range(4):
                totals[k] += rankings[k]
        stat = Stat.from_dir(sub_dir, 'mortal_base_v2')
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
    ap = argparse.ArgumentParser(description='mortal_base_v2 1v3 评估')
    ap.add_argument('--state-file', default=None, help='评估用 checkpoint，默认取 control.state_file')
    ap.add_argument('--games', type=int, default=None, help='覆盖 eval.games')
    args = ap.parse_args()
    device = torch.device(config['control']['device'])
    state_file = args.state_file or config['control']['state_file']
    model = load_model(state_file, device)
    result = run_eval(model, device, games=args.games)
    print(f'avg rank: {result["avg_rank"]:.4}')
    print(f'avg pt: {result["avg_pt"]:.4}')
