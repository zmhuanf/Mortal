"""mortal_v7 评估：DT 引擎 vs baseline_v1 固定对手

DTEngine 按 libriichi 透传的 index 维护每局序列窗口，RTG 用本家分数实时更新
"""

import os
import shutil
from collections import deque
from os import path

import numpy as np
import torch
from torch import nn

import config_v7  # 注册 config 模块，必须先于 from config import config
from config import config
from model import DecisionTransformer, ConvNeXtEncoder
from engine import MortalEngine
from libriichi.arena import OneVsThree
from libriichi.stat import Stat
from libriichi.consts import obs_shape, ACTION_SPACE


class V4Brain(nn.Module):
    """baseline 等 v4 对手的策略网络：ConvNeXt 编码 + GELU + policy 头，与 mortal/model.py 一致"""

    def __init__(self, *, conv_channels, num_blocks, layer_scale=1e-6, drop_rate=0.0, **kwargs):
        super().__init__()
        self.encoder = ConvNeXtEncoder(
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

# v4 obs 布局索引，与 libriichi/src/state/obs_repr.rs 编码顺序一一对应
_SCORE_IDX = 7          # 每玩家 2 行：/100000 与 /30000
_RANK_IDX = 15          # 本家座位 one-hot
_HONBA_IDX = 23         # honba/10 rescale
_GRAND_KYOKU_IDX = 27   # (bakaze-E).min(1)*4+kyoku 的 /7 rescale


def extract_own_score(obs):
    """从 obs 提取本家当前分数（/100000 归一化列还原）"""
    rank = int(np.argmax(obs[_RANK_IDX:_RANK_IDX + 4, 0]))
    return float(obs[_SCORE_IDX + 2 * rank, 0]) * 100000


def extract_kyoku_key(obs):
    """局标识 = (局番, 本场)，任一变化即新局"""
    grand_kyoku = round(float(obs[_GRAND_KYOKU_IDX, 0]) * 7)
    honba = round(float(obs[_HONBA_IDX, 0]) * 10)
    return grand_kyoku, honba


class DTEngine:
    def __init__(self, model, device, *, name='mortal_v7', version=4,
                 score_scale=10000.0, target_score=35000.0, window=96):
        self.engine_type = 'mortal'
        self.name = name
        self.is_oracle = False
        self.version = version
        # 全部决策走模型，保证窗口序列完整
        self.enable_quick_eval = False
        self.enable_rule_based_agari_guard = True
        self.device = device
        self.model = model.eval().to(device)
        self.score_scale = score_scale
        self.target_score = target_score
        self.window = window
        self.games = {}  # index -> {obs/rtg/acts 双端队列, key: 局标识, rtg: 当前值}

    def _new_rtg(self, obs):
        own_score = extract_own_score(obs)
        return (self.target_score - own_score) / self.score_scale

    def _forward_window(self, w):
        obs = np.stack(w['obs'])  # (W, 1012, 34)
        rtg = np.asarray(w['rtg'], dtype=np.float32)  # (W,)
        acts = np.asarray(w['acts'], dtype=np.int64)  # (W,)，末位为待预测占位
        obs_t = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        rtg_t = torch.from_numpy(rtg).to(self.device).unsqueeze(0)
        acts_t = torch.from_numpy(acts).to(self.device).unsqueeze(0)
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=True):
            logits = self.model(obs_t, rtg_t, acts_t)
        return logits[0, -1].cpu().numpy()  # 最后动作位置 (A,)

    def react_batch(self, obs, masks, invisible_obs, indexes=None):
        actions, values, masks_out, is_greedy = [], [], [], []
        for i, (o, m) in enumerate(zip(obs, masks)):
            idx = indexes[i] if indexes is not None else i
            w = self.games.setdefault(idx, {
                'obs': deque(maxlen=self.window),
                'rtg': deque(maxlen=self.window),
                'acts': deque(maxlen=self.window),
                'key': None,
            })
            key = extract_kyoku_key(o)
            if w['key'] != key:
                # 新局：清空旧局窗口（位置从 0 起，与训练整局序列一致），重算 RTG
                w['obs'].clear()
                w['rtg'].clear()
                w['acts'].clear()
                w['key'] = key
                w['rtg_val'] = self._new_rtg(o)
            w['obs'].append(o)
            w['rtg'].append(w['rtg_val'])
            # 当前动作占位，模型 shift 后不看它；回填供下一步推理作为上一动作输入
            w['acts'].append(0)

            logits = self._forward_window(w)
            logits = np.where(np.asarray(m, dtype=bool), logits, -np.inf)
            a = int(np.argmax(logits))
            w['acts'][-1] = a

            actions.append(a)
            values.append(logits.tolist())
            masks_out.append(m.tolist())
            is_greedy.append(True)
        return actions, values, masks_out, is_greedy


def load_model(state_file, device):
    state = torch.load(state_file, weights_only=False, map_location=device)
    model = DecisionTransformer(**state['config']['model']).to(device)
    model.load_state_dict(state['model'])
    return model


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
            engine_chal = DTEngine(model, device,
                                   window=config['rtg']['window'],
                                   score_scale=config['rtg']['score_scale'],
                                   target_score=config['rtg']['target_score'])
            env = OneVsThree(disable_progress_bar=False, log_dir=sub_dir)
            rankings, _ = env.py_vs_py(
                challenger=engine_chal,
                champion=champ,
                seed_start=(10000 + i * per_opp + seg_start, 0x2000),
                seed_count=seg_count,
            )
            for k in range(4):
                totals[k] += rankings[k]
            del engine_chal
        stat = Stat.from_dir(sub_dir, 'mortal_v7')
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

    parser = argparse.ArgumentParser(description='mortal_v7 1v3 评估')
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
