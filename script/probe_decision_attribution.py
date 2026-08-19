"""913k mortal_base vs baseline_v1 决策归因：同牌谱双模型逐决策分歧 + 场景分类

裁判信号：真实 turn 标记（和牌/放炮/立直 turn）+ 真实玩家动作，注意这是相关性统计而非因果模拟
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / 'mortal_base'
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from libriichi.dataset import GameplayLoader  # noqa: E402
from model import Brain, DQN  # noqa: E402
from evaluate import V4Brain, V4DQN  # noqa: E402

RIICHI, AGARI, PASS = 37, 43, 45
INDEX = BASE_DIR / 'out' / 'file_index.pth'
BASE_CKPT = BASE_DIR / 'out' / 'mortal.pth'
V1_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'


def load_pair(device):
    sb = torch.load(BASE_CKPT, weights_only=False, map_location='cpu')
    sv = torch.load(V1_CKPT, weights_only=True, map_location='cpu')
    brain_b = Brain(version=4, **sb['config']['model']).to(device).eval()
    brain_b.load_state_dict(sb['mortal'])
    brain_v = V4Brain(version=4, **sv['config']['resnet']).to(device).eval()
    brain_v.load_state_dict(sv['mortal'])
    return brain_b, brain_v


def main():
    import argparse

    ap = argparse.ArgumentParser(description='mortal_base(913k) vs baseline_v1 决策归因')
    ap.add_argument('--files', type=int, default=60)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    idx = torch.load(INDEX, weights_only=True)
    files = idx['file_list'][:args.files]

    obs_all, mask_all, act_all, shan_all = [], [], [], []
    rii_all, aga_all, hou_all = [], [], []
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(files):
        for game in file:
            obs_all.append(np.asarray(game.take_obs(), dtype=np.float32))
            mask_all.append(np.asarray(game.take_masks()))
            act_all.append(np.asarray(game.take_actions(), dtype=np.int64))
            shan_all.append(np.asarray(game.take_shantens(), dtype=np.int64))
            rii_all.append(np.array(game.take_is_riichi_turn(), dtype=bool))
            aga_all.append(np.array(game.take_is_agari_turn(), dtype=bool))
            hou_all.append(np.array(game.take_is_houjuu_turn(), dtype=bool))

    obs = np.concatenate(obs_all)
    masks = np.concatenate(mask_all)
    actions = np.concatenate(act_all)
    shantens = np.concatenate(shan_all)
    is_riichi = np.concatenate(rii_all)
    is_agari = np.concatenate(aga_all)
    is_houjuu = np.concatenate(hou_all)
    print(f'transitions: {len(obs):,}')

    brain_b, brain_v = load_pair(device)
    a_b, a_v = np.zeros(len(obs), np.int64), np.zeros(len(obs), np.int64)
    chunk = 2048
    for s in range(0, len(obs), chunk):
        e = min(s + chunk, len(obs))
        o = torch.from_numpy(obs[s:e]).to(device)
        m = torch.from_numpy(masks[s:e]).to(device)
        with torch.no_grad():
            lb = brain_b.policy_logits(brain_b(o)).masked_fill(~m, -1e9)
            lv = brain_v.policy_logits(brain_v(o)).masked_fill(~m, -1e9)
        a_b[s:e] = lb.argmax(-1).cpu().numpy()
        a_v[s:e] = lv.argmax(-1).cpu().numpy()

    diverge = a_b != a_v
    match_b = a_b == actions
    match_v = a_v == actions
    print(f'\n顶1动作: base命中真实 {match_b.mean()*100:.1f}%  v1命中真实 {match_v.mean()*100:.1f}%  '
          f'base==v1 {100 - diverge.mean()*100:.1f}% 分歧 {(diverge).mean()*100:.1f}%')

    def group_report(name, sel, detail=False):
        n = sel.sum()
        if n == 0:
            print(f'\n{name}: 无样本')
            return
        d = diverge[sel]
        mb = match_b[sel]
        mv = match_v[sel]
        print(f'\n{name}: n={n}')
        print(f'  分歧率 {d.mean()*100:.1f}%   base命中 {mb.mean()*100:.1f}%   v1命中 {mv.mean()*100:.1f}%')
        if detail:
            rb = (a_b[sel] == RIICHI)
            rv = (a_v[sel] == RIICHI)
            ab = (a_b[sel] == AGARI)
            av = (a_v[sel] == AGARI)
            pb = (a_b[sel] == PASS)
            pv = (a_v[sel] == PASS)
            print(f'  选立直 base {rb.mean()*100:.1f}% vs v1 {rv.mean()*100:.1f}%')
            print(f'  选择和牌 base {ab.mean()*100:.1f}% vs v1 {av.mean()*100:.1f}%')
            print(f'  选Pass   base {pb.mean()*100:.1f}% vs v1 {pv.mean()*100:.1f}%')

    group_report('全体', np.ones(len(obs), bool))
    group_report('和牌turn(真实玩家和牌)', is_agari, detail=True)
    group_report('放炮turn(真实玩家放炮)', is_houjuu)
    group_report('立直turn', is_riichi, detail=True)
    for lo, hi, label in [(0, 0, '听牌 shanten=0'), (1, 2, '1~2向听'), (3, 8, '3+向听')]:
        sel = (shantens >= lo) & (shantens <= hi)
        group_report(label, sel)

    # 分歧中 base 的立直倾向
    d = diverge
    print(f'\n分歧样本中: base多立直 {((a_b==RIICHI)&(a_v!=RIICHI))[d].sum()}  '
          f'v1多立直 {((a_v==RIICHI)&(a_b!=RIICHI))[d].sum()}  '
          f'base多Pass {((a_b==PASS)&(a_v!=PASS))[d].sum()}  '
          f'v1多Pass {((a_v==PASS)&(a_b!=PASS))[d].sum()}')


if __name__ == '__main__':
    main()