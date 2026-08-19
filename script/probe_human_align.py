"""938k mortal_base vs baseline_v1 人类牌谱深度对齐

输出四块：分歧动作对×结局、分歧中命中真实的一侧×结局、放炮防守、赢局/输局内命中
裁判均为局内真实结局（终局得分/排名），相关性统计而非因果模拟
"""

import argparse
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
    ap = argparse.ArgumentParser(description='mortal_base vs baseline_v1 人类牌谱深度对齐')
    ap.add_argument('--files', type=int, default=60)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    idx = torch.load(INDEX, weights_only=True)
    files = idx['file_list'][:args.files]

    obs_all, mask_all, act_all, shan_all = [], [], [], []
    rii_all, aga_all, hou_all = [], [], []
    fin_all, rank_all = [], []
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
            player_id = game.take_player_id()
            g = game.take_grp()  # take_grp 消费式，只能取一次
            fs = np.array(g.take_final_scores())
            rank_of = (-fs).argsort(kind='stable').argsort(kind='stable')
            nt = obs_all[-1].shape[0]
            fin_all.append(np.full(nt, fs[player_id]))
            rank_all.append(np.full(nt, rank_of[player_id]))

    obs = np.concatenate(obs_all)
    masks = np.concatenate(mask_all)
    actions = np.concatenate(act_all)
    shantens = np.concatenate(shan_all)
    is_riichi = np.concatenate(rii_all)
    is_agari = np.concatenate(aga_all)
    is_houjuu = np.concatenate(hou_all)
    fscore = np.concatenate(fin_all)
    frank = np.concatenate(rank_all)
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
          f'base==v1 {(~diverge).mean()*100:.1f}% 分歧 {diverge.mean()*100:.1f}%')

    # A 分桶命中（向听 × 输赢局）
    def bucket_report(mask, name):
        n = mask.sum()
        if n == 0:
            return
        d, mb, mv = diverge[mask], match_b[mask], match_v[mask]
        print(f'{name}: n={n}  分歧 {d.mean()*100:.1f}%  base命中 {mb.mean()*100:.1f}%  v1命中 {mv.mean()*100:.1f}%')

    print('\n[A] 场景分桶命中率')
    bucket_report(np.ones(len(obs), bool), '全体')
    bucket_report((shantens == 0) & ~is_riichi, '听牌(未立直)')
    bucket_report((shantens >= 1) & (shantens <= 2), '1~2向听')
    bucket_report(shantens >= 3, '3+向听')
    bucket_report(frank <= 1, '玩家最终1~2位(赢局)')
    bucket_report(frank >= 2, '玩家最终3~4位(输局)')
    bucket_report(is_agari, '真实和牌turn')
    bucket_report(is_houjuu, '真实放炮turn')
    bucket_report(is_riichi, '立直turn')

    # B 分歧 turn 按动作对分组，看真实结局
    d_sel = diverge
    print('\n[B] 分歧turn按动作对(base, v1)分组 → 该玩家终局')
    uniq, cnt = np.unique(np.stack((a_b[d_sel], a_v[d_sel]), -1), axis=0, return_counts=True)
    for i in np.argsort(-cnt)[:15]:
        ab, av = uniq[i]
        sel = (a_b == ab) & (a_v == av) & d_sel
        n = sel.sum()
        if n < 5:
            continue
        print(f'  base={ab:>2} v1={av:>2}  n={n:>4}  终局得分均值 {fscore[sel].mean():+8.0f}  1~2位率 {(frank[sel]<=1).mean()*100:.0f}%')

    # C 分歧中谁命中真实 → 结局
    print('\n[C] 分歧turn按命中侧分组 → 结局（命中人类侧是否真赢）')
    both_wrong = d_sel & ~match_b & ~match_v
    only_b = d_sel & match_b & ~match_v
    only_v = d_sel & match_v & ~match_b
    for sel, name in [(only_b, '仅base命中'), (only_v, '仅v1命中'), (both_wrong, '双错')]:
        n = sel.sum()
        if n == 0:
            continue
        print(f'  {name}: n={n:>4}  终局得分均值 {fscore[sel].mean():+8.0f}  1~2位率 {(frank[sel]<=1).mean()*100:.0f}%')

    # D 放炮防守
    print('\n[D] 放炮防守')
    for name, sel in [('真实放炮turn', is_houjuu), ('非放炮turn', ~is_houjuu)]:
        n = sel.sum()
        if n == 0:
            continue
        print(f'  {name}: n={n:>5}  base选中真实动作 {match_b[sel].mean()*100:.1f}%  v1 {match_v[sel].mean()*100:.1f}%'
              f'  该组玩家终局得分均值 {fscore[sel].mean():+7.0f}')

    # E 分歧 turn 整体强度
    print('\n[E] 分歧turn玩家的终局分布（分歧是否集中在弱/强局势）')
    n = d_sel.sum()
    print(f'  分歧turn共 {n}：玩家终局得分均值 {fscore[d_sel].mean():+.0f}  1~2位率 {(frank[d_sel]<=1).mean()*100:.0f}%')
    print(f'  非分歧turn共 {(~d_sel).sum()}：玩家终局得分均值 {fscore[~d_sel].mean():+.0f}  1~2位率 {(frank[~d_sel]<=1).mean()*100:.0f}%')


if __name__ == '__main__':
    main()