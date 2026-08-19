"""全网络健康检查：Q 校准 + policy 置信度 + aux 头准确率，对照 v1

纯只读：加载 checkpoint、前向、比对标签，不写文件；分批小量防显存
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
from model import Brain, DQN, AuxNet  # noqa: E402
from evaluate import V4Brain, V4DQN  # noqa: E402

INDEX = BASE_DIR / 'out' / 'file_index.pth'
BASE_CKPT = BASE_DIR / 'out' / 'mortal.pth'
V1_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', type=int, default=30)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    idx = torch.load(INDEX, weights_only=True)
    files = idx['file_list'][:args.files]

    obs_all, mask_all, act_all = [], [], []
    shan_all, fuu_all, rii_all = [], [], []
    fin_rank_all = []
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    for file in loader.load_gz_log_files(files):
        for game in file:
            obs_g = np.asarray(game.take_obs(), dtype=np.float32)  # take_obs 消费式
            nt = len(obs_g)
            if nt == 0:
                continue
            obs_all.append(obs_g)
            mask_all.append(np.asarray(game.take_masks()))
            act_all.append(np.asarray(game.take_actions(), dtype=np.int64))
            shan_all.append(np.asarray(game.take_shantens(), dtype=np.int64).clip(0, 6))
            fuu_all.append(np.frombuffer(game.take_fuuro_counts(), dtype=np.uint8).astype(np.int64).clip(0, 6))
            rii_all.append(np.frombuffer(game.take_riichi_turns(), dtype=np.uint8).astype(np.int64))
            player_id = game.take_player_id()
            g = game.take_grp()  # take_grp 消费式
            rank = np.frombuffer(g.take_rank_by_player(), dtype=np.uint8)
            fin_rank_all.append(np.full(nt, rank[player_id]))

    obs = np.concatenate(obs_all)
    masks = np.concatenate(mask_all)
    actions = np.concatenate(act_all)
    shantens = np.concatenate(shan_all)
    fuuro = np.concatenate(fuu_all)
    rii_t = np.concatenate(rii_all)
    fin_rank = np.concatenate(fin_rank_all)
    print(f'transitions: {len(obs):,}')

    sb = torch.load(BASE_CKPT, weights_only=False, map_location='cpu')
    sv = torch.load(V1_CKPT, weights_only=True, map_location='cpu')
    brain_b = Brain(version=4, **sb['config']['model']).to(device).eval()
    brain_b.load_state_dict(sb['mortal'])
    dqn_b = DQN(version=4, **sb['config']['dqn']).to(device).eval()
    dqn_b.load_state_dict(sb['current_dqn'])
    aux = AuxNet().to(device).eval()
    aux.load_state_dict(sb['aux_net'])
    brain_v = V4Brain(version=4, **sv['config']['resnet']).to(device).eval()
    brain_v.load_state_dict(sv['mortal'])
    dqn_v = V4DQN(num_heads=sv['config']['dqn']['num_heads']).to(device).eval()
    dqn_v.load_state_dict(sv['current_dqn'])

    n = len(obs)
    q_b = np.empty(n); q_v = np.empty(n)
    conf_b = np.empty(n); conf_v = np.empty(n)
    acc_b = np.empty(n); acc_v = np.empty(n)
    acc_b = np.empty(n); acc_v = np.empty(n)
    aux_out = {k: np.empty(n) for k in ('shanten', 'fuuro', 'rii')}
    a_spread_b = np.empty(n); a_spread_v = np.empty(n)

    for s in range(0, n, 1024):
        e = min(s + 1024, n)
        o = torch.from_numpy(obs[s:e]).to(device)
        m = torch.from_numpy(masks[s:e]).to(device)
        a = torch.from_numpy(actions[s:e]).to(device)
        with torch.no_grad():
            phib = brain_b(o); phiv = brain_v(o)
            lb = brain_b.policy_logits(phib).masked_fill(~m, -1e9)
            lv = brain_v.policy_logits(phiv).masked_fill(~m, -1e9)
            pb = lb.log_softmax(-1); pv = lv.log_softmax(-1)
            conf_b[s:e] = pb.gather(1, a.unsqueeze(-1)).squeeze(-1).exp().cpu().numpy()
            conf_v[s:e] = pv.gather(1, a.unsqueeze(-1)).squeeze(-1).exp().cpu().numpy()
            acc_b[s:e] = (lb.argmax(-1) == a).cpu().numpy()
            acc_v[s:e] = (lv.argmax(-1) == a).cpu().numpy()
            qb = dqn_b(phib, m)
            qv = dqn_v(phiv, m)
            qbn = qb.cpu().numpy()
            qvn = qv.cpu().numpy()
            mkn = m.cpu().numpy()[:, None, :]  # (N,1,A) 非法动作置 NaN
            qb_ok = np.where(mkn, qbn, np.nan)
            qv_ok = np.where(mkn, qvn, np.nan)
            # 局面价值 = 合法动作 Q 的均值（K-heads 再平均）
            q_b[s:e] = np.nanmean(qb_ok, -1).mean(1)
            q_v[s:e] = np.nanmean(qv_ok, -1).mean(1)
            # 动作价值幅度：合法动作上 Q 的 max−mean，K-heads 平均
            a_spread_b[s:e] = (np.nanmax(qb_ok, -1) - np.nanmean(qb_ok, -1)).mean(1)
            a_spread_v[s:e] = (np.nanmax(qv_ok, -1) - np.nanmean(qv_ok, -1)).mean(1)
            shan_l, fuu_l, rii_l, _ = aux(phib)
            aux_out['shanten'][s:e] = shan_l.argmax(-1).cpu().numpy()
            aux_out['fuuro'][s:e] = fuu_l.argmax(-1).cpu().numpy()
            aux_out['rii'][s:e] = rii_l.argmax(-1).cpu().numpy()

    print(f'\n[A] Q 校准（按玩家终局名次，K-heads 平均）')
    print(f'{"名次":<8}{"真实pts":>8}{"baseQ":>9}{"v1Q":>9}')
    for r in range(4):
        sel = fin_rank == r
        print(f'rank{r}     {[10,4,-1,-5][r]:>8}{q_b[sel].mean():>9.2f}{q_v[sel].mean():>9.2f}')

    print(f'\n[B] policy 置信度校准（真实动作上的概率 vs 命中率）')
    print(f'{"置信桶":<12}{"n":>7}{"base命中":>10}{"v1命中":>10}')
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        sel = (conf_b >= lo) & (conf_b < hi)
        n_sel = sel.sum()
        if n_sel < 50:
            continue
        print(f'{lo:.1f}~{hi:.1f}    {n_sel:>7}{acc_b[sel].mean()*100:>9.1f}%{acc_v[sel].mean()*100:>9.1f}%')

    print(f'\n[C] aux 头准确率（base 独有，对照真实标签）')
    for key, lab, name in [('shanten', shantens, '向听数(0~6)'), ('fuuro', fuuro, '副露数(0~6)'), ('rii', rii_t, '立直时机')]:
        hit = aux_out[key] == lab
        print(f'{name:<12} acc {hit.mean()*100:.1f}%  标签分布 {np.bincount(lab, minlength=7)[:7].tolist()}')

    print(f'\n[D] dueling 优势幅度（A 跨动作 std，K-heads 平均）')
    print(f'base: {a_spread_b.mean():.3f}   v1: {a_spread_v.mean():.3f}')
    print(f'base 顶1命中 {acc_b.mean()*100:.1f}%   v1 {acc_v.mean()*100:.1f}%')


if __name__ == '__main__':
    main()