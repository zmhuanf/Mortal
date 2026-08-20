"""两项诊断：V 预测能力上限 + 事件模型每类精度

回答：动作价值能否从离线数据学到（决定 RL 是否值得继续）
- V 诊断：V(s) 随距终局接近是否收敛到 final_pts；跨局相关上限
- 事件诊断：每类事件召回率（无事占多数，需单独看稀有事件的精度）
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config_base  # noqa: E402
from config import config  # noqa: E402
from model import Brain, DQN, GRP, EventModel  # noqa: E402
from dataset import RewardCalculator  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402

N_FILES = 40
PER_GAME = 120
BATCH = 256
NAMES = ('无事', '立直', '和牌', '放铳', '流局', '被自摸')


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def collect():
    """抽样 obs / V 目标元信息 / 事件标签，返回 CPU 数据"""
    grp = GRP(**config['grp']['network'])
    grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')['model'])
    reward_calc = RewardCalculator(grp, config['env']['pts'])
    gamma_pow = np.float32(config['env']['gamma']) ** np.arange(513, dtype=np.float32)

    idx = torch.load(config['dataset']['file_index'], weights_only=True)
    rng = np.random.default_rng(5)
    files = rng.choice(idx['file_list'], N_FILES, replace=False)

    obs_all, msk_all, act_all, end_all, final_all, ev_traj_all = [], [], [], [], [], []
    loader = GameplayLoader(version=config['control']['version'], oracle=False)
    horizon = int(config['event']['horizon'])
    for file in loader.load_gz_log_files(files):
        for game in file:
            obs = np.asarray(game.take_obs(), dtype=np.float32)
            T = len(obs)
            if T == 0:
                continue
            at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
            dones = np.asarray(game.take_dones(), dtype=bool)
            apply_gamma = np.asarray(game.take_apply_gamma(), dtype=np.int64)
            is_riichi = np.array(game.take_is_riichi_turn(), dtype=bool)
            is_agari = np.array(game.take_is_agari_turn(), dtype=bool)
            is_houjuu = np.array(game.take_is_houjuu_turn(), dtype=bool)
            grp_obj = game.take_grp()
            grp_feature = grp_obj.take_feature()
            if grp_feature.shape[0] > 12:
                continue
            rank_by_player = grp_obj.take_rank_by_player()
            final_scores = grp_obj.take_final_scores()
            player_id = game.take_player_id()
            final_pts = np.float32(config['env']['pts'][rank_by_player[player_id]])

            gamma_prefix = np.concatenate(([0], np.cumsum(apply_gamma)))
            kyoku_end_turns = np.flatnonzero(dones)
            # 事件标签（与 dataset.py 一致）
            event_ids = np.zeros(T, dtype=np.int64)
            event_ids[is_riichi] = 1
            event_ids[is_agari] = 2
            event_ids[is_houjuu] = 3
            other_done = dones & (event_ids == 0)
            score_seq = np.concatenate((grp_feature[:, 3:7] * 1e4, [final_scores]))
            diff = score_seq[1:, player_id] - score_seq[:-1, player_id]
            event_ids[other_done] = np.where(diff[at_kyoku[other_done]] < 0, 5, 4)
            kyoku_end = np.searchsorted(at_kyoku, at_kyoku, side='right') - 1
            col = np.arange(horizon)[None, :]
            idx2 = np.arange(T)[:, None] + col
            valid = idx2 <= kyoku_end[:, None]
            ev_traj = np.full((T, horizon), -1, dtype=np.int64)
            ev_traj[valid] = event_ids[idx2[valid]]

            pick = np.unique(np.linspace(0, T - 1, PER_GAME).round().astype(int))
            obs_all.append(obs[pick])
            msk_all.append(np.asarray(game.take_masks())[pick])
            act_all.append(np.asarray(game.take_actions())[pick])
            end_all.append(np.array([int(gamma_prefix[(kyoku_end_turns[at_kyoku[i]] if at_kyoku[i] < len(kyoku_end_turns) else T - 1) + 1] - gamma_prefix[i]) for i in pick]))
            final_all.append(np.full(len(pick), final_pts, dtype=np.float32))
            ev_traj_all.append(ev_traj[pick])
    return tuple(np.concatenate(x) for x in (obs_all, msk_all, act_all, end_all, final_all, ev_traj_all))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    s = torch.load(config['control']['state_file'], weights_only=False, map_location=device)
    cfg = s['config']
    brain = Brain(version=cfg['control']['version'], **cfg['model']).to(device).eval()
    brain.load_state_dict(s['mortal'])
    dqn = DQN(version=cfg['control']['version'], **cfg['dqn']).to(device).eval()
    dqn.load_state_dict(s['current_dqn'])
    ev_net = EventModel(phi_dim=cfg['model']['phi_dim'],
                        **{k: v for k, v in cfg.get('event', {}).items() if k != 'weight'}).to(device).eval()
    ev_net.load_state_dict(s['event_model'])

    obs, masks, acts, disc, final_pts, ev_traj = (torch.as_tensor(x, device=device) for x in collect())
    B = len(obs)
    print(f'samples: {B}')

    v_np, ev_wise = [], []
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'), torch.inference_mode():
        for i in range(0, B, BATCH):
            phi = brain(obs[i:i + BATCH])
            v_np.append(dqn.value(phi).mean(-1).float().cpu().numpy())
            a = acts[i:i + BATCH]
            ev_wise.append(ev_net(phi, a).float().cpu().numpy())
    v = np.concatenate(v_np)
    ev_logits = np.concatenate(ev_wise)  # (B, H, 6)
    final_pts = final_pts.cpu().numpy()
    disc = disc.cpu().numpy()
    ev_traj = ev_traj.cpu().numpy()

    print('\n== V 诊断：V 是否能收敛到最终结果 ==')
    print(f'全局 V vs final_pts spearman: {spearman(v, final_pts):+.4f}')
    print('按距终局折扣步分桶: V vs final_pts（后期应强收敛）')
    for lo, hi in ((0, 3), (3, 8), (8, 16), (16, 30), (30, 10**9)):
        m = (disc >= lo) & (disc < hi)
        if m.sum() > 30:
            print(f'  steps[{lo:>2},{hi:>4}) n={m.sum():>5} rho={spearman(v[m], final_pts[m]):+.4f}')
    # V 的局间可解释性：不同最终结局的 V 均值应单调对齐排名
    print('4 种最终结果的 V 均值（应单调 10>>4>-1>-5）: ' +
          ', '.join(f'pts{p}={v[final_pts == p].mean():+.3f}' for p in (10, 4, -1, -5)))

    print('\n== 事件模型精度 ==')
    pred = ev_logits.argmax(-1)  # (B, H)
    valid = ev_traj >= 0
    conf = np.zeros((6, 6), dtype=np.int64)
    for c in range(6):
        for p in range(6):
            conf[c, p] = ((ev_traj == c) & (pred == p) & valid).sum()
    hdr = ('真实\预测'.ljust(10) + ''.join(f'{NAMES[j]:>7}' for j in range(6)) + '  recall')
    print(hdr)
    for c in range(6):
        row = conf[c]
        rec = row[c] / max(1, row.sum())
        print(f'{NAMES[c]:<10}' + ''.join(f'{x:>7}' for x in row) + f'{rec:>8.3f}')
    # 事件1/2/3/5 的预测命中（信号事件）
    sig = (ev_traj >= 1) & (ev_traj <= 3) | (ev_traj == 5)
    print(f'信号事件占比: {sig.sum()/max(1, valid.sum())*100:.2f}% | 预测命中率: {(pred[sig] == ev_traj[sig]).sum()/max(1, sig.sum())*100:.2f}%')


if __name__ == '__main__':
    main()