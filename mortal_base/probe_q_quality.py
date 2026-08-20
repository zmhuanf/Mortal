"""Q/A 学习质量探针：事件回报回归后的排序相关与命中率

对比基线（改前）：Q top-1 命中真实动作 19.8%、Q vs MC rho +0.157、A 幅度 0.09
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
PER_GAME = 200


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def main():
    device = torch.device('cuda:0')
    s = torch.load(config['control']['state_file'], weights_only=False, map_location=device)
    cfg = s['config']
    brain = Brain(version=cfg['control']['version'], **cfg['model']).to(device).eval()
    brain.load_state_dict(s['mortal'])
    dqn = DQN(version=cfg['control']['version'], **cfg['dqn']).to(device).eval()
    dqn.load_state_dict(s['current_dqn'])
    ev_net = EventModel(phi_dim=cfg['model']['phi_dim'],
                        **{k: v for k, v in cfg.get('event', {}).items() if k != 'weight'}).to(device).eval()
    ev_net.load_state_dict(s['event_model'])

    grp = GRP(**cfg['grp']['network'])
    grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')['model'])
    reward_calc = RewardCalculator(grp, config['env']['pts'])
    gamma_pow = np.float32(config['env']['gamma']) ** np.arange(513, dtype=np.float32)

    idx = torch.load(config['dataset']['file_index'], weights_only=True)
    rng = np.random.default_rng(11)
    files = rng.choice(idx['file_list'], N_FILES, replace=False)

    obs_all, m_all, a_all, t_all = [], [], [], []
    loader = GameplayLoader(version=config['control']['version'], oracle=False)
    for file in loader.load_gz_log_files(files):
        for game in file:
            obs = np.asarray(game.take_obs(), dtype=np.float32)
            T = len(obs)
            if T == 0:
                continue
            at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
            dones = np.asarray(game.take_dones(), dtype=bool)
            apply_gamma = np.asarray(game.take_apply_gamma(), dtype=np.int64)
            grp_obj = game.take_grp()
            grp_feature = grp_obj.take_feature()
            if grp_feature.shape[0] > 12:
                continue
            rank_by_player = grp_obj.take_rank_by_player()
            kyoku_rewards = reward_calc.calc_delta_pt(game.take_player_id(), grp_feature, rank_by_player)
            gamma_prefix = np.concatenate(([0], np.cumsum(apply_gamma)))
            kyoku_end_turns = np.flatnonzero(dones)
            pick = np.unique(np.linspace(0, T - 1, PER_GAME).round().astype(int))
            tgt = np.zeros(T, dtype=np.float32)
            for i in range(T):
                end_turn = kyoku_end_turns[at_kyoku[i]] if at_kyoku[i] < len(kyoku_end_turns) else T - 1
                d = int(gamma_prefix[end_turn + 1] - gamma_prefix[i])
                tgt[i] = gamma_pow[d] * kyoku_rewards[at_kyoku[i]]
            obs_all.append(obs[pick]); m_all.append(np.asarray(game.take_masks())[pick])
            a_all.append(np.asarray(game.take_actions())[pick]); t_all.append(tgt[pick])
    obs = torch.from_numpy(np.concatenate(obs_all)).to(device)
    masks = np.concatenate(m_all).astype(bool)
    acts = np.concatenate(a_all); tgt = np.concatenate(t_all)
    B = len(obs)

    q_np, a_np, v_np, ev_np = [], [], [], []
    with torch.autocast('cuda', dtype=torch.bfloat16), torch.inference_mode():
        for i in range(0, B, 256):
            mm = torch.from_numpy(masks[i:i + 256]).to(device)
            aa = torch.from_numpy(acts[i:i + 256]).to(device)
            phi = brain(obs[i:i + 256])
            q = dqn(phi, mm)
            q_np.append(q.mean(1).float().cpu().numpy())
            a_np.append(dqn.advantage(phi, mm).mean(1).float().cpu().numpy())
            v_np.append(dqn.value(phi).mean(-1).float().cpu().numpy())
            ev_np.append(ev_net.rollout_returns(ev_net(phi, aa), gamma=config['env']['gamma'],
                                                rewards=config['event']['rewards']).float().cpu().numpy())
    q = np.concatenate(q_np); a = np.concatenate(a_np)
    v = np.concatenate(v_np); ev_t = np.concatenate(ev_np)

    q_filled = np.where(masks, q, -np.inf)
    a_filled = np.where(masks, a, -np.inf)
    a_q = q_filled.argmax(-1); a_a = a_filled.argmax(-1)
    print(f'samples: {B}')
    print(f'A 幅度(合法位): {np.where(masks, a - a.mean(-1, keepdims=True), 0).std(-1).mean():.4f}')
    print(f'Q top-1 命中真实动作: {(a_q == acts).mean()*100:.2f}%  (基线 19.8%)')
    print(f'A top-1 命中真实动作: {(a_a == acts).mean()*100:.2f}%')
    print(f'Q(s,a_true) vs MC 回报 rho: {spearman(q[np.arange(B), acts], tgt):+.4f}  (基线 +0.157)')
    print(f'V vs MC 回报 rho: {spearman(v, tgt):+.4f}  (基线 +0.157)')
    print(f'A(s,a_true) vs 事件期望回报 rho: {spearman(a[np.arange(B), acts], ev_t):+.4f}')
    print(f'A(s,a_true) vs MC 回报 rho: {spearman(a[np.arange(B), acts], tgt):+.4f}')


if __name__ == '__main__':
    main()