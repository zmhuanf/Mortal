"""诊断 q_target 的动作信号强度：结算覆盖 / 信号-噪声分解 / MC 差分信噪比

回答"调大 n_step 不设手动奖励是否可行"前的量化依据
指标
  - 结算窗口外 turn 占比：Q 目标与动作无关的部分（当前全 0 shaping + n_step=3 应接近 100%）
  - 按距局末折扣步分桶的 q_target 构成：真实结算占比 vs 自举 V 占比
  - a_target（纯 MC 到局末）相邻差分分布：动作边际信号的信噪比
  - q_target 方差分解：来自 V 的部分 vs 动作相关的部分
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config_base  # noqa: E402
from config import config  # noqa: E402
from model import Brain, DQN, GRP  # noqa: E402
from dataset import RewardCalculator  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402

N_FILES = 60
PER_GAME = 200
BATCH = 256
N_STEP = config['env']['n_step']
GAMMA = float(config['env']['gamma'])
GAMMA_N = GAMMA ** N_STEP


def collect(file_list):
    """抽样每局 200 turn 的 obs/next_obs/mask 与 numpy 目标量"""
    grp = GRP(**config['grp']['network'])
    grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')['model'])
    reward_calc = RewardCalculator(grp, config['env']['pts'])
    gamma_pow = np.float32(GAMMA) ** np.arange(512 + 1, dtype=np.float32)

    loader = GameplayLoader(version=config['control']['version'], oracle=False)
    r = {k: float(config['reward'].get(k, 0.0)) for k in ('riichi', 'agari', 'houjuu', 'tsumogiri')}
    obs_all, nxt_all, msk_all, tgt_all, atgt_all, disc_all, nrel_all, end_all = [], [], [], [], [], [], [], []
    for file in loader.load_gz_log_files(file_list):
        for game in file:
            obs = np.asarray(game.take_obs(), dtype=np.float32)
            T = len(obs)
            if T == 0:
                continue
            at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
            dones = np.asarray(game.take_dones(), dtype=bool)
            apply_gamma = np.asarray(game.take_apply_gamma(), dtype=np.int64)
            masks = np.asarray(game.take_masks())
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
            kyoku_rewards = reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
            final_pts = np.float32(config['env']['pts'][rank_by_player[player_id]])

            gamma_prefix = np.concatenate(([0], np.cumsum(apply_gamma)))
            kyoku_end_turns = np.flatnonzero(dones)
            # 与 dataset.py 一致的每手即时奖励（含被自摸）
            other_done = dones & ~(is_riichi | is_agari | is_houjuu)
            score_diff = np.concatenate((grp_feature[:, 3:7] * 1e4, [final_scores]))
            kyoku_diff = score_diff[1:, player_id] - score_diff[:-1, player_id]
            is_tsumogiri = other_done & (kyoku_diff[at_kyoku] < 0)
            step_rew = (is_riichi * r['riichi'] + is_agari * r['agari']
                        + is_houjuu * r['houjuu'] + is_tsumogiri * r['tsumogiri']).astype(np.float32)

            idx = np.unique(np.linspace(0, T - 1, PER_GAME).round().astype(int))
            for i in idx:
                # 窗口内事件奖励 + 结算 delta_pt，按折扣步折现
                nrel = 0.0
                j = i
                while j < T and int(gamma_prefix[j] - gamma_prefix[i]) < N_STEP:
                    gd = int(gamma_prefix[j] - gamma_prefix[i])
                    nrel += float(gamma_pow[gd] * step_rew[j])
                    if dones[j]:
                        d = int(gamma_prefix[j + 1] - gamma_prefix[i])
                        if d < N_STEP:
                            nrel += float(gamma_pow[d] * kyoku_rewards[at_kyoku[j]])
                    j += 1
                # 到局末折扣步数
                end_turn = kyoku_end_turns[at_kyoku[i]] if at_kyoku[i] < len(kyoku_end_turns) else T - 1
                disc = int(gamma_prefix[end_turn + 1] - gamma_prefix[i])
                # 纯 MC 到局末（=a_target，仅结算无 shaping）
                atgt = np.float32(gamma_pow[disc] * kyoku_rewards[at_kyoku[i]])
                # next_idx 与训练一致
                is_end = disc < N_STEP or np.searchsorted(gamma_prefix, gamma_prefix[i] + N_STEP, side='left') >= T
                if is_end:
                    ni = i
                    q_raw = final_pts
                else:
                    ni = min(int(np.searchsorted(gamma_prefix, gamma_prefix[i] + N_STEP, side='left')), T - 1)
                    q_raw = nrel
                obs_all.append(obs[i])
                nxt_all.append(obs[ni])
                msk_all.append(masks[i])
                tgt_all.append(q_raw)
                atgt_all.append(atgt)
                disc_all.append(disc)
                nrel_all.append(nrel)
                end_all.append(is_end)

    return tuple(np.stack(x) for x in (obs_all, nxt_all, msk_all, tgt_all, atgt_all, disc_all, nrel_all, end_all))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    s = torch.load(config['control']['state_file'], weights_only=False, map_location=device)
    cfg = s['config']
    mortal = Brain(version=cfg['control']['version'], **cfg['model']).to(device).eval()
    mortal.load_state_dict(s['mortal'])
    tgt_dqn = DQN(version=cfg['control']['version'], **cfg['dqn']).to(device).eval()
    tgt_dqn.load_state_dict(s['target_dqn'])

    idx = torch.load(config['dataset']['file_index'], weights_only=True)
    rng = np.random.default_rng(42)
    files = rng.choice(idx['file_list'], N_FILES, replace=False)
    obs, nxt, _msk, tgt_raw, atgt, disc, nrel, is_end = (torch.as_tensor(x, device=device) for x in collect(files))
    B = len(obs)
    print(f'samples: {B}')

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'), torch.inference_mode():
        v_all, vn_all = [], []
        for i in range(0, B, BATCH):
            v_all.append(tgt_dqn.value(mortal(obs[i:i + BATCH])).mean(-1))
            vn_all.append(tgt_dqn.value(mortal(nxt[i:i + BATCH])).mean(-1))
    v = torch.cat(v_all)
    vn = torch.cat(vn_all)

    boot = GAMMA_N * vn * (~is_end).float()          # 自举 V 部分
    q_target = tgt_raw.float() + boot                 # 与训练一致的完整 target
    atgt = atgt.float()

    print('\n== 窗口覆盖（Q 目标与动作相关性的上限）==')
    win = (nrel != 0).float()
    print(f'n_step={N_STEP} 时窗口含事件/结算的 turn 占比: {win.mean().item()*100:.2f}%  (其余 q_target 与动作无关)')
    hist, edges = np.histogram(disc.cpu().numpy(), bins=[0, 1, 3, N_STEP, 12, 40, 10**9])
    print('  距局末折扣步分布:', dict(zip(edges[:-1], hist)))

    print('\n== q_target 构成（全部 turn / 非终局 turn）==')
    for name, m in (('全部', torch.ones(B, dtype=torch.bool)), ('非终局', ~is_end)):
        seg = q_target[m]
        bseg = boot[m]
        rseg = tgt_raw[m].float()
        print(f'  {name}: q_target std {seg.std().item():.4f} | 自举V部分 std {bseg.std().item():.4f} | 结算部分 std {rseg.std().item():.4f}')

    print('\n== 按距局末分桶的 q_target 信号强度 ==')
    print('  disc      n    qstd      V部分std   事件/结算部分std')
    for lo, hi in ((0, 1), (1, 3), (3, N_STEP), (N_STEP, 12), (12, 40), (40, 10**9)):
        m = (disc >= lo) & (disc < hi)
        if m.sum() < 20:
            continue
        seg = q_target[m]
        vseg = boot[m]
        rseg = tgt_raw[m].float()
        print(f'  [{lo:>2},{hi:>4}) {m.sum():>5}  {seg.std().item():7.4f}  {vseg.std().item():7.4f}  {rseg.std().item():7.4f}')

    print('\n== a_target（纯 MC 到局末）相邻差分 = 动作边际信号强度 ==')
    # 同局内相邻采样 turn 的 a_target 差，近似该步动作对回报的边际贡献
    d = atgt[1:] - atgt[:-1]
    same_game = disc[1:] < disc[:-1]  # 同局内 disc 递减（越走越离局末近）
    d = d[same_game]
    print(f'  相邻差分: mean {d.mean().item():+.4f} std {d.std().item():.4f} |50% {d.abs().median().item():.4f} |90% {d.abs().quantile(0.9).item():.4f}')
    print(f'  a_target 自身: std {atgt.std().item():.4f} → 若 Q 能学会动作差异，A 分支幅度应接近该量级')

    print('\n== 给决策的结论 ==')
    act_signal = win.mean().item()  # 当前窗口含信号的 turn 占比
    print(f'  当前 Q 目标中动作无关 turn 占 {(1-act_signal)*100:.0f}% → 调大 n_step 至窗口覆盖更多结算时，'
          f'信号占比上升，但方差随 MC 化同步上升；a_target 差分 std {d.std().item():.4f} 给出可学会信号的上限')


if __name__ == '__main__':
    main()