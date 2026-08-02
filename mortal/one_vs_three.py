import prelude

import numpy as np
import torch
import secrets
import os
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from config import config

def main():
    cfg = config['1v3']
    games_per_iter = cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = cfg['log_dir']
    use_akochan = cfg['akochan']['enabled']

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    if use_akochan:
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        state = torch.load(cfg['champion']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        cham_cfg = state['config']
        version = cham_cfg['control'].get('version', 4)
        mortal = Brain(version=version, **cham_cfg['resnet']).eval()
        dqn = DQN(version=version, num_heads=cham_cfg.get('dqn', {}).get('num_heads', 1)).eval()
        # 旧 checkpoint 无 policy_head，strict=False 随机初始化并退回 q 选动作
        mortal.load_state_dict(state['mortal'], strict=False)
        dqn.load_state_dict(state['current_dqn'])
        if cfg['champion']['enable_compile']:
            mortal.compile()
            dqn.compile()
        engine_cham = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = version,
            device = torch.device(cfg['champion']['device']),
            enable_amp = cfg['champion']['enable_amp'],
            enable_rule_based_agari_guard = cfg['champion']['enable_rule_based_agari_guard'],
            name = cfg['champion']['name'],
            action_source = 'policy' if 'policy_head.weight' in state['mortal'] else 'q',
        )

    state = torch.load(cfg['challenger']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    chal_cfg = state['config']
    version = chal_cfg['control'].get('version', 4)
    mortal = Brain(version=version, **chal_cfg['resnet']).eval()
    dqn = DQN(version=version, num_heads=chal_cfg.get('dqn', {}).get('num_heads', 1)).eval()
    # 旧 checkpoint 无 policy_head，strict=False 随机初始化并退回 q 选动作
    mortal.load_state_dict(state['mortal'], strict=False)
    dqn.load_state_dict(state['current_dqn'])
    if cfg['challenger']['enable_compile']:
        mortal.compile()
        dqn.compile()
    engine_chal = MortalEngine(
        mortal,
        dqn,
        is_oracle = False,
        version = version,
        device = torch.device(cfg['challenger']['device']),
        enable_amp = cfg['challenger']['enable_amp'],
        enable_rule_based_agari_guard = cfg['challenger']['enable_rule_based_agari_guard'],
        name = cfg['challenger']['name'],
        action_source = 'policy' if 'policy_head.weight' in state['mortal'] else 'q',
    )

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = log_dir,
        )
        if use_akochan:
            rankings = env.ako_vs_py(
                engine = engine_chal,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        else:
            rankings = env.py_vs_py(
                challenger = engine_chal,
                champion = engine_cham,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        rankings = np.array(rankings)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
