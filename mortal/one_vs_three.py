import prelude

from datetime import datetime

import numpy as np
import torch
import secrets
import os
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from config import config


def load_py_engine(cfg):
    state = torch.load(cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
    model_cfg = state['config']
    version = model_cfg['control'].get('version', 4)
    mortal = Brain(version=version, **model_cfg['resnet']).eval()
    dqn = DQN(version=version, num_heads=model_cfg.get('dqn', {}).get('num_heads', 1)).eval()
    # 旧 checkpoint 无 policy_head，strict=False 随机初始化并退回 q 选动作
    mortal.load_state_dict(state['mortal'], strict=False)
    dqn.load_state_dict(state['current_dqn'])
    if cfg['enable_compile']:
        mortal.compile()
        dqn.compile()
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(cfg['device']),
        enable_amp=cfg['enable_amp'],
        enable_rule_based_agari_guard=cfg['enable_rule_based_agari_guard'],
        name=cfg['name'],
        action_source='policy' if 'policy_head.weight' in state['mortal'] else 'q',
    )


def setup_akochan_env(cfg):
    os.environ['AKOCHAN_DIR'] = cfg['dir']
    os.environ['AKOCHAN_TACTICS'] = cfg['tactics']


def main():
    cfg = config['1v3']
    games_per_iter = cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = os.path.join(cfg['log_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=True)
    one = cfg['one']
    three = cfg['three']

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    if one['type'] == 'akochan':
        setup_akochan_env(one)
    if three['type'] == 'akochan':
        setup_akochan_env(three)
    engine_one = load_py_engine(one) if one['type'] == 'py' else None
    engine_three = load_py_engine(three) if three['type'] == 'py' else None

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar=False,
            log_dir=log_dir,
        )
        match (three['type'], one['type']):
            case ('py', 'py'):
                one_rankings, three_rankings = env.py_vs_py(
                    challenger=engine_one,
                    champion=engine_three,
                    seed_start=(seed, key),
                    seed_count=seeds_per_iter,
                )
            case ('py', 'akochan'):
                one_rankings, three_rankings = env.py3_vs_ako(
                    engine=engine_three,
                    seed_start=(seed, key),
                    seed_count=seeds_per_iter,
                )
            case ('akochan', 'py'):
                one_rankings, three_rankings = env.ako_vs_py(
                    engine=engine_three,
                    seed_start=(seed, key),
                    seed_count=seeds_per_iter,
                )
            case _:
                raise ValueError(f'不支持的组合: three={three["type"]}, one={one["type"]}')
        for name, rankings in [('one', one_rankings), ('three', three_rankings)]:
            rankings = np.array(rankings)
            avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
            avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
            print(f'{name} rankings: {rankings} ({avg_rank}, {avg_pt}pt)')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
