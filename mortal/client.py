import prelude
import config_v2  # 注册 config 模块，使用 v2 单文件配置

import logging
import socket
import torch
import numpy as np
import time
import gc
from io import BytesIO
from os import path
from model import Brain, DQN
from player import TrainPlayer
from common import send_msg, recv_msg, get_pool, get_opponent, promote
from config import config

def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']

    mortal = Brain(version=version, **config['resnet']).to(device).eval()
    dqn = DQN(version=version, num_heads=config.get('dqn', {}).get('num_heads', 1)).to(device)
    if config['online']['enable_compile']:
        mortal.compile()
        dqn.compile()

    train_player = TrainPlayer()
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    pool_cfg = config['online']['pool']
    promote_avg_rank = pool_cfg['promote_avg_rank']
    promote_min_sessions = pool_cfg['promote_min_sessions']
    promote_cooldown = pool_cfg['promote_cooldown']
    champion_file = None
    cooldown_left = 0

    while True:
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        mortal.load_state_dict(rsp['mortal'])
        dqn.load_state_dict(rsp['dqn'])
        logging.info('param has been updated')

        pool = get_pool()
        current_opponent = pool['opponents'][-1]
        pool_version = pool['version']
        if current_opponent['state_file'] != champion_file:
            # 跨机时本地无权重文件，从 server 拉取对手权重
            rsp = get_opponent(current_opponent['id'])
            state = torch.load(BytesIO(rsp['weights']), weights_only=True, map_location=device)
            train_player.load_champion_state(state, current_opponent['name'])
            champion_file = current_opponent['state_file']

        # 表现自适应探索温度，trainee 越强温度越低，首轮无历史用上限充分探索
        if history:
            sum_rankings = np.sum(history, axis=0)
            ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()
            progress = float(np.clip(ma_avg_pt / train_player.target_pt, 0, 1))
        else:
            ma_avg_pt = -float('inf')
            progress = 0.
        temperature = train_player.temp_max * (train_player.temp_min / train_player.temp_max) ** progress
        logging.info(f'exploration temperature: {temperature:.4f} (ma_avg_pt={ma_avg_pt:.4f})')

        rankings, file_list = train_player.train_play(mortal, dqn, device, temperature)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        cooldown_left = max(0, cooldown_left - 1)
        if len(history) >= promote_min_sessions and cooldown_left == 0 and ma_avg_rank < promote_avg_rank:
            rsp = promote(mortal, dqn, {
                'avg_rank': float(ma_avg_rank),
                'avg_pt': float(ma_avg_pt),
                'sessions': len(history),
            }, pool_version=pool_version)
            if rsp['status'] == 'stale':
                logging.info(f'promote rejected: pool already updated to v{rsp["version"]}, resetting history')
            else:
                logging.info(
                    f'promoted to opponent pool v{rsp["version"]}: {rsp["current"]["name"]} '
                    f'(ma_avg_rank={ma_avg_rank:.6}, ma_avg_pt={ma_avg_pt:.6}pt)'
                )
                cooldown_left = promote_cooldown
                champion_file = None
            history.clear()

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
