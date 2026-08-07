import prelude
import config_v2  # 注册 config 模块，使用 v2 单文件配置

import logging
import random
import socket
import torch
import numpy as np
import time
import gc
from os import path
from model import Brain, DQN
from player import TrainPlayer
from common import send_msg, recv_msg, get_pool, get_opponent
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
    champion_file = None
    last_pool_version = -1

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
        pool_version = pool['version']
        if pool_version != last_pool_version:
            # 对手池升级说明 trainee 已晋级，清空历史重置探索强度
            history.clear()
            last_pool_version = pool_version
        current_opponent = random.choice(pool['opponents'])
        if current_opponent['state_file'] != champion_file:
            # 跨机时本地无权重文件，从 server 拉取对手权重
            rsp = get_opponent(current_opponent['id'])
            if rsp.get('status') == 'not found':
                logging.warning(f'opponent {current_opponent["name"]} evicted, re-fetching pool')
                last_pool_version = -1
                continue
            train_player.load_champion_state(rsp['state'], current_opponent['name'])
            champion_file = current_opponent['state_file']

        # 表现自适应探索，trainee 越强温度与 epsilon 越低，首轮无历史用上限充分探索
        if history:
            sum_rankings = np.sum(history, axis=0)
            ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()
            progress = float(np.clip(ma_avg_pt / train_player.target_pt, 0, 1))
        else:
            ma_avg_pt = -float('inf')
            progress = 0.
        temperature = train_player.temp_max * (train_player.temp_min / train_player.temp_max) ** progress
        epsilon = train_player.boltzmann_epsilon * (train_player.eps_min / train_player.boltzmann_epsilon) ** progress
        logging.info(f'exploration: temperature={temperature:.4f} epsilon={epsilon:.4f} (ma_avg_pt={ma_avg_pt:.4f})')

        rankings, file_list = train_player.train_play(mortal, dqn, device, temperature, epsilon)
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
