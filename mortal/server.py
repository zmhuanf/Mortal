import prelude

import logging
import shutil
import struct
import torch
import sys
import os
import json
from os import path
from io import BytesIO
from typing import *
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from socketserver import ThreadingTCPServer, BaseRequestHandler
from threading import Lock
from common import send_msg, recv_msg, UnexpectedEOF
from config import config

@dataclass
class State:
    buffer_dir: str
    drain_dir: str
    capacity: int
    force_sequential: bool
    opponents_dir: str
    dir_lock: Lock
    param_lock: Lock
    pool_lock: Lock
    # fields below are protected by dir_lock
    buffer_size: int
    submission_id: int
    # fields below are protected by param_lock
    mortal_param: Optional[OrderedDict]
    dqn_param: Optional[OrderedDict]
    param_version: int
    idle_param_version: int
    # fields below are protected by pool_lock
    pool_opponents: List[dict]
    pool_version: int
S = None

class Handler(BaseRequestHandler):
    def handle(self):
        msg = self.recv_msg()
        match msg['type']:
            # called by workers
            case 'get_param':
                self.handle_get_param(msg)
            case 'submit_replay':
                self.handle_submit_replay(msg)
            # called by trainer
            case 'submit_param':
                self.handle_submit_param(msg)
            case 'drain':
                self.handle_drain()
            # opponent pool
            case 'get_pool':
                self.handle_get_pool()
            case 'get_opponent':
                self.handle_get_opponent(msg)
            case 'promote':
                self.handle_promote(msg)

    def handle_get_param(self, msg):
        with S.dir_lock:
            overflow = S.buffer_size >= S.capacity
            with S.param_lock:
                has_param = S.mortal_param is not None and S.dqn_param is not None
        if overflow:
            self.send_msg({'status': 'samples overflow'})
            return
        if not has_param:
            self.send_msg({'status': 'empty param'})
            return

        client_param_version = msg['param_version']
        buf = BytesIO()
        with S.param_lock:
            if S.force_sequential and S.idle_param_version <= client_param_version:
                res = {'status': 'trainer is busy'}
            else:
                res = {
                    'status': 'ok',
                    'mortal': S.mortal_param,
                    'dqn': S.dqn_param,
                    'param_version': S.param_version,
                }
            torch.save(res, buf)
        self.send_msg(buf.getbuffer(), packed=True)

    def handle_submit_replay(self, msg):
        with S.dir_lock:
            for filename, content in msg['logs'].items():
                filepath = path.join(S.buffer_dir, f'{S.submission_id}_{filename}')
                with open(filepath, 'wb') as f:
                    f.write(content)
            S.buffer_size += len(msg['logs'])
            S.submission_id += 1
            logging.info(f'total buffer size: {S.buffer_size}')

    def handle_submit_param(self, msg):
        with S.param_lock:
            S.mortal_param = msg['mortal']
            S.dqn_param = msg['dqn']
            S.param_version += 1
            if msg['is_idle']:
                S.idle_param_version = S.param_version

    def handle_drain(self):
        drained_size = 0
        with S.dir_lock:
            buffer_list = os.listdir(S.buffer_dir)
            raw_count = len(buffer_list)
            assert raw_count == S.buffer_size
            if (not S.force_sequential or raw_count >= S.capacity) and raw_count > 0:
                old_drain_list = os.listdir(S.drain_dir)
                for filename in old_drain_list:
                    filepath = path.join(S.drain_dir, filename)
                    os.remove(filepath)
                for filename in buffer_list:
                    src = path.join(S.buffer_dir, filename)
                    dst = path.join(S.drain_dir, filename)
                    shutil.move(src, dst)
                drained_size = raw_count
                S.buffer_size = 0
                logging.info(f'files transferred to trainer: {drained_size}')
                logging.info(f'total buffer size: {S.buffer_size}')
        self.send_msg({
            'count': drained_size,
            'drain_dir': S.drain_dir,
        })

    def handle_get_pool(self):
        with S.pool_lock:
            self.send_msg({
                'status': 'ok',
                'version': S.pool_version,
                'opponents': S.pool_opponents,
            })

    def handle_get_opponent(self, msg):
        # client 与 server 跨机时本地无权重文件，需按 id 传输对手权重原始字节
        with S.pool_lock:
            state_file = S.pool_opponents[msg['id']]['state_file']
        with open(state_file, 'rb') as f:
            weights = f.read()
        # 直接传输原始字节，避免 torch.save 对大权重重复序列化导致内存峰值过高
        self.request.sendall(struct.pack('<Q', len(weights)))
        self.request.sendall(weights)

    def handle_promote(self, msg):
        with S.pool_lock:
            # 乐观锁校验，防止多 client 并发重复晋级同一对手
            if msg.get('pool_version') is not None and msg['pool_version'] != S.pool_version:
                logging.info(
                    f'promote rejected: stale pool_version={msg["pool_version"]} '
                    f'!= current={S.pool_version}'
                )
                self.send_msg({'status': 'stale', 'version': S.pool_version})
                return
            op_id = len(S.pool_opponents)
            name = f'op_{op_id:04d}'
            state_file = path.join(S.opponents_dir, f'{op_id:04d}_{name}.pth')
            torch.save({
                'mortal': msg['mortal'],
                'current_dqn': msg['dqn'],
                'config': config,
                'timestamp': datetime.now().timestamp(),
                'meta': msg.get('meta') or {},
            }, state_file)
            S.pool_opponents.append({
                'id': op_id,
                'name': name,
                'state_file': state_file,
                'meta': msg.get('meta') or {},
            })
            S.pool_version += 1
            write_index()
            meta = S.pool_opponents[-1]['meta']
            logging.info(
                f'opponent promoted: {name} (avg_rank={meta.get("avg_rank")}, '
                f'avg_pt={meta.get("avg_pt")}), pool_version={S.pool_version}'
            )
            self.send_msg({
                'status': 'ok',
                'version': S.pool_version,
                'current': S.pool_opponents[-1],
            })

    def send_msg(self, msg, packed=False):
        return send_msg(self.request, msg, packed)

    def recv_msg(self):
        return recv_msg(self.request)

def write_index():
    index_file = path.join(S.opponents_dir, 'index.json')
    tmp_file = index_file + '.tmp'
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(
            {'version': S.pool_version, 'opponents': S.pool_opponents},
            f, ensure_ascii=False, indent=2, default=float,
        )
    os.replace(tmp_file, index_file)

class Server(ThreadingTCPServer):
    def handle_error(self, request, client_address):
        typ, _, _ = sys.exc_info()
        if typ is BrokenPipeError or typ is UnexpectedEOF:
            return
        return super().handle_error(request, client_address)

def main():
    global S
    cfg = config['online']['server']
    opponents_dir = path.abspath(config['online']['pool']['opponents_dir'])
    S = State(
        buffer_dir = path.abspath(cfg['buffer_dir']),
        drain_dir = path.abspath(cfg['drain_dir']),
        capacity = cfg['capacity'],
        force_sequential = cfg['force_sequential'],
        opponents_dir = opponents_dir,
        dir_lock = Lock(),
        param_lock = Lock(),
        pool_lock = Lock(),
        buffer_size = 0,
        submission_id = 0,
        mortal_param = None,
        dqn_param = None,
        param_version = 0,
        idle_param_version = 0,
        pool_opponents = [],
        pool_version = 0,
    )
    load_pool()

    bind_addr = (config['online']['remote']['host'], config['online']['remote']['port'])
    if path.isdir(S.buffer_dir):
        shutil.rmtree(S.buffer_dir)
    if path.isdir(S.drain_dir):
        shutil.rmtree(S.drain_dir)
    os.makedirs(S.buffer_dir)
    os.makedirs(S.drain_dir)

    with Server(bind_addr, Handler, bind_and_activate=False) as server:
        server.allow_reuse_address = True
        server.daemon_threads = True
        server.server_bind()
        server.server_activate()
        host, port = bind_addr
        logging.info(f'listening on {host}:{port}')
        server.serve_forever()

def load_pool():
    os.makedirs(S.opponents_dir, exist_ok=True)
    index_file = path.join(S.opponents_dir, 'index.json')
    if path.exists(index_file):
        with open(index_file, encoding='utf-8') as f:
            data = json.load(f)
        S.pool_version = data['version']
        S.pool_opponents = data['opponents']
    else:
        S.pool_opponents = [{
            'id': 0,
            'name': 'baseline',
            'state_file': path.abspath(config['baseline']['train']['state_file']),
            'meta': {},
        }]
        S.pool_version = 1
        write_index()
        logging.info(f'initialized opponent pool at {S.opponents_dir}')
    logging.info(f'opponent pool v{S.pool_version}: {[op["name"] for op in S.pool_opponents]}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
