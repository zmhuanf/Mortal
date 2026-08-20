"""mortal_base_v2 训练：纯策略行为克隆（CE 单损失）

首次运行从 mortal_base/out/mortal.pth 复制 policy 参数，之后只保存 policy 权重
post_training 后训练：随机读文件 + 无步数上限，Ctrl+C 手动停止
"""

import os
import sys
import math
import random
import logging
from os import path
from glob import glob
from datetime import datetime

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config  # noqa: F401  注册 sys.modules['config']
from config import config
from model import Brain
from dataset import FileDatasetsIter, worker_init_fn
from evaluate import make_engine, run_eval

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def build_file_list():
    file_index = config['dataset']['file_index']
    if path.exists(file_index):
        return torch.load(file_index, weights_only=True)['file_list']
    file_list = []
    for pat in config['dataset']['globs']:
        file_list.extend(glob(pat, recursive=True))
    file_list.sort(reverse=True)
    os.makedirs(path.dirname(file_index), exist_ok=True)
    torch.save({'file_list': file_list}, file_index)
    return file_list


def main():
    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    amp_dtype = torch.bfloat16 if config['control'].get('amp_dtype', 'float16') == 'bfloat16' else torch.float16
    version = config['control']['version']
    state_file = config['control']['state_file']
    init_from = config['control'].get('init_from')

    brain = Brain(version=version, **config['model']).to(device)
    logging.info(f'brain params: {sum(p.numel() for p in brain.parameters()):,}')

    decay_params, no_decay_params = [], []
    params_dict = {}
    to_decay = set()
    for mod_name, mod in brain.named_modules():
        for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
            params_dict[name] = param
            if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                to_decay.add(name)
    decay_params.extend(params_dict[n] for n in sorted(to_decay))
    no_decay_params.extend(params_dict[n] for n in sorted(params_dict.keys() - to_decay))
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': config['optim']['weight_decay']},
        {'params': no_decay_params},
    ], lr=1, weight_decay=0, betas=config['optim']['betas'], eps=config['optim']['eps'])
    scaler = GradScaler(device.type, enabled=enable_amp and amp_dtype == torch.float16)

    steps = 0
    data_offset = 0
    best_perf = {'avg_rank': 4.0, 'avg_pt': -135.0}
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=False, map_location=device)
        brain.load_state_dict(state['mortal'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        steps = state['steps']
        data_offset = state.get('data_offset', 0)
        best_perf = state.get('best_perf', best_perf)
        logging.info(f'resumed from step {steps} (data offset {data_offset})')
    elif init_from and path.exists(init_from):
        src = torch.load(init_from, weights_only=True, map_location=device)
        brain.load_state_dict(src['mortal'])
        logging.info(f'initialized weights from {init_from} (no optimizer state)')

    optimizer.zero_grad(set_to_none=True)
    writer = SummaryWriter(config['control']['tensorboard_dir'])

    if config['control']['enable_compile']:
        brain.compile()

    file_list = build_file_list()
    logging.info(f'offline files: {len(file_list):,}')
    ds_cfg = config['dataset']
    post_training = config['train'].get('post_training', False)
    max_steps = config['train']['max_steps']
    data = FileDatasetsIter(
        version=version, file_list=file_list,
        file_batch_size=ds_cfg['file_batch_size'],
        num_epochs=ds_cfg['num_epochs'],
        enable_augmentation=ds_cfg['enable_augmentation'],
        augmented_first=ds_cfg['augmented_first'],
        resume_files=data_offset,
        shuffle_seed=ds_cfg.get('shuffle_seed', 42),
        random_files=post_training,
    )
    loader = iter(DataLoader(
        data, batch_size=config['control']['batch_size'], drop_last=False,
        num_workers=ds_cfg['num_workers'], pin_memory=True, worker_init_fn=worker_init_fn,
        prefetch_factor=ds_cfg.get('prefetch_factor', 4), persistent_workers=True,
    ))
    B = config['control']['batch_size']
    lr_peak = config['optim']['lr']
    warm_up = config['optim']['warm_up_steps']
    ce = nn.CrossEntropyLoss()
    eval_every = config['train']['eval_every']
    eval_games = config['train']['eval_games']
    best_state_file = config['control']['best_state_file']
    eval_log_dir = config['eval']['log_dir']

    def current_lr():
        if steps < warm_up:
            return lr_peak * steps / warm_up
        return lr_peak

    def train_batch(obs, actions):
        nonlocal steps
        obs = obs.to(device=device, non_blocking=True)
        actions = actions.to(device=device, non_blocking=True)
        lr = current_lr()
        for g in optimizer.param_groups:
            g['lr'] = lr
        with torch.autocast(device.type, dtype=amp_dtype, enabled=enable_amp):
            logits = brain(obs)
            loss = ce(logits, actions)
        scaler.scale(loss).backward()
        steps += 1
        if config['optim']['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(optimizer.param_groups[0]['params'], config['optim']['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return loss

    def maybe_evaluate():
        nonlocal best_perf
        brain.eval()
        engine = make_engine(brain, device, version, enable_amp=True)
        result = run_eval(engine, device, games=eval_games, log_dir=eval_log_dir)
        brain.train()
        writer.add_scalar('eval/avg_rank', result['avg_rank'], steps)
        writer.add_scalar('eval/avg_pt', result['avg_pt'], steps)
        writer.flush()
        logging.info(f'eval @{steps}: avg_rank {result["avg_rank"]:.4f} avg_pt {result["avg_pt"]:.4f}')
        if (result['avg_rank'] < best_perf['avg_rank']
                or (result['avg_rank'] == best_perf['avg_rank'] and result['avg_pt'] > best_perf['avg_pt'])):
            best_perf = {'avg_rank': result['avg_rank'], 'avg_pt': result['avg_pt']}
            torch.save({'mortal': brain.state_dict(), 'config': config,
                        'steps': steps, 'best_perf': best_perf,
                        'timestamp': datetime.now().timestamp()}, best_state_file)
            logging.info(f'new best @{steps}: avg_rank {result["avg_rank"]:.4f}')

    loss_sum = 0.0
    n_batch = 0
    pb = tqdm(total=None if post_training else max_steps, initial=steps, desc='BC', unit='step')
    while True:
        try:
            obs, actions, _masks = next(loader)
        except StopIteration:
            break
        if obs.shape[0] != B:
            continue
        loss = train_batch(obs, actions)
        loss_sum += loss.item()
        n_batch += 1
        pb.update(1)
        if steps % config['control']['save_every'] == 0:
            writer.add_scalar('loss/policy', loss_sum / n_batch, steps)
            writer.add_scalar('hparam/lr', optimizer.param_groups[0]['lr'], steps)
            writer.flush()
            loss_sum = 0.0
            n_batch = 0
            torch.save({'mortal': brain.state_dict(), 'config': config,
                        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                        'steps': steps, 'data_offset': data.cursor.value,
                        'best_perf': best_perf, 'timestamp': datetime.now().timestamp()}, state_file)
            logging.info(f'step {steps:,} | loss {loss.item():.4f}')
        if eval_every > 0 and steps % eval_every == 0:
            maybe_evaluate()
        if not post_training and steps >= max_steps:
            break
    pb.close()
    logging.info(f'training done at step {steps}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
