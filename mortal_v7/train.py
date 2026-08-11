"""mortal_v7 训练：Decision Transformer 纯监督训练

对每个决策点预测动作（交叉熵），无 bootstrap 无值函数
"""

import os
import logging
from os import path
from glob import glob
from datetime import datetime
from itertools import chain

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config_v7  # 注册 config 模块，必须先于 from config import config
from config import config
from model import DecisionTransformer
from dataset import FileDatasetsIter, worker_init_fn, collate_single, collate_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def build_file_list():
    file_index = config['dataset']['file_index']
    if path.exists(file_index):
        file_list = torch.load(file_index, weights_only=True)['file_list']
    else:
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

    model = DecisionTransformer(**config['model']).to(device)
    logging.info(f'mortal params: {sum(p.numel() for p in model.parameters()):,}')

    decay_params, no_decay_params = [], []
    for name, mod in model.named_modules():
        for pname, param in mod.named_parameters(prefix=name, recurse=False):
            if isinstance(mod, (nn.Linear, nn.Conv1d)) and pname.endswith('weight'):
                decay_params.append(param)
            else:
                no_decay_params.append(param)
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': config['optim']['weight_decay']},
        {'params': no_decay_params},
    ], lr=config['optim']['lr'], weight_decay=0, betas=config['optim']['betas'], eps=config['optim']['eps'])
    scaler = GradScaler(device.type, enabled=enable_amp)

    steps = 0
    data_offset = 0
    state_file = config['control']['state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=False, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        steps = state['steps']
        data_offset = state.get('data_offset', 0)
        logging.info(f'resumed from step {steps} (data offset {data_offset})')

    optimizer.zero_grad(set_to_none=True)
    writer = SummaryWriter(config['control']['tensorboard_dir'])

    file_list = build_file_list()
    logging.info(f'offline files: {len(file_list):,}')
    ds_cfg = config['dataset']
    data = FileDatasetsIter(
        version=config['control']['version'],
        file_list=file_list,
        pts=config['env']['pts'],
        file_batch_size=ds_cfg['file_batch_size'],
        reserve_ratio=ds_cfg['reserve_ratio'],
        num_epochs=ds_cfg['num_epochs'],
        enable_augmentation=ds_cfg['enable_augmentation'],
        augmented_first=ds_cfg['augmented_first'],
        resume_files=data_offset,
        batch_size=config['control']['batch_size'],
    )
    loader_kwargs = {
        'batch_size': 1,  # dataset 已产出组好的 batch，此处仅解包
        'drop_last': False,
        'num_workers': ds_cfg['num_workers'],
        'pin_memory': True,
        'collate_fn': collate_single,
        'worker_init_fn': worker_init_fn,
    }
    if ds_cfg['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = ds_cfg['prefetch_factor']
        loader_kwargs['persistent_workers'] = ds_cfg['persistent_workers']
    loader = iter(DataLoader(data, **loader_kwargs))

    loss_sum = 0.0
    acc_sum = 0.0
    n_batch = 0
    pending = []
    B = config['control']['batch_size']
    pb = tqdm(total=config['train']['max_steps'], initial=steps, desc='TRAIN', unit='step')
    for entry in loader:
        pending.append(entry)
        if len(pending) < B:
            continue
        # 主进程攒批：T 排序组 batch，段级到达均匀不形成 GPU 饥饿
        pending.sort(key=lambda e: e[0].shape[0])
        chunk = pending[:B]
        del pending[:B]
        obs, rtg, acts, masks, valid = collate_batch(chunk)
        obs = obs.to(device=device, non_blocking=True)  # 保持 bf16 省显存
        rtg = rtg.to(device=device, non_blocking=True)
        acts = acts.to(device=device, non_blocking=True)
        valid = valid.to(device=device, non_blocking=True)

        with torch.autocast(device.type, enabled=enable_amp):
            logits = model(obs, rtg, acts)  # (B, 3T, A)
            # 动作位置 p = 3t + 2
            act_logits = logits[:, 2::3]
            loss = F.cross_entropy(act_logits[valid], acts[valid])

        scaler.scale(loss).backward()
        steps += 1
        n_batch += 1
        loss_sum += loss.item()
        acc_sum += (act_logits[valid].argmax(-1) == acts[valid]).float().mean().item()

        if config['optim']['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
            clip_grad_norm_(params, config['optim']['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        pb.update(1)

        if steps % config['control']['save_every'] == 0:
            writer.add_scalar('loss/ce', loss_sum / n_batch, steps)
            writer.add_scalar('train/acc', acc_sum / n_batch, steps)
            writer.add_scalar('hparam/lr', config['optim']['lr'], steps)
            writer.add_scalar('hparam/rtg_mean', rtg.mean().item(), steps)
            writer.flush()
            loss_sum = 0.0
            acc_sum = 0.0
            n_batch = 0

            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'steps': steps,
                'data_offset': data.cursor.value,
                'timestamp': datetime.now().timestamp(),
                'config': config,
            }
            torch.save(state, state_file)
            logging.info(f'step {steps:,} | loss {loss.item():.4f}')

        if steps >= config['train']['max_steps']:
            pb.close()
            logging.info(f'training done at step {steps}')
            break
    else:
        # 数据耗尽自然结束（max_steps 未到）
        pb.close()
        logging.info(f'data exhausted at step {steps}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
