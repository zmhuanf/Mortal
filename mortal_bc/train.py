"""纯加权行为克隆训练：固定学习率，可随时续训

用法：python train.py
"""
import os
import sys
import gc
import shutil
import logging
from os import path
from glob import glob
from datetime import datetime
from itertools import chain

ROOT = path.dirname(path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from config import config
from model import Brain, AuxNet
from dataset import FileDatasetsIter, worker_init_fn
from evaluate import run_eval


def parameter_count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_file_list():
    idx = config['dataset']['file_index']
    if path.exists(idx):
        return torch.load(idx, weights_only=True)['file_list']
    fs = [f for pat in config['dataset']['globs'] for f in glob(pat, recursive=True)]
    fs.sort(reverse=True)
    os.makedirs(path.dirname(idx), exist_ok=True)
    torch.save({'file_list': fs}, idx)
    return fs


def build_optimizer(models):
    """AdamW 分组：Conv1d/Linear 的 weight 衰减，其余不衰减"""
    decay, no_decay = [], []
    for m in models:
        pd, to_decay = {}, set()
        for name, mod in m.named_modules():
            for n, p in mod.named_parameters(prefix=name, recurse=False):
                pd[n] = p
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and n.endswith('weight'):
                    to_decay.add(n)
        decay.extend(pd[n] for n in sorted(to_decay))
        no_decay.extend(pd[n] for n in sorted(pd.keys() - to_decay))
    o = config['optim']
    return optim.AdamW(
        [{'params': decay, 'weight_decay': o['weight_decay']}, {'params': no_decay}],
        lr=o['lr'], betas=o['betas'], eps=o['eps'],
    )


def make_loader(version):
    ds_cfg = config['dataset']
    ds = FileDatasetsIter(
        version=version,
        file_list=build_file_list(),
        file_batch_size=ds_cfg['file_batch_size'],
        num_epochs=ds_cfg['num_epochs'],
        enable_augmentation=ds_cfg['enable_augmentation'],
    )
    kwargs = {
        'batch_size': config['control']['batch_size'],
        'drop_last': False,
        'num_workers': ds_cfg['num_workers'],
        'pin_memory': True,
        'worker_init_fn': worker_init_fn,
    }
    # num_workers>0 时启用预取与持久 worker，让数据供给跟上 GPU 消费
    if ds_cfg['num_workers'] > 0:
        kwargs['prefetch_factor'] = ds_cfg['prefetch_factor']
        kwargs['persistent_workers'] = ds_cfg['persistent_workers']
    return iter(DataLoader(ds, **kwargs))


def save_checkpoint(f, *, mortal, aux_net, optimizer, steps, best_eval):
    torch.save({
        'mortal': mortal.state_dict(),
        'aux_net': aux_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'steps': steps,
        'best_eval': best_eval,
        'timestamp': datetime.now().timestamp(),
        'config': config,
    }, f)


def do_eval(mortal, aux_net, optimizer, device, steps, best_eval, writer, state_file, best_file):
    mortal.eval()
    r = run_eval(mortal, device)
    mortal.train()
    avg_rank, avg_pt = r['avg_rank'], r['avg_pt']
    writer.add_scalar('eval/avg_rank', avg_rank, steps)
    writer.add_scalar('eval/avg_pt', avg_pt, steps)
    writer.flush()

    better = best_eval is None or avg_pt > best_eval['avg_pt'] or (
        avg_pt == best_eval['avg_pt'] and avg_rank < best_eval['avg_rank']
    )
    if better:
        best_eval = {'avg_rank': avg_rank, 'avg_pt': avg_pt, 'steps': steps}
        save_checkpoint(state_file, mortal=mortal, aux_net=aux_net,
                        optimizer=optimizer, steps=steps, best_eval=best_eval)
        shutil.copy(state_file, best_file)
        logging.info(f'new best: {avg_pt:.4f}pt / {avg_rank:.4f} rank @ {steps:,}')
    else:
        logging.info(f'eval @ {steps:,}: {avg_pt:.4f}pt / {avg_rank:.4f} rank (best {best_eval["avg_pt"]:.4f}pt)')
    return best_eval


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format='%(asctime)s %(levelname)8s %(message)s')
    ctrl = config['control']
    device = torch.device(ctrl['device'])
    torch.backends.cudnn.benchmark = ctrl['enable_cudnn_benchmark']
    enable_amp = ctrl['enable_amp']
    version = ctrl['version']

    mortal = Brain(version=version, **config['model']).to(device)
    aux_net = AuxNet().to(device)
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    optimizer = build_optimizer([mortal, aux_net])
    ce = nn.CrossEntropyLoss()
    rank_w = torch.tensor(config['bc']['rank_weights'], device=device)
    aux_w = config['aux']
    max_grad = config['optim']['max_grad_norm']

    steps = 0
    best_eval = None
    state_file = ctrl['state_file']
    best_file = ctrl['best_state_file']
    if path.exists(state_file):
        s = torch.load(state_file, map_location=device, weights_only=True)
        mortal.load_state_dict(s['mortal'])
        aux_net.load_state_dict(s['aux_net'])
        optimizer.load_state_dict(s['optimizer'])
        steps = s['steps']
        best_eval = s.get('best_eval')
        # 固定 lr：续训时强制用当前 config 的 lr，忽略 checkpoint 内旧值
        for g in optimizer.param_groups:
            g['lr'] = config['optim']['lr']
        logging.info(f'resume from step {steps:,}')

    writer = SummaryWriter(ctrl['tensorboard_dir'])
    loader = make_loader(version)
    save_every = ctrl['save_every']
    eval_every = ctrl['eval_every']
    stats = {'ce': 0., 'next_rank': 0., 'shanten': 0., 'fuuro': 0., 'riichi': 0.}
    nb = 0
    pb = tqdm(desc='BC', unit='batch', dynamic_ncols=True, ascii=True)

    while True:
        try:
            batch = next(loader)
        except StopIteration:
            loader = make_loader(version)
            logging.info(f'epoch done @ {steps:,}')
            continue

        obs, actions, masks, final_rank, next_rank, shanten, fuuro, riichi = batch
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        actions = actions.to(device=device, non_blocking=True)
        masks = masks.to(device=device, non_blocking=True)
        final_rank = final_rank.to(device=device, non_blocking=True)
        next_rank = next_rank.to(device=device, non_blocking=True)
        shanten = shanten.to(device=device, non_blocking=True)
        fuuro = fuuro.to(device=device, non_blocking=True)
        riichi = riichi.to(device=device, non_blocking=True)
        weight = rank_w[final_rank]

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enable_amp):
            phi = mortal(obs)
            policy_logits = mortal.policy_logits(phi)
            ce_per = F.cross_entropy(policy_logits, actions, reduction='none')
            ce_loss = (ce_per * weight).mean()
            nr, sh, fc, rt = aux_net(phi)
            l_nr = ce(nr, next_rank)
            l_sh = ce(sh, shanten)
            l_fc = ce(fc, fuuro)
            l_rt = ce(rt, riichi)
            loss = ce_loss + (
                l_nr * aux_w['next_rank_weight']
                + l_sh * aux_w['shanten_weight']
                + l_fc * aux_w['fuuro_weight']
                + l_rt * aux_w['riichi_turn_weight']
            )

        stats['ce'] += ce_loss.item()
        stats['next_rank'] += l_nr.item()
        stats['shanten'] += l_sh.item()
        stats['fuuro'] += l_fc.item()
        stats['riichi'] += l_rt.item()
        nb += 1

        loss.backward()
        if max_grad > 0:
            clip_grad_norm_(chain.from_iterable(g['params'] for g in optimizer.param_groups), max_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        pb.update(1)

        if steps % save_every == 0:
            save_checkpoint(state_file, mortal=mortal, aux_net=aux_net,
                            optimizer=optimizer, steps=steps, best_eval=best_eval)
            writer.add_scalar('loss/ce', stats['ce'] / nb, steps)
            writer.add_scalar('loss/next_rank', stats['next_rank'] / nb, steps)
            writer.add_scalar('loss/shanten', stats['shanten'] / nb, steps)
            writer.add_scalar('loss/fuuro', stats['fuuro'] / nb, steps)
            writer.add_scalar('loss/riichi', stats['riichi'] / nb, steps)
            writer.flush()
            for k in stats:
                stats[k] = 0
            nb = 0

        if steps % eval_every == 0:
            best_eval = do_eval(mortal, aux_net, optimizer, device, steps, best_eval, writer, state_file, best_file)
            gc.collect()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
