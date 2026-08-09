"""mortal_v5 离线两阶段训练：BC 预训练 → IQL 精调

用法：python train.py
阶段与步数由 config_v5.py 的 train 段控制，BC 完成后自动切换 IQL
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
MORTAL_DIR = path.join(ROOT, '..', 'mortal')
# 追加而非前置：mortal_v5 同目录文件（model/dataloader）优先，libriichi 等仍可解析
sys.path.append(MORTAL_DIR)

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import config_v5  # 注册 config 模块，必须先于 from config import config
from config import config
from dataloader import FileDatasetsIter, worker_init_fn
from lr_scheduler import LinearWarmUpCosineAnnealingLR
from model import Brain, DQN, AuxNet
from libriichi.consts import obs_shape
from evaluate import run_eval


def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_optimizer(models):
    """AdamW 分组：Linear/Conv1d 的 weight 走 weight_decay，其余不衰减"""
    train_cfg = config['train']
    decay, no_decay = [], []
    for model in models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay.extend(params_dict[n] for n in sorted(to_decay))
        no_decay.extend(params_dict[n] for n in sorted(params_dict.keys() - to_decay))
    return optim.AdamW(
        [{'params': decay, 'weight_decay': train_cfg['weight_decay']}, {'params': no_decay}],
        lr=1, weight_decay=0, betas=train_cfg['betas'], eps=train_cfg['eps'],
    )


def make_loader(version):
    dataset_cfg = config['dataset']
    file_index = dataset_cfg['file_index']
    if path.exists(file_index):
        file_list = torch.load(file_index, weights_only=True)['file_list']
    else:
        file_list = [f for pat in dataset_cfg['globs'] for f in glob(pat, recursive=True)]
        file_list.sort(reverse=True)
        os.makedirs(path.dirname(file_index), exist_ok=True)
        torch.save({'file_list': file_list}, file_index)
    data = FileDatasetsIter(
        version=version,
        file_list=file_list,
        pts=config['env']['pts'],
        file_batch_size=dataset_cfg['file_batch_size'],
        reserve_ratio=dataset_cfg['reserve_ratio'],
        num_epochs=dataset_cfg['num_epochs'],
        enable_augmentation=dataset_cfg['enable_augmentation'],
        augmented_first=dataset_cfg['augmented_first'],
        include_final_rank=False,
        include_kyoku_delta=False,
    )
    loader_kwargs = {
        'dataset': data,
        'batch_size': config['control']['batch_size'],
        'drop_last': False,
        'num_workers': dataset_cfg['num_workers'],
        'pin_memory': True,
        'worker_init_fn': worker_init_fn,
    }
    if dataset_cfg['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = dataset_cfg['prefetch_factor']
        loader_kwargs['persistent_workers'] = dataset_cfg['persistent_workers']
    return iter(DataLoader(**loader_kwargs))


def save_checkpoint(state_file, *, mortal, aux_net, steps, stage, best_eval, epoch=0,
                    dqn=None, target_mortal=None, target_dqn=None, optimizer=None, scheduler=None):
    state = {
        'mortal': mortal.state_dict(),
        'aux_net': aux_net.state_dict(),
        'steps': steps,
        'stage': stage,
        'epoch': epoch,
        'timestamp': datetime.now().timestamp(),
        'best_eval': best_eval,
        'config': config,
    }
    for key, obj in (('dqn', dqn), ('target_mortal', target_mortal), ('target_dqn', target_dqn),
                     ('optimizer', optimizer), ('scheduler', scheduler)):
        if obj is not None:
            state[key] = obj.state_dict()
    torch.save(state, state_file)


def maybe_eval(mortal, dqn, device, steps, best_eval, writer, state_file, best_state_file):
    mortal.eval()
    if dqn is not None:
        dqn.eval()
    result = run_eval(mortal, dqn, device)
    mortal.train()
    if dqn is not None:
        dqn.train()

    writer.add_scalar('eval/avg_rank', result['avg_rank'], steps)
    writer.add_scalar('eval/avg_pt', result['avg_pt'], steps)
    for name, rankings, _ in result['results']:
        op_avg = sum((i + 1) * c for i, c in enumerate(rankings)) / max(1, sum(rankings))
        writer.add_scalar(f'eval/{name}/avg_rank', op_avg, steps)

    better = best_eval is None or (
        result['avg_pt'] >= best_eval['avg_pt'] and result['avg_rank'] <= best_eval['avg_rank']
    )
    if better:
        best_eval = {'avg_rank': float(result['avg_rank']), 'avg_pt': float(result['avg_pt']), 'steps': steps}
        if path.exists(state_file):
            shutil.copy(state_file, best_state_file)
        logging.info(f'new best: {best_eval["avg_pt"]:.4}pt / {best_eval["avg_rank"]:.4} rank @ {steps:,}')
    else:
        logging.info(f'eval @ {steps:,}: {result["avg_pt"]:.4}pt / {result["avg_rank"]:.4} rank (best {best_eval["avg_pt"]:.4}pt)')
    writer.flush()
    return best_eval


def train_bc(mortal, aux_net, device, enable_amp):
    """BC 预训练：策略 CE + 辅助任务，在人类牌谱上学强先验"""
    cfg = config
    ctrl = cfg['control']
    state_file = ctrl['state_file']
    best_state_file = ctrl['best_state_file']
    version = ctrl['version']
    opt_step_every = ctrl['opt_step_every']
    save_every = ctrl['save_every']
    eval_every = cfg['eval']['eval_every']
    train_cfg = cfg['train']
    aux_w = cfg['aux']
    target = train_cfg['bc_steps']
    max_grad_norm = train_cfg['max_grad_norm']

    optimizer = build_optimizer([mortal, aux_net])
    scheduler = LinearWarmUpCosineAnnealingLR(
        optimizer,
        peak=train_cfg['bc_peak'], final=train_cfg['bc_final'],
        warm_up_steps=train_cfg['warm_up_steps'], max_steps=target,
    )

    steps = 0
    epoch = 0
    best_eval = None
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        if state['stage'] == 'bc':
            mortal.load_state_dict(state['mortal'])
            aux_net.load_state_dict(state['aux_net'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.last_epoch = state['scheduler']['last_epoch']
            steps = state['steps']
            epoch = state.get('epoch', 0)
            best_eval = state.get('best_eval')
            logging.info(f'resume BC from step {steps:,}')
        elif state['stage'] == 'iql':
            # 用户把阶段切回 BC：优先取 best，否则沿用 iql 最后模型状态（optimizer 不兼容故重建）
            src = best_state_file if path.exists(best_state_file) else state_file
            state = torch.load(src, weights_only=True, map_location=device)
            mortal.load_state_dict(state['mortal'])
            aux_net.load_state_dict(state['aux_net'])
            best_eval = state.get('best_eval')
            logging.info('restart BC from checkpoint')

    writer = SummaryWriter(os.path.join(ctrl['tensorboard_dir'], 'bc'))
    loader = make_loader(version)
    pb = tqdm(total=target, initial=steps, desc='BC', unit='batch', dynamic_ncols=True)
    stats = {'policy': 0., 'next_rank': 0., 'shanten': 0., 'fuuro': 0., 'riichi_turn': 0.}
    n_batches = 0

    while steps < target:
        try:
            batch = next(loader)
        except StopIteration:
            loader = make_loader(version)
            epoch += 1
            logging.info(f'BC epoch {epoch} done @ step {steps:,}')
            continue

        obs, actions, _masks, player_ranks, *_rest = batch
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
        player_ranks = player_ranks.to(dtype=torch.int64, device=device, non_blocking=True)
        shantens = batch[8].to(dtype=torch.int64, device=device, non_blocking=True)
        fuuro_counts = batch[9].to(dtype=torch.int64, device=device, non_blocking=True)
        riichi_turns = batch[10].to(dtype=torch.int64, device=device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enable_amp):
            phi = mortal(obs)
            policy_loss = F.cross_entropy(mortal.policy_logits(phi), actions)
            next_rank_logits, shanten_logits, fuuro_logits, riichi_turn_logits = aux_net(phi)
            next_rank_loss = F.cross_entropy(next_rank_logits, player_ranks)
            shanten_loss = F.cross_entropy(shanten_logits, shantens)
            fuuro_loss = F.cross_entropy(fuuro_logits, fuuro_counts)
            riichi_turn_loss = F.cross_entropy(riichi_turn_logits, riichi_turns)
            loss = (
                policy_loss
                + next_rank_loss * aux_w['next_rank_weight']
                + shanten_loss * aux_w['shanten_weight']
                + fuuro_loss * aux_w['fuuro_weight']
                + riichi_turn_loss * aux_w['riichi_turn_weight']
            )

        stats['policy'] += policy_loss.item()
        stats['next_rank'] += next_rank_loss.item()
        stats['shanten'] += shanten_loss.item()
        stats['fuuro'] += fuuro_loss.item()
        stats['riichi_turn'] += riichi_turn_loss.item()
        n_batches += 1

        (loss / opt_step_every).backward()
        steps += 1
        if steps % opt_step_every == 0:
            if max_grad_norm > 0:
                clip_grad_norm_(chain.from_iterable(g['params'] for g in optimizer.param_groups), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        pb.update(1)

        if steps % save_every == 0:
            save_checkpoint(
                state_file, mortal=mortal, aux_net=aux_net, steps=steps, stage='bc', epoch=epoch,
                best_eval=best_eval, optimizer=optimizer, scheduler=scheduler,
            )
            writer.add_scalar('data/epoch', epoch, steps)
            writer.add_scalar('loss/policy', stats['policy'] / n_batches, steps)
            writer.add_scalar('loss/next_rank', stats['next_rank'] / n_batches, steps)
            writer.add_scalar('loss/shanten', stats['shanten'] / n_batches, steps)
            writer.add_scalar('loss/fuuro', stats['fuuro'] / n_batches, steps)
            writer.add_scalar('loss/riichi_turn', stats['riichi_turn'] / n_batches, steps)
            writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
            writer.flush()
            for k in stats:
                stats[k] = 0
            n_batches = 0
        if steps % eval_every == 0:
            best_eval = maybe_eval(mortal, None, device, steps, best_eval, writer, state_file, best_state_file)

    pb.close()
    if train_cfg['auto_proceed']:
        # 用 BC best 作为 IQL 起点，保存阶段切换点（无 optimizer，IQL 全新起步）
        if path.exists(best_state_file):
            state = torch.load(best_state_file, weights_only=True, map_location=device)
            mortal.load_state_dict(state['mortal'])
            aux_net.load_state_dict(state['aux_net'])
            best_eval = state.get('best_eval')
        save_checkpoint(state_file, mortal=mortal, aux_net=aux_net, steps=0, stage='iql', epoch=epoch, best_eval=best_eval)
        logging.info(f'BC 完成，已保存 IQL 切换点 (best {best_eval})')
        return True
    save_checkpoint(
        state_file, mortal=mortal, aux_net=aux_net, steps=steps, stage='bc', epoch=epoch,
        best_eval=best_eval, optimizer=optimizer, scheduler=scheduler,
    )
    return False


def train_iql(mortal, dqn, aux_net, device, enable_amp):
    """IQL 精调：expectile V + Huber Q + AWR 策略提取 + 辅助任务"""
    cfg = config
    ctrl = cfg['control']
    state_file = ctrl['state_file']
    best_state_file = ctrl['best_state_file']
    version = ctrl['version']
    opt_step_every = ctrl['opt_step_every']
    save_every = ctrl['save_every']
    eval_every = cfg['eval']['eval_every']
    train_cfg = cfg['train']
    aux_w = cfg['aux']
    iql_cfg = cfg['iql']
    target = train_cfg['iql_steps']
    max_grad_norm = train_cfg['max_grad_norm']
    gamma_n = float(cfg['env']['gamma']) ** int(cfg['env']['n_step'])

    optimizer = build_optimizer([mortal, dqn, aux_net])
    scheduler = LinearWarmUpCosineAnnealingLR(
        optimizer,
        peak=train_cfg['iql_peak'], final=train_cfg['iql_final'],
        warm_up_steps=train_cfg['warm_up_steps'], max_steps=target,
    )

    from copy import deepcopy
    target_mortal = deepcopy(mortal).eval()
    target_dqn = deepcopy(dqn).eval()
    for p in target_mortal.parameters():
        p.requires_grad_(False)
    for p in target_dqn.parameters():
        p.requires_grad_(False)

    def update_target():
        with torch.no_grad():
            for tp, p in zip(target_mortal.parameters(), mortal.parameters()):
                tp.lerp_(p, 1 - iql_cfg['ema_decay'])
            for tp, p in zip(target_dqn.parameters(), dqn.parameters()):
                tp.lerp_(p, 1 - iql_cfg['ema_decay'])

    steps = 0
    epoch = 0
    best_eval = None
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        if state['stage'] == 'iql':
            mortal.load_state_dict(state['mortal'])
            aux_net.load_state_dict(state['aux_net'])
            # BC 切换点无 dqn/target，此时保持随机初始化
            if 'dqn' in state:
                dqn.load_state_dict(state['dqn'])
            if 'target_mortal' in state:
                target_mortal.load_state_dict(state['target_mortal'])
            if 'target_dqn' in state:
                target_dqn.load_state_dict(state['target_dqn'])
            if 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])
                scheduler.last_epoch = state['scheduler']['last_epoch']
            steps = state.get('steps', 0)
            epoch = state.get('epoch', 0)
            best_eval = state.get('best_eval')
            logging.info(f'resume IQL from step {steps:,}')
        elif state['stage'] == 'bc':
            # 手动切阶段：优先取 BC best，否则沿用最近 BC checkpoint
            src = best_state_file if path.exists(best_state_file) else state_file
            state = torch.load(src, weights_only=True, map_location=device)
            mortal.load_state_dict(state['mortal'])
            aux_net.load_state_dict(state['aux_net'])
            best_eval = state.get('best_eval')
            logging.info('start IQL from BC checkpoint')
    elif path.exists(best_state_file):
        # 无训练进度但存在 best：直接从 best 起步
        state = torch.load(best_state_file, weights_only=True, map_location=device)
        mortal.load_state_dict(state['mortal'])
        aux_net.load_state_dict(state['aux_net'])
        best_eval = state.get('best_eval')
        logging.info('start IQL from best checkpoint')

    ce = nn.CrossEntropyLoss()
    writer = SummaryWriter(os.path.join(ctrl['tensorboard_dir'], 'iql'))
    loader = make_loader(version)
    pb = tqdm(total=target, initial=steps, desc='IQL', unit='batch', dynamic_ncols=True)
    stats = {'v': 0., 'dqn': 0., 'policy': 0., 'next_rank': 0.,
             'shanten': 0., 'fuuro': 0., 'riichi_turn': 0.}
    all_q = []
    all_q_target = []
    n_batches = 0

    while steps < target:
        try:
            batch = next(loader)
        except StopIteration:
            loader = make_loader(version)
            epoch += 1
            logging.info(f'IQL epoch {epoch} done @ step {steps:,}')
            continue

        obs, actions, masks, player_ranks, next_obs, rewards, next_masks, is_end, shantens, fuuro_counts, riichi_turns = batch
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
        masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
        player_ranks = player_ranks.to(dtype=torch.int64, device=device, non_blocking=True)
        next_obs = next_obs.to(dtype=torch.float32, device=device, non_blocking=True)
        rewards = rewards.to(dtype=torch.float32, device=device, non_blocking=True)
        next_masks = next_masks.to(dtype=torch.bool, device=device, non_blocking=True)
        is_end = is_end.to(dtype=torch.bool, device=device, non_blocking=True)
        shantens = shantens.to(dtype=torch.int64, device=device, non_blocking=True)
        fuuro_counts = fuuro_counts.to(dtype=torch.int64, device=device, non_blocking=True)
        riichi_turns = riichi_turns.to(dtype=torch.int64, device=device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enable_amp):
            phi = mortal(obs)
            q = dqn(phi, masks)[range(obs.shape[0]), :, actions]  # (N, K)

            # Q 回归到 n 步回报 + 折扣 V，V 用 target 网络 expectile 保守估计
            with torch.no_grad():
                next_phi = target_mortal(next_obs)
                next_v = target_dqn.value(next_phi)
                q_target = rewards.unsqueeze(-1) + gamma_n * next_v * (~is_end).unsqueeze(-1)

            v = dqn.value(phi)
            td = q_target - v
            v_loss = torch.where(td > 0, iql_cfg['tau'] * td ** 2, (1 - iql_cfg['tau']) * td ** 2).mean()
            dqn_loss = F.huber_loss(q, q_target, delta=10)

            # AWR：advantage 取 Q 减 V，指数加权提取策略
            with torch.no_grad():
                exp_adv = ((q.detach() - v.detach()).mean(-1) / iql_cfg['beta']).clamp(max=iql_cfg['clip']).exp()
            policy_logits = mortal.policy_logits(phi)
            log_prob = policy_logits.log_softmax(-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            policy_loss = -(exp_adv * log_prob).mean()

            next_rank_logits, shanten_logits, fuuro_logits, riichi_turn_logits = aux_net(phi)
            next_rank_loss = ce(next_rank_logits, player_ranks)
            shanten_loss = ce(shanten_logits, shantens)
            fuuro_loss = ce(fuuro_logits, fuuro_counts)
            riichi_turn_loss = ce(riichi_turn_logits, riichi_turns)

            loss = (
                v_loss + dqn_loss + policy_loss
                + next_rank_loss * aux_w['next_rank_weight']
                + shanten_loss * aux_w['shanten_weight']
                + fuuro_loss * aux_w['fuuro_weight']
                + riichi_turn_loss * aux_w['riichi_turn_weight']
            )

        stats['v'] += v_loss.item()
        stats['dqn'] += dqn_loss.item()
        stats['policy'] += policy_loss.item()
        stats['next_rank'] += next_rank_loss.item()
        stats['shanten'] += shanten_loss.item()
        stats['fuuro'] += fuuro_loss.item()
        stats['riichi_turn'] += riichi_turn_loss.item()
        n_batches += 1
        with torch.inference_mode():
            all_q.append(q.mean(-1))
            all_q_target.append(q_target.mean(-1))

        (loss / opt_step_every).backward()
        steps += 1
        if steps % opt_step_every == 0:
            if max_grad_norm > 0:
                clip_grad_norm_(chain.from_iterable(g['params'] for g in optimizer.param_groups), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_target()
        scheduler.step()
        pb.update(1)

        if steps % save_every == 0:
            save_checkpoint(
                state_file, mortal=mortal, dqn=dqn, aux_net=aux_net,
                target_mortal=target_mortal, target_dqn=target_dqn,
                steps=steps, stage='iql', epoch=epoch, best_eval=best_eval,
                optimizer=optimizer, scheduler=scheduler,
            )
            q_cat = torch.cat(all_q).cpu().numpy()[::64]
            q_target_cat = torch.cat(all_q_target).cpu().numpy()[::64]
            all_q.clear()
            all_q_target.clear()
            writer.add_scalar('loss/v', stats['v'] / n_batches, steps)
            writer.add_scalar('loss/dqn', stats['dqn'] / n_batches, steps)
            writer.add_scalar('loss/policy', stats['policy'] / n_batches, steps)
            writer.add_scalar('loss/next_rank', stats['next_rank'] / n_batches, steps)
            writer.add_scalar('loss/shanten', stats['shanten'] / n_batches, steps)
            writer.add_scalar('loss/fuuro', stats['fuuro'] / n_batches, steps)
            writer.add_scalar('loss/riichi_turn', stats['riichi_turn'] / n_batches, steps)
            writer.add_scalar('data/epoch', epoch, steps)
            writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
            writer.add_histogram('q_predicted', q_cat, steps)
            writer.add_histogram('q_target', q_target_cat, steps)
            writer.flush()
            for k in stats:
                stats[k] = 0
            n_batches = 0
        if steps % eval_every == 0:
            best_eval = maybe_eval(mortal, dqn, device, steps, best_eval, writer, state_file, best_state_file)

    pb.close()
    save_checkpoint(
        state_file, mortal=mortal, dqn=dqn, aux_net=aux_net,
        target_mortal=target_mortal, target_dqn=target_dqn,
        steps=steps, stage='iql', epoch=epoch, best_eval=best_eval,
        optimizer=optimizer, scheduler=scheduler,
    )
    logging.info('IQL 阶段完成')


def main():
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format='%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s',
    )

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']
    version = config['control']['version']

    mortal = Brain(version=version, **config['model']).to(device)
    aux_net = AuxNet(phi_dim=config['model']['phi_dim'], dims=(4, 7, 7, 7)).to(device)
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    if enable_compile:
        mortal.compile()
        aux_net.compile()

    stage = config['train']['stage']
    state_file = config['control']['state_file']
    if stage == 'auto':
        # 以 checkpoint 实际阶段为准，无 checkpoint 时从 BC 起步
        if path.exists(state_file):
            ckpt_stage = torch.load(state_file, weights_only=True, map_location='cpu').get('stage')
            stage = ckpt_stage if ckpt_stage in ('bc', 'iql') else 'bc'
        else:
            stage = 'bc'
    if stage == 'bc':
        proceed = train_bc(mortal, aux_net, device, enable_amp)
        if proceed:
            gc.collect()
            config['train']['stage'] = 'iql'
            dqn = DQN(phi_dim=config['model']['phi_dim'], **config['dqn']).to(device)
            if enable_compile:
                dqn.compile()
            train_iql(mortal, dqn, aux_net, device, enable_amp)
    elif stage == 'iql':
        dqn = DQN(phi_dim=config['model']['phi_dim'], **config['dqn']).to(device)
        if enable_compile:
            dqn.compile()
        train_iql(mortal, dqn, aux_net, device, enable_amp)
    else:
        raise ValueError(f'未知训练阶段: {stage}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
