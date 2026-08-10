"""mortal_v6 离线两阶段训练：BC 预训练 → XQL 精调（含事件世界模型监督）

用法：python train.py
阶段与步数由 config_v6.py 的 train 段控制，BC 完成后自动切换 XQL
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
sys.path.append(MORTAL_DIR)

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import config_v6  # 注册 config 模块，必须先于 from config import config
from config import config
from dataloader import FileDatasetsIter, worker_init_fn
from lr_scheduler import LinearWarmUpCosineAnnealingLR
from model import Brain, QHead, EventModel, AuxNet
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


def save_checkpoint(state_file, *, mortal, q_head, event_model, aux_net, steps, stage, best_eval,
                    epoch=0, target_mortal=None, target_q=None, optimizer=None, scheduler=None):
    state = {
        'mortal': mortal.state_dict(),
        'q_head': q_head.state_dict(),
        'event_model': event_model.state_dict(),
        'aux_net': aux_net.state_dict(),
        'steps': steps,
        'stage': stage,
        'epoch': epoch,
        'timestamp': datetime.now().timestamp(),
        'best_eval': best_eval,
        'config': config,
    }
    for key, obj in (('target_mortal', target_mortal), ('target_q', target_q),
                     ('optimizer', optimizer), ('scheduler', scheduler)):
        if obj is not None:
            state[key] = obj.state_dict()
    torch.save(state, state_file)


def maybe_eval(mortal, q_head, event_model, device, steps, best_eval, writer, state_file, best_state_file):
    mortal.eval()
    q_head.eval()
    event_model.eval()
    result = run_eval(mortal, q_head, event_model, device)
    mortal.train()
    q_head.train()
    event_model.train()

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


def train_bc(mortal, q_head, event_model, aux_net, device, enable_amp):
    """BC 预训练：策略 CE + 事件世界模型监督 + 辅助任务"""
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
    event_w = cfg['event_loss']['weight']
    target = train_cfg['bc_steps']
    max_grad_norm = train_cfg['max_grad_norm']

    # BC 无 Q 监督，q_head 入 optimizer 会被 weight_decay 持续衰减
    optimizer = build_optimizer([mortal, event_model, aux_net])
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
            q_head.load_state_dict(state['q_head'])
            event_model.load_state_dict(state['event_model'])
            aux_net.load_state_dict(state['aux_net'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.last_epoch = state['scheduler']['last_epoch']
            steps = state['steps']
            epoch = state.get('epoch', 0)
            best_eval = state.get('best_eval')
            logging.info(f'resume BC from step {steps:,}')
        elif state['stage'] == 'xql':
            # 用户把阶段切回 BC：优先取 best，否则沿用 xql 最后模型状态（optimizer 不兼容故重建）
            src = best_state_file if path.exists(best_state_file) else state_file
            state = torch.load(src, weights_only=True, map_location=device)
            mortal.load_state_dict(state['mortal'])
            q_head.load_state_dict(state['q_head'])
            event_model.load_state_dict(state['event_model'])
            aux_net.load_state_dict(state['aux_net'])
            best_eval = state.get('best_eval')
            logging.info('restart BC from checkpoint')

    writer = SummaryWriter(os.path.join(ctrl['tensorboard_dir'], 'bc'))
    loader = make_loader(version)
    pb = tqdm(total=target, initial=steps, desc='BC', unit='batch', dynamic_ncols=True)
    stats = {'policy': 0., 'event': 0., 'next_rank': 0., 'shanten': 0., 'fuuro': 0., 'riichi_turn': 0.}
    n_batches = 0
    # resume 恰落在评估步（上次崩于评估中途）时补一次评估
    if steps and steps % eval_every == 0:
        best_eval = maybe_eval(mortal, q_head, event_model, device, steps, best_eval,
                               writer, state_file, best_state_file)

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
        event_traj = batch[12].to(dtype=torch.int64, device=device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enable_amp):
            phi = mortal(obs)
            policy_loss = F.cross_entropy(mortal.policy_logits(phi), actions)
            event_loss = F.cross_entropy(
                event_model(phi, actions).permute(0, 2, 1), event_traj, ignore_index=-1
            )
            next_rank_logits, shanten_logits, fuuro_logits, riichi_turn_logits = aux_net(phi)
            next_rank_loss = F.cross_entropy(next_rank_logits, player_ranks)
            shanten_loss = F.cross_entropy(shanten_logits, shantens)
            fuuro_loss = F.cross_entropy(fuuro_logits, fuuro_counts)
            riichi_turn_loss = F.cross_entropy(riichi_turn_logits, riichi_turns)
            loss = (
                policy_loss
                + event_loss * event_w
                + next_rank_loss * aux_w['next_rank_weight']
                + shanten_loss * aux_w['shanten_weight']
                + fuuro_loss * aux_w['fuuro_weight']
                + riichi_turn_loss * aux_w['riichi_turn_weight']
            )

        stats['policy'] += policy_loss.item()
        stats['event'] += event_loss.item()
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
                state_file, mortal=mortal, q_head=q_head, event_model=event_model, aux_net=aux_net,
                steps=steps, stage='bc', epoch=epoch, best_eval=best_eval,
                optimizer=optimizer, scheduler=scheduler,
            )
            writer.add_scalar('data/epoch', epoch, steps)
            writer.add_scalar('loss/policy', stats['policy'] / n_batches, steps)
            writer.add_scalar('loss/event', stats['event'] / n_batches, steps)
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
            best_eval = maybe_eval(mortal, q_head, event_model, device, steps, best_eval,
                                   writer, state_file, best_state_file)

    pb.close()
    if train_cfg['auto_proceed']:
        # 用 BC best 作为 XQL 起点，保存阶段切换点（无 optimizer，XQL 全新起步）
        if path.exists(best_state_file):
            state = torch.load(best_state_file, weights_only=True, map_location=device)
            mortal.load_state_dict(state['mortal'])
            q_head.load_state_dict(state['q_head'])
            event_model.load_state_dict(state['event_model'])
            aux_net.load_state_dict(state['aux_net'])
            best_eval = state.get('best_eval')
        save_checkpoint(state_file, mortal=mortal, q_head=q_head, event_model=event_model,
                        aux_net=aux_net, steps=0, stage='xql', epoch=epoch, best_eval=best_eval)
        logging.info(f'BC 完成，已保存 XQL 切换点 (best {best_eval})')
        return True
    save_checkpoint(
        state_file, mortal=mortal, q_head=q_head, event_model=event_model, aux_net=aux_net,
        steps=steps, stage='bc', epoch=epoch, best_eval=best_eval,
        optimizer=optimizer, scheduler=scheduler,
    )
    return False


def train_xql(mortal, q_head, event_model, aux_net, device, enable_amp):
    """XQL 精调：Gumbel 加权 Q 学习 + 优势策略提取 + 事件世界模型持续监督"""
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
    xql_cfg = cfg['xql']
    event_w = cfg['event_loss']['weight']
    target = train_cfg['xql_steps']
    max_grad_norm = train_cfg['max_grad_norm']
    gamma_n = float(cfg['env']['gamma']) ** int(cfg['env']['n_step'])
    tau = xql_cfg['tau']

    optimizer = build_optimizer([mortal, q_head, event_model, aux_net])
    scheduler = LinearWarmUpCosineAnnealingLR(
        optimizer,
        peak=train_cfg['xql_peak'], final=train_cfg['xql_final'],
        warm_up_steps=train_cfg['warm_up_steps'], max_steps=target,
    )

    from copy import deepcopy
    target_mortal = deepcopy(mortal).eval()
    target_q = deepcopy(q_head).eval()
    for p in target_mortal.parameters():
        p.requires_grad_(False)
    for p in target_q.parameters():
        p.requires_grad_(False)

    def update_target():
        with torch.no_grad():
            for tp, p in zip(target_mortal.parameters(), mortal.parameters()):
                tp.lerp_(p, 1 - xql_cfg['ema_decay'])
            for tp, p in zip(target_q.parameters(), q_head.parameters()):
                tp.lerp_(p, 1 - xql_cfg['ema_decay'])

    steps = 0
    epoch = 0
    best_eval = None
    state = None
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        if state['stage'] == 'xql':
            mortal.load_state_dict(state['mortal'])
            q_head.load_state_dict(state['q_head'])
            event_model.load_state_dict(state['event_model'])
            aux_net.load_state_dict(state['aux_net'])
            if 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])
                scheduler.last_epoch = state['scheduler']['last_epoch']
            steps = state.get('steps', 0)
            epoch = state.get('epoch', 0)
            best_eval = state.get('best_eval')
            logging.info(f'resume XQL from step {steps:,}')
        elif state['stage'] == 'bc':
            # 手动切阶段：优先取 BC best，否则沿用最近 BC checkpoint
            src = best_state_file if path.exists(best_state_file) else state_file
            state = torch.load(src, weights_only=True, map_location=device)
            mortal.load_state_dict(state['mortal'])
            q_head.load_state_dict(state['q_head'])
            event_model.load_state_dict(state['event_model'])
            aux_net.load_state_dict(state['aux_net'])
            best_eval = state.get('best_eval')
            logging.info('start XQL from BC checkpoint')
    elif path.exists(best_state_file):
        # 无训练进度但存在 best：直接从 best 起步
        state = torch.load(best_state_file, weights_only=True, map_location=device)
        mortal.load_state_dict(state['mortal'])
        q_head.load_state_dict(state['q_head'])
        event_model.load_state_dict(state['event_model'])
        aux_net.load_state_dict(state['aux_net'])
        best_eval = state.get('best_eval')
        logging.info('start XQL from best checkpoint')

    # EMA 起点：无保存 target（BC 切换点/best 起步）则拷贝在线权重
    if state is not None and 'target_mortal' in state and 'target_q' in state:
        target_mortal.load_state_dict(state['target_mortal'])
        target_q.load_state_dict(state['target_q'])
    else:
        target_mortal.load_state_dict(mortal.state_dict())
        target_q.load_state_dict(q_head.state_dict())

    ce = nn.CrossEntropyLoss()
    writer = SummaryWriter(os.path.join(ctrl['tensorboard_dir'], 'xql'))
    loader = make_loader(version)
    pb = tqdm(total=target, initial=steps, desc='XQL', unit='batch', dynamic_ncols=True)
    stats = {'q': 0., 'policy': 0., 'event': 0., 'next_rank': 0.,
             'shanten': 0., 'fuuro': 0., 'riichi_turn': 0.}
    all_q = []
    all_q_target = []
    n_batches = 0
    # resume 恰落在评估步（上次崩于评估中途）时补一次评估
    if steps and steps % eval_every == 0:
        best_eval = maybe_eval(mortal, q_head, event_model, device, steps, best_eval,
                               writer, state_file, best_state_file)

    while steps < target:
        try:
            batch = next(loader)
        except StopIteration:
            loader = make_loader(version)
            epoch += 1
            logging.info(f'XQL epoch {epoch} done @ step {steps:,}')
            continue

        obs, actions, masks, player_ranks, next_obs, rewards, next_masks, is_end, shantens, fuuro_counts, riichi_turns, next_actions, event_traj = batch
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
        masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
        player_ranks = player_ranks.to(dtype=torch.int64, device=device, non_blocking=True)
        next_obs = next_obs.to(dtype=torch.float32, device=device, non_blocking=True)
        rewards = rewards.to(dtype=torch.float32, device=device, non_blocking=True)
        is_end = is_end.to(dtype=torch.bool, device=device, non_blocking=True)
        shantens = shantens.to(dtype=torch.int64, device=device, non_blocking=True)
        fuuro_counts = fuuro_counts.to(dtype=torch.int64, device=device, non_blocking=True)
        riichi_turns = riichi_turns.to(dtype=torch.int64, device=device, non_blocking=True)
        next_actions = next_actions.to(dtype=torch.int64, device=device, non_blocking=True)
        event_traj = event_traj.to(dtype=torch.int64, device=device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enable_amp):
            phi = mortal(obs)
            q_all = q_head(phi)  # (N, A)，未 mask 以保留基线计算
            q_a = q_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # Gumbel 加权回归：τ>0.5 对正 TD 权重大（乐观），直接学 Q 免去 V 中间步
            with torch.no_grad():
                next_phi = target_mortal(next_obs)
                next_q_a = target_q(next_phi).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                q_target = rewards + gamma_n * next_q_a * (~is_end).float()
            td = q_target - q_a
            w = torch.where(td > 0, tau, 1 - tau)
            q_loss = (w * td ** 2).mean() * xql_cfg['q_scale']

            # AWR 式策略提取：基线取合法动作 Q 均值
            with torch.no_grad():
                q_masked = q_all.masked_fill(~masks, 0.)
                baseline = q_masked.sum(-1) / masks.sum(-1).clamp_min(1)
                exp_adv = ((q_a - baseline) / xql_cfg['beta']).clamp(max=xql_cfg['clip']).exp()
            log_prob = mortal.policy_logits(phi).log_softmax(-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            policy_loss = -(exp_adv * log_prob).mean()

            event_loss = F.cross_entropy(
                event_model(phi, actions).permute(0, 2, 1), event_traj, ignore_index=-1
            )
            next_rank_logits, shanten_logits, fuuro_logits, riichi_turn_logits = aux_net(phi)
            next_rank_loss = ce(next_rank_logits, player_ranks)
            shanten_loss = ce(shanten_logits, shantens)
            fuuro_loss = ce(fuuro_logits, fuuro_counts)
            riichi_turn_loss = ce(riichi_turn_logits, riichi_turns)

            loss = (
                q_loss + policy_loss
                + event_loss * event_w
                + next_rank_loss * aux_w['next_rank_weight']
                + shanten_loss * aux_w['shanten_weight']
                + fuuro_loss * aux_w['fuuro_weight']
                + riichi_turn_loss * aux_w['riichi_turn_weight']
            )

        stats['q'] += q_loss.item()
        stats['policy'] += policy_loss.item()
        stats['event'] += event_loss.item()
        stats['next_rank'] += next_rank_loss.item()
        stats['shanten'] += shanten_loss.item()
        stats['fuuro'] += fuuro_loss.item()
        stats['riichi_turn'] += riichi_turn_loss.item()
        n_batches += 1
        with torch.inference_mode():
            all_q.append(q_a.detach().float().cpu())
            all_q_target.append(q_target.detach().float().cpu())

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
                state_file, mortal=mortal, q_head=q_head, event_model=event_model, aux_net=aux_net,
                target_mortal=target_mortal, target_q=target_q,
                steps=steps, stage='xql', epoch=epoch, best_eval=best_eval,
                optimizer=optimizer, scheduler=scheduler,
            )
            q_cat = torch.cat(all_q).numpy()[::64]
            q_target_cat = torch.cat(all_q_target).numpy()[::64]
            all_q.clear()
            all_q_target.clear()
            writer.add_scalar('loss/q', stats['q'] / n_batches, steps)
            writer.add_scalar('loss/policy', stats['policy'] / n_batches, steps)
            writer.add_scalar('loss/event', stats['event'] / n_batches, steps)
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
            best_eval = maybe_eval(mortal, q_head, event_model, device, steps, best_eval,
                                   writer, state_file, best_state_file)

    pb.close()
    save_checkpoint(
        state_file, mortal=mortal, q_head=q_head, event_model=event_model, aux_net=aux_net,
        target_mortal=target_mortal, target_q=target_q,
        steps=steps, stage='xql', epoch=epoch, best_eval=best_eval,
        optimizer=optimizer, scheduler=scheduler,
    )
    logging.info('XQL 阶段完成')


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
    q_head = QHead(phi_dim=config['model']['phi_dim'], **config['q_head']).to(device)
    event_model = EventModel(phi_dim=config['model']['phi_dim'], **config['event']).to(device)
    aux_net = AuxNet(phi_dim=config['model']['phi_dim'], dims=(4, 7, 7, 7)).to(device)
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'q_head params: {parameter_count(q_head):,}')
    logging.info(f'event_model params: {parameter_count(event_model):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    if enable_compile:
        mortal.compile()
        q_head.compile()
        event_model.compile()
        aux_net.compile()

    stage = config['train']['stage']
    state_file = config['control']['state_file']
    if stage == 'auto':
        # 以 checkpoint 实际阶段为准，无 checkpoint 时从 BC 起步
        if path.exists(state_file):
            ckpt_stage = torch.load(state_file, weights_only=True, map_location='cpu').get('stage')
            stage = ckpt_stage if ckpt_stage in ('bc', 'xql') else 'bc'
        else:
            stage = 'bc'
    if stage == 'bc':
        proceed = train_bc(mortal, q_head, event_model, aux_net, device, enable_amp)
        if proceed:
            gc.collect()
            config['train']['stage'] = 'xql'
            train_xql(mortal, q_head, event_model, aux_net, device, enable_amp)
    elif stage == 'xql':
        train_xql(mortal, q_head, event_model, aux_net, device, enable_amp)
    else:
        raise ValueError(f'未知训练阶段: {stage}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
