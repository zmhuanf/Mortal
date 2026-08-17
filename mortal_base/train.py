"""mortal_base 训练：离线 IQL + AWR 策略提取 + 辅助任务

loss 配方与 mortal/train.py 离线分支逐行一致：
v_loss(expectile) + dqn_loss(Huber) + policy_loss(exp-adv CE) + aux CE 加权
lr 调度复刻 LinearWarmUpCosineAnnealingLR（peak=final 时等效恒定）
"""

import os
import gzip
import json
import math
import random
import logging
from copy import deepcopy
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

import config_base  # 注册 config 模块，必须先于 from config import config
from config import config
from model import Brain, DQN, AuxNet
from dataset import FileDatasetsIter, worker_init_fn
from evaluate import make_engine, run_eval

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def read_player_names():
    names = set()
    for filename in config['dataset']['player_names_files']:
        with open(filename, encoding='utf-8') as f:
            names.update(line.strip() for line in f if line.strip())
    return names


def build_file_list(player_names_set):
    file_index = config['dataset']['file_index']
    if path.exists(file_index):
        file_list = torch.load(file_index, weights_only=True)['file_list']
    else:
        logging.info('building file index...')
        file_list = []
        for pat in config['dataset']['globs']:
            file_list.extend(glob(pat, recursive=True))
        # 过滤仅发生在首次构建 index 时，之后直接读缓存
        if len(player_names_set) > 0:
            filtered = []
            for filename in tqdm(file_list, unit='file'):
                with gzip.open(filename, 'rt') as f:
                    start = json.loads(next(f))
                    if not set(start['names']).isdisjoint(player_names_set):
                        filtered.append(filename)
            file_list = filtered
        file_list.sort(reverse=True)
        os.makedirs(path.dirname(file_index), exist_ok=True)
        torch.save({'file_list': file_list}, file_index)
    return file_list


def main():
    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    # Blackwell 用 bf16（无 fp16 溢出风险），其余回退 fp16
    amp_dtype = torch.bfloat16 if config['control'].get('amp_dtype', 'float16') == 'bfloat16' else torch.float16
    version = config['control']['version']

    mortal = Brain(version=version, **config['model']).to(device)
    dqn = DQN(version=version, num_heads=config['dqn']['num_heads']).to(device)
    aux_net = AuxNet().to(device)
    target_mortal = deepcopy(mortal).eval()
    target_dqn = deepcopy(dqn).eval()
    for p in chain(target_mortal.parameters(), target_dqn.parameters()):
        p.requires_grad_(False)
    logging.info(f'mortal params: {sum(p.numel() for p in mortal.parameters()):,}')
    logging.info(f'dqn params: {sum(p.numel() for p in dqn.parameters()):,}')
    logging.info(f'aux params: {sum(p.numel() for p in aux_net.parameters()):,}')

    if config['control']['enable_compile']:
        for m in (mortal, dqn, aux_net):
            m.compile()

    # weight decay 仅作用于 Conv1d/Linear 的 weight，参数按名字排序保证确定性
    decay_params, no_decay_params = [], []
    for model in (mortal, dqn, aux_net):
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': config['optim']['weight_decay']},
        {'params': no_decay_params},
    ], lr=1, weight_decay=0, betas=config['optim']['betas'], eps=config['optim']['eps'])
    scaler = GradScaler(device.type, enabled=enable_amp and amp_dtype == torch.float16)

    steps = 0
    data_offset = 0
    best_perf = {'avg_rank': 4., 'avg_pt': -135., 'pool_version': -1}
    state_file = config['control']['state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=False, map_location=device)
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if 'target_mortal' in state:
            target_mortal.load_state_dict(state['target_mortal'])
            target_dqn.load_state_dict(state['target_dqn'])
        else:
            target_mortal.load_state_dict(state['mortal'])
            target_dqn.load_state_dict(state['current_dqn'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        steps = state['steps']
        data_offset = state.get('data_offset', 0)
        best_perf = state.get('best_perf', best_perf)
        logging.info(f'resumed from step {steps} (data offset {data_offset})')

    optimizer.zero_grad(set_to_none=True)
    writer = SummaryWriter(config['control']['tensorboard_dir'])

    player_names = list(read_player_names())
    file_list = build_file_list(set(player_names))
    logging.info(f'offline files: {len(file_list):,}')
    ds_cfg = config['dataset']
    post_training = config['train'].get('post_training', False)
    max_steps = config['train']['max_steps']
    data = FileDatasetsIter(
        version=version,
        file_list=file_list,
        pts=config['env']['pts'],
        file_batch_size=ds_cfg['file_batch_size'],
        reserve_ratio=ds_cfg['reserve_ratio'],
        player_names=player_names,
        num_epochs=ds_cfg['num_epochs'],
        enable_augmentation=ds_cfg['enable_augmentation'],
        augmented_first=ds_cfg['augmented_first'],
        resume_files=data_offset,
        shuffle_seed=ds_cfg.get('shuffle_seed', 42),
        random_files=post_training,
    )
    loader_kwargs = {
        'batch_size': config['control']['batch_size'],
        'drop_last': False,
        'num_workers': ds_cfg['num_workers'],
        'pin_memory': True,
        'worker_init_fn': worker_init_fn,
    }
    if ds_cfg['num_workers'] > 0:
        # 队列缓冲 24 transition，吸收解析方差避免 GPU 饥饿（v7 工程参数）
        loader_kwargs['prefetch_factor'] = ds_cfg.get('prefetch_factor', 12)
        loader_kwargs['persistent_workers'] = True
    loader = iter(DataLoader(data, **loader_kwargs))

    B = config['control']['batch_size']
    gamma_n = config['env']['gamma'] ** config['env']['n_step']
    iql = config['iql']
    aux_w = config['aux']
    eval_every = config['train']['eval_every']
    eval_games = config['train']['eval_games']
    best_state_file = config['control']['best_state_file']
    eval_log_dir = config['eval']['log_dir']
    # 复刻 LinearWarmUpCosineAnnealingLR：LambdaLR 因子作用于 base lr=1
    sched = config['optim']['scheduler']
    init_lr, peak_lr = sched['init'], sched['peak']
    warm_up, max_steps_lr = sched['warm_up_steps'], sched['max_steps']
    final_lr = sched['final']
    ema_decay = iql['ema_decay']
    ce = nn.CrossEntropyLoss()

    def current_lr():
        """LambdaLR 等效因子：last_epoch = 已完成步数"""
        s = steps
        if s < warm_up:
            return init_lr + (peak_lr - init_lr) / warm_up * s
        if s < max_steps_lr:
            cos_steps = s - warm_up
            cos_max = max_steps_lr - warm_up
            return final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(cos_steps / cos_max * math.pi))
        return final_lr

    def update_target():
        # target = ema_decay * target + (1 - ema_decay) * online
        for tp, p in zip(target_mortal.parameters(), mortal.parameters()):
            tp.lerp_(p, 1 - ema_decay)
        for tp, p in zip(target_dqn.parameters(), dqn.parameters()):
            tp.lerp_(p, 1 - ema_decay)

    def train_batch(obs, actions, masks, player_ranks, next_obs, n_step_rewards, next_masks, is_episode_end, shantens, fuuro_counts, riichi_turns):
        nonlocal steps
        obs = obs.to(device=device, non_blocking=True)
        actions = actions.to(device=device, non_blocking=True)
        masks = masks.to(device=device, non_blocking=True)
        player_ranks = player_ranks.to(device=device, non_blocking=True)
        next_obs = next_obs.to(device=device, non_blocking=True)
        n_step_rewards = n_step_rewards.to(device=device, non_blocking=True)
        next_masks = next_masks.to(device=device, non_blocking=True)
        is_episode_end = is_episode_end.to(device=device, non_blocking=True)
        shantens = shantens.to(device=device, non_blocking=True)
        fuuro_counts = fuuro_counts.to(device=device, non_blocking=True)
        riichi_turns = riichi_turns.to(device=device, non_blocking=True)

        # scheduler.step() 先于 optimizer.step() 生效（LambdaLR 语义）
        lr = current_lr()
        for g in optimizer.param_groups:
            g['lr'] = lr

        with torch.autocast(device.type, dtype=amp_dtype, enabled=enable_amp):
            phi = mortal(obs)
            q_out = dqn(phi, masks)  # (N, K, A)
            q = q_out[range(B), :, actions]  # (N, K)

            # IQL：expectile V 回归 + Q 回归，target 用 EMA 网络估值
            with torch.no_grad():
                next_phi = target_mortal(next_obs)
                next_v = target_dqn.value(next_phi)  # (N, K)
                q_target = n_step_rewards.unsqueeze(-1) + gamma_n * next_v * (~is_episode_end).unsqueeze(-1)
            v = dqn.value(phi)  # (N, K)
            td = q_target - v
            v_loss = torch.where(td > 0, iql['tau'] * td ** 2, (1 - iql['tau']) * td ** 2).mean()
            dqn_loss = F.huber_loss(q, q_target, delta=10)

            # AWR 策略提取：优势指数加权 CE
            with torch.no_grad():
                adv = q_target - v
                exp_adv = (adv.mean(-1) / iql['beta']).clamp(max=iql['clip']).exp()
            log_prob = mortal.policy_logits(phi).log_softmax(-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            policy_loss = -(exp_adv * log_prob).mean()

            next_rank_logits, shanten_logits, fuuro_logits, riichi_turn_logits = aux_net(phi)
            next_rank_loss = ce(next_rank_logits, player_ranks)
            shanten_loss = ce(shanten_logits, shantens)
            fuuro_loss = ce(fuuro_logits, fuuro_counts)
            riichi_turn_loss = ce(riichi_turn_logits, riichi_turns)

            loss = (
                v_loss + policy_loss + dqn_loss
                + next_rank_loss * aux_w['next_rank_weight']
                + shanten_loss * aux_w['shanten_weight']
                + fuuro_loss * aux_w['fuuro_weight']
                + riichi_turn_loss * aux_w['riichi_turn_weight']
            )

        scaler.scale(loss).backward()
        steps += 1
        if config['optim']['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
            clip_grad_norm_(params, config['optim']['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update_target()
        return loss, v_loss, policy_loss, dqn_loss, next_rank_loss, shanten_loss, fuuro_loss, riichi_turn_loss

    import math
    import random
    loss_sum = {k: 0.0 for k in ('v', 'policy', 'dqn', 'next_rank', 'shanten', 'fuuro', 'riichi_turn')}
    n_batch = 0
    pb = tqdm(total=None if post_training else max_steps, initial=steps, desc='TRAIN', unit='step')

    def record(losses):
        """统计 + 周期保存，循环与尾部补整共用"""
        nonlocal n_batch, loss_sum
        n_batch += 1
        for k, v in zip(loss_sum, losses[1:]):
            loss_sum[k] += v.item()
        pb.update(1)
        if steps % config['control']['save_every'] == 0:
            for k in loss_sum:
                writer.add_scalar(f'loss/{k}', loss_sum[k] / n_batch, steps)
            writer.add_scalar('hparam/lr', optimizer.param_groups[0]['lr'], steps)
            writer.flush()
            loss_sum = {k: 0.0 for k in loss_sum}
            n_batch = 0

            state = {
                'mortal': mortal.state_dict(),
                'current_dqn': dqn.state_dict(),
                'aux_net': aux_net.state_dict(),
                'target_mortal': target_mortal.state_dict(),
                'target_dqn': target_dqn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': {'last_epoch': steps},
                'scaler': scaler.state_dict(),
                'steps': steps,
                'data_offset': data.cursor.value,
                'timestamp': datetime.now().timestamp(),
                'best_perf': best_perf,
                'pool_version': best_perf['pool_version'],
                'config': config,
            }
            torch.save(state, state_file)
            logging.info(f'step {steps:,} | loss {losses[0].item():.4f}')

    def maybe_evaluate():
        """周期 1v3 评估 vs baseline_v1，超越历史最优则存 best.pth"""
        nonlocal best_perf
        mortal.eval()
        dqn.eval()
        engine = make_engine(mortal, dqn, device, version, enable_amp=True)
        result = run_eval(engine, device, games=eval_games, log_dir=eval_log_dir)
        mortal.train()
        dqn.train()
        writer.add_scalar('eval/avg_rank', result['avg_rank'], steps)
        writer.add_scalar('eval/avg_pt', result['avg_pt'], steps)
        writer.flush()
        logging.info(f'eval @{steps}: avg_rank {result["avg_rank"]:.4f} avg_pt {result["avg_pt"]:.4f}')
        if (result['avg_rank'] < best_perf['avg_rank']
                or (result['avg_rank'] == best_perf['avg_rank'] and result['avg_pt'] > best_perf['avg_pt'])):
            best_perf = {'avg_rank': result['avg_rank'], 'avg_pt': result['avg_pt'],
                         'pool_version': best_perf['pool_version']}
            best_state = {
                'mortal': mortal.state_dict(),
                'current_dqn': dqn.state_dict(),
                'aux_net': aux_net.state_dict(),
                'config': config,
                'steps': steps,
                'best_perf': best_perf,
                'timestamp': datetime.now().timestamp(),
            }
            torch.save(best_state, best_state_file)
            logging.info(f'new best @{steps}: avg_rank {result["avg_rank"]:.4f} avg_pt {result["avg_pt"]:.4f}')

    # 恢复后 best 仍为初始值（未验证）时，先按当前配方补一次评估
    if eval_every > 0 and steps > 0 and best_perf['avg_rank'] >= 4.0:
        logging.info(f'resume: best_perf unverified at step {steps}, evaluating now')
        maybe_evaluate()

    # 尾部攒批补整：不足 batch 的块先缓存，凑满后切整批训练（同 baseline）
    remaining = None
    for batch_tensors in loader:
        if batch_tensors[0].shape[0] != B:
            if remaining is None:
                # 张量数量随 oracle 开关变化，按首个非整批动态初始化
                remaining = [[] for _ in batch_tensors]
            for lst, t in zip(remaining, batch_tensors):
                lst.append(t)
            continue
        losses = train_batch(*batch_tensors)
        record(losses)
        if eval_every > 0 and steps % eval_every == 0:
            maybe_evaluate()
        if not post_training and steps >= max_steps:
            break

    # 攒批尾部补整（剩余不足整批的丢弃）
    if remaining and any(remaining[0]):
        tail = [torch.cat(lst, dim=0) for lst in remaining]
        tail_bs = tail[0].shape[0]
        for start in range(0, tail_bs, B):
            if start + B > tail_bs:
                break
            losses = train_batch(*[c[start:start + B] for c in tail])
            record(losses)
            if not post_training and steps >= max_steps:
                break
    pb.close()
    logging.info(f'training done at step {steps}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
