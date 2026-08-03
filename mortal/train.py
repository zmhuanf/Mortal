def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import numpy as np
    import torch
    from copy import deepcopy
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    import torch.nn.functional as F
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm, get_pool
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from replay_buffer import PrioritizedReplayBuffer
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']

    online = config['control']['online']
    is_baseline = config['control'].get('is_baseline', False)
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    next_rank_weight = config['aux']['next_rank_weight']
    shanten_weight = config['aux'].get('shanten_weight', 0)
    fuuro_weight = config['aux'].get('fuuro_weight', 0)
    riichi_turn_weight = config['aux'].get('riichi_turn_weight', 0)
    online_human_ratio = config['dataset'].get('online_human_ratio', 0.0)
    per_cfg = config.get('per', {})
    per_alpha = per_cfg.get('alpha', 0.0)
    per_beta = per_cfg.get('beta', 0.0)
    per_epsilon = per_cfg.get('epsilon', 1e-6)
    per_capacity = per_cfg.get('capacity', 200000)
    per_min_size = per_cfg.get('min_size', 8192)
    per_beta_end = per_cfg.get('beta_end', 1.0)
    per_beta_anneal_steps = per_cfg.get('beta_anneal_steps', 100000)
    use_per = online and per_alpha > 0
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    gamma = config['env']['gamma']
    n_step = config['env'].get('n_step', 3)
    iql_tau = config['iql']['tau']
    iql_beta = config['iql']['beta']
    iql_clip = config['iql']['clip']
    ema_decay = config['iql']['ema_decay']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    dqn_num_heads = config.get('dqn', {}).get('num_heads', 1)

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version, num_heads=dqn_num_heads).to(device)
    aux_net = AuxNet((4, 7, 7, 7)).to(device)
    all_models = (mortal, dqn, aux_net)

    target_mortal = deepcopy(mortal).eval()
    target_dqn = deepcopy(dqn).eval()
    for p in target_mortal.parameters():
        p.requires_grad_(False)
    for p in target_dqn.parameters():
        p.requires_grad_(False)

    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = None if is_baseline else TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
        'pool_version': -1,
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if 'target_mortal' in state:
            target_mortal.load_state_dict(state['target_mortal'])
            target_dqn.load_state_dict(state['target_dqn'])
        else:
            target_mortal.load_state_dict(state['mortal'])
            target_dqn.load_state_dict(state['current_dqn'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        if 'pool_version' not in best_perf:
            best_perf['pool_version'] = state.get('pool_version', -1)
        steps = state['steps']
    last_pool_version = state.get('pool_version', best_perf.get('pool_version', -1)) if path.exists(state_file) else -1

    optimizer.zero_grad(set_to_none=True)
    ce = nn.CrossEntropyLoss()

    per_buffer = None
    if use_per:
        per_buffer = PrioritizedReplayBuffer(
            capacity=per_capacity,
            alpha=per_alpha,
            beta=per_beta,
            beta_end=per_beta_end,
            beta_anneal_steps=per_beta_anneal_steps,
            eps=per_epsilon,
        )
        logging.info(f'PER enabled: capacity={per_capacity}, alpha={per_alpha}, beta={per_beta}->{per_beta_end}, min_size={per_min_size}')

    def update_target():
        with torch.no_grad():
            for tp, p in zip(target_mortal.parameters(), mortal.parameters()):
                tp.lerp_(p, 1 - ema_decay)
            for tp, p in zip(target_dqn.parameters(), dqn.parameters()):
                tp.lerp_(p, 1 - ema_decay)

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'v_loss': 0,
        'policy_loss': 0,
        'dqn_loss': 0,
        'next_rank_loss': 0,
        'shanten_loss': 0,
        'fuuro_loss': 0,
        'riichi_turn_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        human_file_list = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            selfplay_file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))

            # 混合人类牌谱
            if online_human_ratio > 0:
                human_file_index = config['dataset']['file_index']
                if path.exists(human_file_index):
                    index = torch.load(human_file_index, weights_only=True)
                    human_file_list = index['file_list']
                else:
                    human_file_list = []
                    for pat in config['dataset']['globs']:
                        human_file_list.extend(glob(pat, recursive=True))
                    human_file_list.sort(reverse=True)
                    torch.save({'file_list': human_file_list}, human_file_index)

                human_batch_size = max(1, int(batch_size * online_human_ratio))
                selfplay_batch_size = max(1, batch_size - human_batch_size)
                logging.info(f'mixed training: {human_batch_size} human + {selfplay_batch_size} self-play per batch')
            else:
                human_batch_size = 0
                selfplay_batch_size = batch_size

            file_list = selfplay_file_list
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)

        # 混合数据源：self-play + 人类牌谱
        if online and online_human_ratio > 0 and len(human_file_list) > 0:
            selfplay_data = FileDatasetsIter(
                version = version,
                file_list = selfplay_file_list,
                pts = pts,
                file_batch_size = file_batch_size,
                reserve_ratio = reserve_ratio,
                player_names = player_names,
                num_epochs = num_epochs,
                enable_augmentation = enable_augmentation,
                augmented_first = augmented_first,
            )
            human_data = FileDatasetsIter(
                version = version,
                file_list = human_file_list,
                pts = pts,
                file_batch_size = file_batch_size,
                reserve_ratio = reserve_ratio,
                num_epochs = num_epochs,
                enable_augmentation = enable_augmentation,
                augmented_first = augmented_first,
            )
            selfplay_loader = iter(DataLoader(
                dataset = selfplay_data,
                batch_size = selfplay_batch_size,
                drop_last = True,
                num_workers = num_workers,
                pin_memory = True,
                worker_init_fn = worker_init_fn,
            ))
            human_loader = iter(DataLoader(
                dataset = human_data,
                batch_size = human_batch_size,
                drop_last = True,
                num_workers = num_workers,
                pin_memory = True,
                worker_init_fn = worker_init_fn,
            ))
            # 交错迭代两个 loader
            data_loader = (tuple(torch.cat([sp, hu], dim=0) for sp, hu in zip(sp_batch, hu_batch))
                           for sp_batch, hu_batch in zip(selfplay_loader, human_loader))
        else:
            file_data = FileDatasetsIter(
                version = version,
                file_list = file_list,
                pts = pts,
                file_batch_size = file_batch_size,
                reserve_ratio = reserve_ratio,
                player_names = player_names,
                num_epochs = num_epochs,
                enable_augmentation = enable_augmentation,
                augmented_first = augmented_first,
            )
            data_loader = iter(DataLoader(
                dataset = file_data,
                batch_size = batch_size,
                drop_last = False,
                num_workers = num_workers,
                pin_memory = True,
                worker_init_fn = worker_init_fn,
            ))

        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_player_ranks = []
        remaining_next_obs = []
        remaining_n_step_rewards = []
        remaining_next_masks = []
        remaining_is_episode_end = []
        remaining_shantens = []
        remaining_fuuro_counts = []
        remaining_riichi_turns = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, player_ranks, next_obs, n_step_rewards, next_masks, is_episode_end, shantens, fuuro_counts, riichi_turns, buffer_indices=None, is_weights=None):
            nonlocal steps
            nonlocal idx
            nonlocal pb
            nonlocal best_perf
            nonlocal last_pool_version

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            next_obs = next_obs.to(dtype=torch.float32, device=device)
            n_step_rewards = n_step_rewards.to(dtype=torch.float32, device=device)
            next_masks = next_masks.to(dtype=torch.bool, device=device)
            is_episode_end = is_episode_end.to(dtype=torch.bool, device=device)
            shantens = shantens.to(dtype=torch.int64, device=device)
            fuuro_counts = fuuro_counts.to(dtype=torch.int64, device=device)
            riichi_turns = riichi_turns.to(dtype=torch.int64, device=device)
            assert masks[range(batch_size), actions].all()

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                q_out = dqn(phi, masks)  # (N, K, A)
                q = q_out[range(batch_size), :, actions]  # (N, K)

                if online:
                    # 在线 self-play：Double DQN，online net 选动作 target net 估值
                    with torch.no_grad():
                        next_phi_online = mortal(next_obs)
                        next_a = dqn(next_phi_online, next_masks).argmax(-1, keepdim=True)  # (N, K, 1)
                        next_phi = target_mortal(next_obs)
                        next_q_target = target_dqn(next_phi, next_masks)  # (N, K, A)
                        next_q_gathered = next_q_target.gather(-1, next_a).squeeze(-1)  # (N, K)
                        q_target = n_step_rewards.unsqueeze(-1) + gamma ** n_step * next_q_gathered * (~is_episode_end).unsqueeze(-1)

                    td_error = (q - q_target).abs().mean(-1)  # (N,)
                    per_batch_loss = 0.5 * (q - q_target).pow(2).mean(-1)  # (N,)
                    if is_weights is not None:
                        dqn_loss = (is_weights.to(device) * per_batch_loss).mean()
                    else:
                        dqn_loss = per_batch_loss.mean()
                    v_loss = torch.tensor(0., device=device)
                else:
                    # 离线：IQL expectile regression + Q 回归
                    with torch.no_grad():
                        next_phi = target_mortal(next_obs)
                        next_v = target_dqn.value(next_phi)  # (N, K)
                        q_target = n_step_rewards.unsqueeze(-1) + gamma ** n_step * next_v * (~is_episode_end).unsqueeze(-1)

                    v = dqn.value(phi)  # (N, K)
                    td = q_target - v
                    v_loss = torch.where(td > 0, iql_tau * td**2, (1 - iql_tau) * td**2).mean()
                    dqn_loss = F.huber_loss(q, q_target, delta=10)

                # AWR 策略学习，在线/离线统一
                with torch.no_grad():
                    if online:
                        adv = q_target - q.detach()
                    else:
                        adv = q_target - v
                    exp_adv = (adv.mean(-1) / iql_beta).clamp(max=iql_clip).exp()
                log_prob = mortal.policy_logits(phi).log_softmax(-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                policy_loss = -(exp_adv * log_prob).mean()

                next_rank_logits, shanten_logits, fuuro_logits, riichi_turn_logits = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)
                shanten_loss = ce(shanten_logits, shantens)
                fuuro_loss = ce(fuuro_logits, fuuro_counts)
                riichi_turn_loss = ce(riichi_turn_logits, riichi_turns)

                loss = (
                    v_loss + policy_loss + dqn_loss + next_rank_loss * next_rank_weight
                    + shanten_loss * shanten_weight
                    + fuuro_loss * fuuro_weight
                    + riichi_turn_loss * riichi_turn_weight
                )
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['v_loss'] += v_loss
                stats['policy_loss'] += policy_loss
                stats['dqn_loss'] += dqn_loss
                stats['next_rank_loss'] += next_rank_loss
                stats['shanten_loss'] += shanten_loss
                stats['fuuro_loss'] += fuuro_loss
                stats['riichi_turn_loss'] += riichi_turn_loss
                all_q[idx] = q.mean(-1)
                all_q_target[idx] = q_target.mean(-1)

                # PER 训练后用新 TD error 更新优先级
                if per_buffer is not None and buffer_indices is not None:
                    td_cpu = td_error.cpu().numpy()
                    per_buffer.update_priorities(buffer_indices, td_cpu)

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_target()
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # downsample to reduce tensorboard event size
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/v_loss', stats['v_loss'] / save_every, steps)
                writer.add_scalar('loss/policy_loss', stats['policy_loss'] / save_every, steps)
                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                writer.add_scalar('loss/shanten_loss', stats['shanten_loss'] / save_every, steps)
                writer.add_scalar('loss/fuuro_loss', stats['fuuro_loss'] / save_every, steps)
                writer.add_scalar('loss/riichi_turn_loss', stats['riichi_turn_loss'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    'target_mortal': target_mortal.state_dict(),
                    'target_dqn': target_dqn.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'pool_version': last_pool_version,
                    'config': config,
                }
                torch.save(state, state_file)

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if not is_baseline and steps % test_every == 0:
                    pool = get_pool()
                    pool_changed = pool['version'] != last_pool_version
                    last_pool_version = pool['version']
                    state['pool_version'] = last_pool_version
                    if pool_changed:
                        logging.info(f'opponent pool upgraded to v{pool["version"]}')

                    results = test_player.test_all(pool['opponents'], test_games, mortal, dqn, device)
                    mortal.train()
                    dqn.train()

                    totals = [0, 0, 0, 0]
                    for _, rankings, _ in results:
                        for k in range(4):
                            totals[k] += rankings[k]
                    total = sum(totals)
                    avg_rank = sum((i + 1) * c for i, c in enumerate(totals)) / total
                    avg_pt = sum(p * c for p, c in zip([90, 45, 0, -135], totals)) / total

                    prev_avg_rank = best_perf['avg_rank']
                    prev_avg_pt = best_perf['avg_pt']
                    # 池子升级后旧基准作废，本次评估建立新基准
                    # 首评除非跨池全面超越旧纪录，否则不覆盖 best.pth
                    better = avg_pt >= prev_avg_pt and avg_rank <= prev_avg_rank
                    best_perf = {
                        'avg_rank': avg_rank,
                        'avg_pt': avg_pt,
                        'pool_version': last_pool_version,
                    }
                    if pool_changed:
                        logging.info(
                            f'pool upgraded, best baseline: {prev_avg_pt:.4}pt/{prev_avg_rank:.4} '
                            f'-> {avg_pt:.4}pt/{avg_rank:.4} (best.pth kept)'
                        )
                    elif better:
                        logging.info(
                            f'new best: {prev_avg_pt:.4}pt/{prev_avg_rank:.4} '
                            f'-> {avg_pt:.4}pt/{avg_rank:.4}'
                        )

                    logging.info(f'avg rank: {avg_rank:.6} (pool v{pool["version"]}, {len(results)} opponents)')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    for op, rankings, _ in results:
                        op_total = sum(rankings)
                        op_avg_rank = sum((i + 1) * c for i, c in enumerate(rankings)) / max(1, op_total)
                        logging.info(f'  vs {op["name"]}: {rankings} ({op_avg_rank:.6})')
                        writer.add_scalar(f'test_play/pool/{op["name"]}/avg_ranking', op_avg_rank, steps)

                    stat = results[-1][2]
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    state['best_perf'] = best_perf
                    torch.save(state, state_file)
                    if better:
                        logging.info(
                            'a new record has been made, '
                            f'pt: {prev_avg_pt:.4} -> {avg_pt:.4}, '
                            f'rank: {prev_avg_rank:.4} -> {avg_rank:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        # 辅助函数：把 DataLoader 产出的 tensor batch 拆为 transition 列表加入 PER buffer
        def feed_per_buffer(*tensors):
            """把 tensor batch 拆为逐条 transition 存入 PER buffer"""
            bs = tensors[0].shape[0]
            samples = []
            for i in range(bs):
                samples.append(tuple(t[i].numpy() for t in tensors))
            per_buffer.add(samples)

        def sample_from_per():
            """从 PER buffer 采样并转为 tensor batch"""
            stacked, indices, weights = per_buffer.sample(batch_size)
            tensors = tuple(torch.from_numpy(arr) for arr in stacked)
            return tensors, indices, torch.from_numpy(weights)

        if use_per:
            # PER 模式：数据先入 buffer，达预热容量后开始训练
            warmup_logged = False
            for batch_tensors in data_loader:
                obs, actions, masks, player_ranks, next_obs, n_step_rewards, next_masks, is_episode_end, shantens, fuuro_counts, riichi_turns = batch_tensors
                feed_per_buffer(obs, actions, masks, player_ranks, next_obs, n_step_rewards, next_masks, is_episode_end, shantens, fuuro_counts, riichi_turns)

                if len(per_buffer) < per_min_size:
                    if not warmup_logged:
                        logging.info(f'PER warmup: {len(per_buffer)}/{per_min_size}')
                        warmup_logged = True
                    continue

                if warmup_logged:
                    logging.info(f'PER warmup done, start training with {len(per_buffer)} transitions')
                    warmup_logged = False

                sampled_tensors, buf_indices, is_weights = sample_from_per()
                train_batch(*sampled_tensors, buffer_indices=buf_indices, is_weights=is_weights)
        else:
            # 非 PER 模式：DataLoader 直接喂入训练
            for obs, actions, masks, player_ranks, next_obs, n_step_rewards, next_masks, is_episode_end, shantens, fuuro_counts, riichi_turns in data_loader:
                bs = obs.shape[0]
                if bs != batch_size:
                    remaining_obs.append(obs)
                    remaining_actions.append(actions)
                    remaining_masks.append(masks)
                    remaining_player_ranks.append(player_ranks)
                    remaining_next_obs.append(next_obs)
                    remaining_n_step_rewards.append(n_step_rewards)
                    remaining_next_masks.append(next_masks)
                    remaining_is_episode_end.append(is_episode_end)
                    remaining_shantens.append(shantens)
                    remaining_fuuro_counts.append(fuuro_counts)
                    remaining_riichi_turns.append(riichi_turns)
                    remaining_bs += bs
                    continue
                train_batch(obs, actions, masks, player_ranks, next_obs, n_step_rewards, next_masks, is_episode_end, shantens, fuuro_counts, riichi_turns)

            remaining_batches = remaining_bs // batch_size
            if remaining_batches > 0:
                obs = torch.cat(remaining_obs, dim=0)
                actions = torch.cat(remaining_actions, dim=0)
                masks = torch.cat(remaining_masks, dim=0)
                player_ranks = torch.cat(remaining_player_ranks, dim=0)
                next_obs = torch.cat(remaining_next_obs, dim=0)
                n_step_rewards = torch.cat(remaining_n_step_rewards, dim=0)
                next_masks = torch.cat(remaining_next_masks, dim=0)
                is_episode_end = torch.cat(remaining_is_episode_end, dim=0)
                shantens = torch.cat(remaining_shantens, dim=0)
                fuuro_counts = torch.cat(remaining_fuuro_counts, dim=0)
                riichi_turns = torch.cat(remaining_riichi_turns, dim=0)
                start = 0
                end = batch_size
                while end <= remaining_bs:
                    train_batch(
                        obs[start:end],
                        actions[start:end],
                        masks[start:end],
                        player_ranks[start:end],
                        next_obs[start:end],
                        n_step_rewards[start:end],
                        next_masks[start:end],
                        is_episode_end[start:end],
                        shantens[start:end],
                        fuuro_counts[start:end],
                        riichi_turns[start:end],
                    )
                    start = end
                    end += batch_size
        pb.close()

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
