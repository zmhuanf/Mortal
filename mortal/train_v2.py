def train():
    import prelude
    import config_v2  # 注册 config 模块，必须先于 from config import config

    import logging
    import sys
    import os
    import gc
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
    from config import config
    from common import submit_param, parameter_count, drain, tqdm, get_pool
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from libriichi.consts import obs_shape

    version = config['control']['version']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    next_rank_weight = config['aux']['next_rank_weight']
    shanten_weight = config['aux']['shanten_weight']
    fuuro_weight = config['aux']['fuuro_weight']
    riichi_turn_weight = config['aux']['riichi_turn_weight']
    online_human_ratio = config['dataset']['online_human_ratio']
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
    iql_tau = config['iql']['tau']
    iql_beta = config['iql']['beta']
    iql_clip = config['iql']['clip']
    ema_decay = config['iql']['ema_decay']
    bc_weight = config['distill']['bc_weight']
    top_k = config['distill']['top_k']
    bc_mode = config['distill']['bc_mode']
    bc_kyoku_threshold = config['distill']['bc_kyoku_threshold']
    dqn_num_heads = config['dqn']['num_heads']

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']
    pts = config['env']['pts']
    gamma = config['env']['gamma']
    n_step = config['env']['n_step']

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

    # AdamW 分组：Conv1d/Linear 的 weight 走 weight_decay，其余不衰减
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
    test_player = TestPlayer()
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
        target_mortal.load_state_dict(state['target_mortal'])
        target_dqn.load_state_dict(state['target_dqn'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        if 'pool_version' not in best_perf:
            best_perf['pool_version'] = state.get('pool_version', -1)
        steps = state['steps']
    else:
        # 热启动：仅加载权重，optimizer 全新，不带病态优化状态起步
        init_from = config['distill']['init_from']
        state = torch.load(init_from, weights_only=True, map_location=device)
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        target_mortal.load_state_dict(state['target_mortal'])
        target_dqn.load_state_dict(state['target_dqn'])
        logging.info(f'warm start from: {init_from}')
    last_pool_version = state.get('pool_version', best_perf.get('pool_version', -1)) if path.exists(state_file) else -1

    optimizer.zero_grad(set_to_none=True)
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    submit_param(mortal, dqn, is_idle=True)
    logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'v_loss': 0,
        'policy_loss': 0,
        'dqn_loss': 0,
        'bc_loss': 0,
        'next_rank_loss': 0,
        'shanten_loss': 0,
        'fuuro_loss': 0,
        'riichi_turn_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def update_target():
        with torch.no_grad():
            for tp, p in zip(target_mortal.parameters(), mortal.parameters()):
                tp.lerp_(p, 1 - ema_decay)
            for tp, p in zip(target_dqn.parameters(), dqn.parameters()):
                tp.lerp_(p, 1 - ema_decay)

    def train_epoch():
        nonlocal steps
        nonlocal idx
        nonlocal last_pool_version

        dirname = drain()
        selfplay_file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))

        # 人类牌谱常驻，file_index 缓存 glob 构建结果
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
        logging.info(f'self-play files: {len(selfplay_file_list):,}, human files: {len(human_file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        human_batch_size = max(1, int(batch_size * online_human_ratio))
        selfplay_batch_size = max(1, batch_size - human_batch_size)

        # 双 loader 按比例交错，self-play 全视角采样，对手动作一并进入行为分布
        # BC 标记字段由 bc_mode 决定，dataloader 只输出一个
        include_final_rank = bc_mode == 'top_k'
        include_kyoku_delta = bc_mode == 'kyoku_plus'
        selfplay_data = FileDatasetsIter(
            version = version,
            file_list = selfplay_file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = [],
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
            include_final_rank = include_final_rank,
            include_kyoku_delta = include_kyoku_delta,
        )
        human_data = FileDatasetsIter(
            version = version,
            file_list = human_file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = [],
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
            include_final_rank = include_final_rank,
            include_kyoku_delta = include_kyoku_delta,
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
        data_loader = (tuple(torch.cat([sp, hu], dim=0) for sp, hu in zip(sp_batch, hu_batch))
                       for sp_batch, hu_batch in zip(selfplay_loader, human_loader))

        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, player_ranks, next_obs, n_step_rewards, next_masks, is_episode_end, shantens, fuuro_counts, riichi_turns, bc_labels):
            nonlocal steps
            nonlocal idx
            nonlocal pb
            nonlocal best_perf
            nonlocal last_pool_version

            bs = obs.shape[0]
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
            masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device, non_blocking=True)
            next_obs = next_obs.to(dtype=torch.float32, device=device, non_blocking=True)
            n_step_rewards = n_step_rewards.to(dtype=torch.float32, device=device, non_blocking=True)
            next_masks = next_masks.to(dtype=torch.bool, device=device, non_blocking=True)
            is_episode_end = is_episode_end.to(dtype=torch.bool, device=device, non_blocking=True)
            shantens = shantens.to(dtype=torch.int64, device=device, non_blocking=True)
            fuuro_counts = fuuro_counts.to(dtype=torch.int64, device=device, non_blocking=True)
            riichi_turns = riichi_turns.to(dtype=torch.int64, device=device, non_blocking=True)
            bc_labels = bc_labels.to(device=device, non_blocking=True)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                q = dqn(phi, masks)[range(bs), :, actions]  # (N, K)

                # IQL：Q 回归到 n 步回报 + 折扣 V，V 用 expectile 保守估计，不自举
                with torch.no_grad():
                    next_phi = target_mortal(next_obs)
                    next_v = target_dqn.value(next_phi)  # (N, K)
                    q_target = n_step_rewards.unsqueeze(-1) + gamma ** n_step * next_v * (~is_episode_end).unsqueeze(-1)

                v = dqn.value(phi)  # (N, K)
                td = q_target - v
                v_loss = torch.where(td > 0, iql_tau * td ** 2, (1 - iql_tau) * td ** 2).mean()
                dqn_loss = F.huber_loss(q, q_target, delta=10)

                # AWR：advantage 取 TD target 减 V，与 Q 网络输出解耦，避免 Q 波动污染策略
                with torch.no_grad():
                    exp_adv = ((q_target - v).mean(-1) / iql_beta).clamp(max=iql_clip).exp()
                policy_logits = mortal.policy_logits(phi)
                log_prob = policy_logits.log_softmax(-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                policy_loss = -(exp_adv * log_prob).mean()

                next_rank_logits, shanten_logits, fuuro_logits, riichi_turn_logits = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)
                shanten_loss = ce(shanten_logits, shantens)
                fuuro_loss = ce(fuuro_logits, fuuro_counts)
                riichi_turn_loss = ce(riichi_turn_logits, riichi_turns)

                loss = (
                    v_loss + policy_loss + dqn_loss
                    + next_rank_loss * next_rank_weight
                    + shanten_loss * shanten_weight
                    + fuuro_loss * fuuro_weight
                    + riichi_turn_loss * riichi_turn_weight
                )
                # BC 蒸馏：按 bc_mode 选取优质样本模仿其动作
                bc_loss = torch.tensor(0., device=device)
                if top_k > 0:
                    bc_mask = bc_labels < top_k if bc_mode == 'top_k' else bc_labels > bc_kyoku_threshold
                    if bc_mask.any():
                        bc_loss = F.cross_entropy(policy_logits[bc_mask], actions[bc_mask])
                        loss = loss + bc_weight * bc_loss
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['v_loss'] += v_loss
                stats['policy_loss'] += policy_loss
                stats['dqn_loss'] += dqn_loss
                stats['bc_loss'] += bc_loss
                stats['next_rank_loss'] += next_rank_loss
                stats['shanten_loss'] += shanten_loss
                stats['fuuro_loss'] += fuuro_loss
                stats['riichi_turn_loss'] += riichi_turn_loss
                all_q[idx] = q.mean(-1)
                all_q_target[idx] = q_target.mean(-1)

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

            if steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # 降采样减小 tensorboard 事件体积，断点续训首区间批数可能不足 save_every
                n_batches = idx
                all_q_1d = all_q[:n_batches].cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target[:n_batches].cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/v_loss', stats['v_loss'] / n_batches, steps)
                writer.add_scalar('loss/policy_loss', stats['policy_loss'] / n_batches, steps)
                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / n_batches, steps)
                writer.add_scalar('loss/bc_loss', stats['bc_loss'] / n_batches, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / n_batches, steps)
                writer.add_scalar('loss/shanten_loss', stats['shanten_loss'] / n_batches, steps)
                writer.add_scalar('loss/fuuro_loss', stats['fuuro_loss'] / n_batches, steps)
                writer.add_scalar('loss/riichi_turn_loss', stats['riichi_turn_loss'] / n_batches, steps)
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

                if steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
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
                    # 池子升级后旧基准作废，本次评估建立新基准且不覆盖 best.pth
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
                    # 在线模式评估后退出重启，规避未知原因导致的卡死
                    sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        for batch_tensors in data_loader:
            train_batch(*batch_tensors)
        pb.close()
        submit_param(mortal, dqn, is_idle=True)
        logging.info('idle param has been submitted')

    while True:
        train_epoch()
        gc.collect()

def main():
    import os
    import sys
    import time
    from subprocess import Popen

    # 勿手动设置该环境变量
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    if os.environ.get(is_sub_proc_key, '0') == '1':
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
