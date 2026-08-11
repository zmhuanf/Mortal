"""mortal_v6 数据加载：复用 libriichi GameplayLoader
每个样本附带 n 步回报、next 动作（XQL 自举）与局内事件轨迹（事件世界模型监督）"""

import random
import torch
import numpy as np
from torch.utils.data import IterableDataset
from libriichi.dataset import GameplayLoader
from config import config


class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle=False,
        file_batch_size=20,
        reserve_ratio=0,
        player_names=None,
        excludes=None,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None

    def build_iter(self):
        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        n_step = config['env'].get('n_step', 5)
        gamma = float(config['env']['gamma'])
        gamma_pow = np.float32(gamma) ** np.arange(n_step + 1, dtype=np.float32)
        event_horizon = int(config['event']['horizon'])
        reward_cfg = config.get('reward', {})
        riichi_reward = float(reward_cfg.get('riichi', 0.0))
        agari_reward = float(reward_cfg.get('agari', 0.0))
        houjuu_reward = float(reward_cfg.get('houjuu', 0.0))
        score_scale = float(reward_cfg.get('score_scale', 1000.0))
        reward_clip = float(reward_cfg.get('clip', 0.0))
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()
                is_riichi_turn = np.array(game.take_is_riichi_turn(), dtype=bool)
                is_agari_turn = np.array(game.take_is_agari_turn(), dtype=bool)
                is_houjuu_turn = np.array(game.take_is_houjuu_turn(), dtype=bool)
                shantens = np.array(game.take_shantens(), dtype=np.int64).clip(0, 6)
                fuuro_counts = np.frombuffer(game.take_fuuro_counts(), dtype=np.uint8).astype(np.int64).clip(0, 6)
                riichi_turns = np.frombuffer(game.take_riichi_turns(), dtype=np.uint8).astype(np.int64).clip(0, 6)

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)
                if game_size == 0:
                    continue

                grp_feature = grp.take_feature()
                if grp_feature.shape[0] > 12:
                    continue
                rank_by_player = grp.take_rank_by_player()
                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:7] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]
                # 局奖励 = 本家每局实际得分差分，无偏且不依赖额外模型
                kyoku_rewards = (scores_seq[1:, player_id] - scores_seq[:-1, player_id]) / score_scale
                if reward_clip > 0:
                    kyoku_rewards = np.clip(kyoku_rewards, -reward_clip, reward_clip)
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                # 末步必然为局末，从倒数第二步起回推可避免 i+1 越界
                for i in reversed(range(game_size - 1)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                # apply_gamma 前缀和，按折扣步数定位 next_idx 而非 transition 偏移
                gamma_prefix = np.concatenate(([0], np.cumsum(np.asarray(apply_gamma, dtype=np.int64))))

                # 局奖励定位到局末步，与 turn shaping 合并成统一奖励序列
                dones_arr = np.asarray(dones, dtype=bool)
                at_kyoku_arr = np.frombuffer(at_kyoku, dtype=np.uint8).astype(np.int64)
                kyoku_reward_at_step = np.zeros(game_size, dtype=np.float32)
                kyoku_reward_at_step[dones_arr] = kyoku_rewards[at_kyoku_arr[dones_arr]]
                step_rewards = kyoku_reward_at_step + (
                    is_riichi_turn * riichi_reward
                    + is_agari_turn * agari_reward
                    + is_houjuu_turn * houjuu_reward
                ).astype(np.float32)

                # 事件轨迹标签：每手事件类别，局外填充 -1 由 CE ignore 处理
                event_ids = np.zeros(game_size, dtype=np.int64)
                event_ids[is_riichi_turn] = 1
                event_ids[is_agari_turn] = 2
                event_ids[is_houjuu_turn] = 3
                other_done = dones_arr & (event_ids == 0)
                # 局末差分<0 为本家旁观自摸，与流局/旁观荣和分开
                event_ids[other_done] = np.where(kyoku_rewards[at_kyoku_arr[other_done]] < 0, 5, 4)
                kyoku_end = np.searchsorted(at_kyoku_arr, at_kyoku_arr, side='right') - 1
                col = np.arange(event_horizon)[None, :]
                idx = np.arange(game_size)[:, None] + col  # (G, H)
                valid = idx <= kyoku_end[:, None]
                event_traj = np.full((game_size, event_horizon), -1, dtype=np.int64)
                event_traj[valid] = event_ids[idx[valid]]

                for i in range(game_size):
                    std = steps_to_done[i]
                    if std < n_step:
                        is_end = True
                        next_idx = i
                        j_end = kyoku_end[i]  # 本局最后物理步
                    else:
                        is_end = False
                        # 从 i 起第 n_step 个 apply_gamma 步之后的 transition
                        next_idx = int(np.searchsorted(gamma_prefix, gamma_prefix[i] + n_step, side='left'))
                        next_idx = min(next_idx, game_size - 1)
                        j_end = next_idx
                    # 物理步按折扣编号归组，0 折扣步并入其前一折扣步
                    gk = gamma_prefix[i:j_end + 1] - gamma_prefix[i]
                    keep = gk < n_step
                    gk = gk[keep]
                    rsum = np.bincount(gk, weights=step_rewards[i:j_end + 1][keep])
                    n_step_r = np.float32(np.dot(gamma_pow[:len(rsum)], rsum))

                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        player_ranks[at_kyoku[i] + 1],
                        obs[next_idx],
                        n_step_r,
                        masks[next_idx],
                        is_end,
                        shantens[i],
                        fuuro_counts[i],
                        riichi_turns[i],
                        actions[next_idx],  # XQL 自举用数据动作
                        event_traj[i],
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*args, **kwargs):
    # 多 worker 各占全核推理互相拖慢，固定单线程
    torch.set_num_threads(1)
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
