import random
import torch
import numpy as np
from torch.utils.data import IterableDataset
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from config import config

class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
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
        # do not put it in __init__, it won't work on Windows
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
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
        n_step = config['env'].get('n_step', 3)
        gamma = float(config['env']['gamma'])
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
                shantens = np.array(game.take_shantens(), dtype=np.int64).clip(0, 6)
                fuuro_counts = np.frombuffer(game.take_fuuro_counts(), dtype=np.uint8).astype(np.int64).clip(0, 6)
                riichi_turns = np.frombuffer(game.take_riichi_turns(), dtype=np.uint8).astype(np.int64)

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
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:7] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                # apply_gamma 前缀和，按折扣步数定位 next_idx 而非 transition 偏移
                gamma_prefix = np.concatenate(([0], np.cumsum(np.asarray(apply_gamma, dtype=np.int64))))

                for i in range(game_size):
                    std = steps_to_done[i]
                    if std < n_step:
                        n_step_r = np.float32(gamma ** std * kyoku_rewards[at_kyoku[i]])
                        is_end = True
                        next_idx = i
                    else:
                        n_step_r = np.float32(0.0)
                        is_end = False
                        # 从 i 起第 n_step 个 apply_gamma 步之后的 transition
                        next_idx = int(np.searchsorted(gamma_prefix, gamma_prefix[i] + n_step, side='left'))
                        next_idx = min(next_idx, game_size - 1)

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
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
