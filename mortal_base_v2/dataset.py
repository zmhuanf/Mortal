"""纯 BC 数据加载：只产出 obs/action/mask，无奖励、无 GRP、无事件标签

管线与 mortal_base 一致（全局游标 + 确定性 shuffle + worker 互斥交错消费）
"""

import random
import time
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import IterableDataset

from libriichi.dataset import GameplayLoader
from config import config


class FileDatasetsIter(IterableDataset):
    def __init__(self, version, file_list, *, file_batch_size=20, player_names=None,
                 num_epochs=1, enable_augmentation=False, augmented_first=False,
                 resume_files=0, shuffle_seed=42, random_files=False):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.file_batch_size = file_batch_size
        self.player_names = player_names
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.random_files = random_files
        self.shuffle_seed = shuffle_seed
        self.iterator = None
        self.cursor = mp.Value('q', resume_files)

    def build_iter(self):
        self.buffer = []
        if self.random_files:
            rng = random.Random()
            self.buffer_rng = random.Random()
            self.loader = GameplayLoader(
                version=self.version, augmented=self.augmented_first, player_names=self.player_names,
            )
            while True:
                yield from self.populate_buffer(rng.sample(self.file_list, self.file_batch_size))
            return

        passes = []
        for _ in range(self.num_epochs):
            passes.append(self.augmented_first)
            if self.enable_augmentation:
                passes.append(not self.augmented_first)
        for pass_idx, augmented in enumerate(passes):
            yield from self.load_files(augmented, pass_idx)

    def load_files(self, augmented, pass_idx):
        rng = random.Random(self.shuffle_seed ^ (pass_idx * 0x9E3779B9))
        shuffled = rng.sample(self.file_list, len(self.file_list))
        self.buffer_rng = random.Random(self.shuffle_seed ^ 0x9E3779B9)
        self.loader = GameplayLoader(
            version=self.version, augmented=augmented, player_names=self.player_names,
        )
        self.buffer = []
        total = len(shuffled)
        base = pass_idx * total
        while True:
            with self.cursor.get_lock():
                pos = self.cursor.value
                start = pos - base
                if start < 0 or start >= total:
                    break
                end = min(pos + self.file_batch_size, base + total)
                self.cursor.value = end
            files = shuffled[start:end - base]
            if not files:
                break
            yield from self.populate_buffer(files)
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                obs = np.asarray(game.take_obs(), dtype=np.float16)
                actions = np.asarray(game.take_actions(), dtype=np.int64)
                masks = np.asarray(game.take_masks())
                T = len(obs)
                if T == 0:
                    continue
                for i in range(T):
                    self.buffer.append([obs[i], actions[i], masks[i]])
        self.buffer_rng.shuffle(self.buffer)
        yield from self.buffer
        del self.buffer[:]

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*args, **kwargs):
    torch.set_num_threads(1)
    time.sleep(random.random() * 2.0)
