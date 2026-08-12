"""牌谱数据加载：产出每步观测、动作、最终排名（加权用）与辅助标签"""
import random
import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import IterableDataset
from libriichi.dataset import GameplayLoader
from config import config


class FileDatasetsIter(IterableDataset):
    def __init__(self, *, version, file_list, file_batch_size, num_epochs,
                 enable_augmentation, resume_files=0, shuffle_seed=42):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.file_batch_size = file_batch_size
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.shuffle_seed = shuffle_seed
        self.iterator = None
        # 全局文件游标：worker 互斥消费，resume 时从保存值精确接续
        self.cursor = mp.Value('q', resume_files)

    def build_iter(self):
        # 每 epoch 先原始遍再增强遍，pass 序号决定 shuffle 种子与游标偏移
        passes = []
        for _ in range(self.num_epochs):
            passes.append(False)
            if self.enable_augmentation:
                passes.append(True)
        for pass_idx, augmented in enumerate(passes):
            yield from self._load_files(augmented, pass_idx)

    def _load_files(self, augmented, pass_idx):
        # 固定 seed 确定性 shuffle：resume 后顺序一致，游标精确对应已训位置
        rng = random.Random(self.shuffle_seed ^ (pass_idx * 0x9E3779B9))
        shuffled = rng.sample(self.file_list, len(self.file_list))
        loader = GameplayLoader(version=self.version, oracle=False, augmented=augmented)
        total = len(shuffled)
        base = pass_idx * total
        while True:
            # 读-判-推进原子化，越过本 pass 时不动游标以保留断点
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
            # 边解析边产出单样本：worker 产出细粒度，多 worker 交错供给 GPU 平滑
            for file in loader.load_gz_log_files(files):
                for game in file:
                    yield from self._populate(game)

    def _populate(self, game):
        obs = game.take_obs()
        if not obs:
            return
        grp = game.take_grp()
        feat = grp.take_feature()
        # 15 局长局可达 16 行，防御性上限放宽到 32，避免长局样本被丢
        if feat.shape[0] > 32:
            return
        player_id = game.take_player_id()
        final_rank = int(grp.take_rank_by_player()[player_id])
        actions = game.take_actions()
        masks = game.take_masks()
        at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
        shantens = np.array(game.take_shantens(), dtype=np.int64).clip(0, 6)
        fuuro = np.frombuffer(game.take_fuuro_counts(), dtype=np.uint8).astype(np.int64).clip(0, 6)
        riichi = np.frombuffer(game.take_riichi_turns(), dtype=np.uint8).astype(np.int64).clip(0, 6)
        # 每局结束后的排名序列，next_rank 标签取当前局结束后的排名
        scores = np.concatenate((feat[:, 3:7] * 1e4, [grp.take_final_scores()]))
        rank_seq = (-scores).argsort(-1, kind='stable').argsort(-1, kind='stable')[:, player_id]
        for i in range(len(obs)):
            yield (obs[i], actions[i], masks[i], final_rank,
                   int(rank_seq[at_kyoku[i] + 1]), int(shantens[i]), int(fuuro[i]), int(riichi[i]))

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*_):
    """多 worker 各占全核会互相拖慢，固定单线程"""
    torch.set_num_threads(1)
