"""牌谱数据加载：产出每步观测、动作、最终排名（加权用）与辅助标签"""
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from libriichi.dataset import GameplayLoader
from config import config


class FileDatasetsIter(IterableDataset):
    def __init__(self, *, version, file_list, file_batch_size, num_epochs, enable_augmentation):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.file_batch_size = file_batch_size
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.iterator = None

    def build_iter(self):
        for _ in range(self.num_epochs):
            yield from self._load_files(False)
            if self.enable_augmentation:
                yield from self._load_files(True)

    def _load_files(self, augmented):
        random.shuffle(self.file_list)
        loader = GameplayLoader(version=self.version, oracle=False, augmented=augmented)
        buffer = []
        for start in range(0, len(self.file_list), self.file_batch_size):
            for file in loader.load_gz_log_files(self.file_list[start:start + self.file_batch_size]):
                for game in file:
                    self._populate(game, buffer)
            random.shuffle(buffer)
            yield from buffer
            buffer.clear()
        random.shuffle(buffer)
        yield from buffer

    @staticmethod
    def _populate(game, buffer):
        obs = game.take_obs()
        if not obs:
            return
        grp = game.take_grp()
        feat = grp.take_feature()
        if feat.shape[0] > 12:
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
            buffer.append((
                obs[i], actions[i], masks[i], final_rank,
                int(rank_seq[at_kyoku[i] + 1]), int(shantens[i]), int(fuuro[i]), int(riichi[i]),
            ))

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*_):
    """多 worker 各占全核会互相拖慢，固定单线程并切分文件列表"""
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    per = -(-len(ds.file_list) // info.num_workers)  # ceil div
    ds.file_list = ds.file_list[info.id * per:(info.id + 1) * per]
