"""mortal_v7 数据加载：每局一个 (RTG, 状态, 动作) 序列样本

RTG 由本家分数差分构造
"""

import random
import time
import multiprocessing as mp
import torch
import numpy as np
from torch.utils.data import IterableDataset
from libriichi.dataset import GameplayLoader
from config import config


class FileDatasetsIter(IterableDataset):
    def __init__(self, version, file_list, pts, *, oracle=False, file_batch_size=20,
                 reserve_ratio=0, player_names=None, excludes=None, num_epochs=1,
                 enable_augmentation=False, augmented_first=False, resume_files=0,
                 shuffle_seed=42, batch_size=8):
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
        self.shuffle_seed = shuffle_seed
        self.batch_size = batch_size
        self.t_max = config['dataset']['t_max']
        self.seg_len = config['dataset']['seg_len']
        self.score_scale = config['rtg']['score_scale']
        self.target_score = config['rtg']['target_score']
        self.target_ratio = config['rtg']['target_ratio']
        self.iterator = None
        # 全局文件游标：worker 互斥消费，resume 时从保存值精确接续
        self.cursor = mp.Value('q', resume_files)

    def build_iter(self):
        # 遍次序列：每 epoch 先 augmented_first 遍再其反遍（若启用增强）
        passes = []
        for _ in range(self.num_epochs):
            passes.append(self.augmented_first)
            if self.enable_augmentation:
                passes.append(not self.augmented_first)
        for pass_idx, augmented in enumerate(passes):
            yield from self.load_files(augmented, pass_idx)

    def load_files(self, augmented, pass_idx):
        # 固定 seed 的确定性 shuffle：resume 后顺序一致，游标精确对应已训位置
        rng = random.Random(self.shuffle_seed ^ (pass_idx * 0x9E3779B9))
        shuffled = rng.sample(self.file_list, len(self.file_list))
        self.loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
        )
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
            yield from self.populate_buffer(files)  # 段级流水线：单段产出，主进程攒批，文件解析方差不形成突发

    def populate_buffer(self, file_list):
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                obs = np.asarray(game.take_obs(), dtype=np.float16)  # (T, 1012, 34)，fp16 减半 buffer 内存带宽
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = np.asarray(game.take_actions(), dtype=np.int64)
                masks = np.asarray(game.take_masks())
                at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)

                T = len(obs)
                if T == 0:
                    continue

                grp = game.take_grp()
                player_id = game.take_player_id()
                grp_feature = grp.take_feature()
                # 15 局长局可达 16 行，防御性上限放宽到 32，避免切段样本被丢
                if grp_feature.shape[0] > 32:
                    continue
                final_scores = grp.take_final_scores()
                # 本家每局开始分数（万分制列还原），RTG = 目标/终局分与当前分之差
                score_at_kyoku = grp_feature[:, 3 + player_id] * 1e4
                final_score = float(final_scores[player_id])
                if random.random() < self.target_ratio:
                    # 目标语义 RTG：距目标分的差距，覆盖推理分布
                    rtg_full = ((self.target_score - score_at_kyoku[at_kyoku]) / self.score_scale).astype(np.float32)
                else:
                    # 实际语义 RTG：终局分与当前分之差，局内常数、局间跳变
                    rtg_full = ((final_score - score_at_kyoku[at_kyoku]) / self.score_scale).astype(np.float32)
                # 长局切段：每段一个样本，RTG 保持全局语义，位置从段首 0 起
                for start in range(0, T, self.seg_len):
                    end = min(start + self.seg_len, T)
                    if self.oracle:
                        entry = (obs[start:end], invisible_obs[start:end], rtg_full[start:end],
                                 actions[start:end], masks[start:end])
                    else:
                        entry = (obs[start:end], rtg_full[start:end],
                                 actions[start:end], masks[start:end])
                    yield entry

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*args, **kwargs):
    # 多 worker 各占全核推理互相拖慢，固定单线程
    torch.set_num_threads(1)
    # 随机相位错开启动，避免同步攒批导致 GPU 周期性饥饿（锯齿）
    time.sleep(random.random() * 2.0)


def collate_single(batch):
    """batch_size=1 时解包单样本，供主进程攒批分桶"""
    return batch[0]


def collate_batch(batch):
    """局间决策数不同，padding 到 batch 内最大值，valid 标记有效位"""
    obs_list, rtg_list, act_list, mask_list = zip(*batch)
    B = len(batch)
    T = max(o.shape[0] for o in obs_list)
    obs = torch.zeros(B, T, *obs_list[0].shape[1:], dtype=torch.bfloat16)  # bf16 省显存，编码特征精度足够
    rtg = torch.zeros(B, T, dtype=torch.float32)
    acts = torch.zeros(B, T, dtype=torch.int64)
    masks = torch.zeros(B, T, *mask_list[0].shape[1:], dtype=torch.bool)
    valid = torch.zeros(B, T, dtype=torch.bool)
    for i, (o, r, a, m) in enumerate(batch):
        t = o.shape[0]
        obs[i, :t] = torch.from_numpy(np.asarray(o))
        rtg[i, :t] = torch.from_numpy(r)
        acts[i, :t] = torch.from_numpy(a)
        masks[i, :t] = torch.from_numpy(np.asarray(m))
        valid[i, :t] = True
    return obs, rtg, acts, masks, valid
