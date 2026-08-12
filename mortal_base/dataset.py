"""mortal_base 数据加载：baseline 配方 + v7 工程管线

reward：GRP 预测排名概率 → 点位期望差分（calc_delta_pt）+ turn-level shaping
改进点（相对 dataloader.py）：n 步窗口按 apply_gamma 折扣步对齐而非 turn 索引；
is_end 以游戏末而非局末为界，窗口内逐局计入结算奖励
管线：v7 式全局游标 + 固定 seed 确定性 shuffle + worker 互斥交错消费
"""

import random
import time
import multiprocessing as mp
import torch
import numpy as np
from torch.utils.data import IterableDataset
from libriichi.dataset import GameplayLoader
from model import GRP
from config import config


class RewardCalculator:
    """GRP 排名概率 → 每局末预期点位差分，与 mortal/reward_calculator.py 一致"""

    def __init__(self, grp, pts, uniform_init=False):
        self.device = torch.device('cpu')
        self.grp = grp.to(self.device).eval()
        self.uniform_init = uniform_init
        self.pts = torch.tensor(pts, dtype=torch.float32, device=self.device)

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        with torch.inference_mode():
            feat = torch.as_tensor(grp_feature, device=self.device, dtype=torch.float32)
            n = feat.shape[0]
            lengths = torch.arange(1, n + 1, device=self.device)
            # 下三角掩码一次铺开全部前缀，免去逐前缀切片
            keep = torch.tril(torch.ones(n, n, dtype=torch.bool, device=self.device))
            padded = feat.unsqueeze(0).expand(n, n, -1).masked_fill(~keep.unsqueeze(-1), 0.)
            matrix = self.grp.calc_matrix(self.grp.forward_padded(padded, lengths))

        final_ranking = torch.zeros((1, 4), device=self.device)
        final_ranking[0, rank_by_player[player_id]] = 1.
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        if self.uniform_init:
            rank_prob[0, :] = 1 / 4
        return rank_prob

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        exp_pts = rank_prob @ self.pts
        return (exp_pts[1:] - exp_pts[:-1]).cpu().numpy()


class FileDatasetsIter(IterableDataset):
    def __init__(self, version, file_list, pts, *, oracle=False, file_batch_size=20,
                 reserve_ratio=0, player_names=None, excludes=None, num_epochs=1,
                 enable_augmentation=False, augmented_first=False, resume_files=0,
                 shuffle_seed=42):
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
        self.n_step = config['env']['n_step']
        self.gamma = float(config['env']['gamma'])
        self.reward_cfg = config['reward']
        # 预计算 γ^k 供 n 步内 turn 奖励折扣累加
        self.gamma_pow = np.float32(self.gamma) ** np.arange(self.n_step + 1, dtype=np.float32)
        self.shuffle_seed = shuffle_seed
        self.iterator = None
        # 全局文件游标：worker 互斥消费，resume 时从保存值精确接续
        self.cursor = mp.Value('q', resume_files)

    def build_iter(self):
        # 不能放 __init__：Windows spawn 下 worker 需重新构造
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(self.grp, self.pts)

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
        self.buffer_rng = random.Random(self.shuffle_seed ^ 0x9E3779B9)
        self.loader = GameplayLoader(
            version=self.version,
            oracle=self.oracle,
            player_names=self.player_names,
            excludes=self.excludes,
            augmented=augmented,
        )
        self.buffer = []
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
            yield from self.populate_buffer(files)
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        riichi_reward = float(self.reward_cfg.get('riichi', 0.0))
        agari_reward = float(self.reward_cfg.get('agari', 0.0))
        houjuu_reward = float(self.reward_cfg.get('houjuu', 0.0))
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                obs = np.asarray(game.take_obs(), dtype=np.float16)  # (T, 1012, 34) fp16 减半 buffer 内存带宽
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = np.asarray(game.take_actions(), dtype=np.int64)
                masks = np.asarray(game.take_masks())
                at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8).astype(np.int64)
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()
                is_riichi_turn = np.array(game.take_is_riichi_turn(), dtype=bool)
                is_agari_turn = np.array(game.take_is_agari_turn(), dtype=bool)
                is_houjuu_turn = np.array(game.take_is_houjuu_turn(), dtype=bool)
                shantens = np.array(game.take_shantens(), dtype=np.int64).clip(0, 6)
                fuuro_counts = np.frombuffer(game.take_fuuro_counts(), dtype=np.uint8).astype(np.int64).clip(0, 6)
                riichi_turns = np.frombuffer(game.take_riichi_turns(), dtype=np.uint8).astype(np.int64)

                T = len(obs)
                if T == 0:
                    continue

                grp = game.take_grp()
                player_id = game.take_player_id()
                grp_feature = grp.take_feature()
                # GRP pos_emb 仅支持 12 行局序，超长局直接丢弃（同 baseline）
                if grp_feature.shape[0] > 12:
                    continue
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1

                # 局序排名序列：局初分数 + 终局分排序，aux 预测下一局排名用
                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:7] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                # 到游戏末结算的折扣步数（含全部 turn），is_end 以游戏末而非局末为界
                steps_to_done = np.zeros(T + 1, dtype=np.int64)
                for i in reversed(range(T)):
                    steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                # apply_gamma 前缀和，按折扣步数定位 next_idx 而非 transition 偏移
                gamma_prefix = np.concatenate(([0], np.cumsum(np.asarray(apply_gamma, dtype=np.int64))))
                # 局末 turn：结算奖励发生在局末 turn 动作之后
                kyoku_end_turns = np.flatnonzero(dones)

                turn_rewards = (
                    is_riichi_turn * riichi_reward
                    + is_agari_turn * agari_reward
                    + is_houjuu_turn * houjuu_reward
                ).astype(np.float32)

                for i in range(T):
                    std = steps_to_done[i]
                    if std < self.n_step:
                        is_end = True
                        next_idx = i
                        end = T
                    else:
                        is_end = False
                        # 从 i 起第 n_step 个 apply_gamma 步之后的 transition
                        next_idx = int(np.searchsorted(gamma_prefix, gamma_prefix[i] + self.n_step, side='left'))
                        next_idx = min(next_idx, T - 1)
                        end = next_idx
                    # turn 奖励按实际折扣步数折扣，非折扣步（副露/杠/和牌选择）不衰减
                    n_step_r = np.float32(np.dot(
                        self.gamma_pow[gamma_prefix[i:end] - gamma_prefix[i]],
                        turn_rewards[i:end],
                    ))
                    # 局结算奖励：结算折扣步数在窗口内的局逐局计入
                    for j in kyoku_end_turns[np.searchsorted(kyoku_end_turns, i, side='left'):]:
                        discount = int(gamma_prefix[j + 1] - gamma_prefix[i])
                        if discount >= self.n_step:
                            break
                        n_step_r += np.float32(self.gamma_pow[discount] * kyoku_rewards[at_kyoku[j]])

                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        np.int64(player_ranks[at_kyoku[i] + 1]),
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

        # buffer 内二次 shuffle：确定性 seed，resume 后同批文件顺序可复现
        self.buffer_rng.shuffle(self.buffer)
        reserved_size = int(len(self.buffer) * self.reserve_ratio)
        if reserved_size > len(self.buffer):
            return
        yield from self.buffer[reserved_size:]
        del self.buffer[reserved_size:]

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*args, **kwargs):
    # 多 worker 各占全核推理互相拖慢，固定单线程
    torch.set_num_threads(1)
    # 随机相位错开启动，避免同步攒批导致 GPU 周期性饥饿
    time.sleep(random.random() * 2.0)
