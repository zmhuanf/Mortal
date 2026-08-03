import random
from collections import deque

import numpy as np


class SumTree:
    """线段树，支持 O(log N) 优先级更新与前缀和采样"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.max_priority = 1.0

    def update(self, idx, priority):
        """更新叶子节点 idx 的优先级并向上传播"""
        tree_idx = idx + self.capacity - 1
        diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        tree_idx = (tree_idx - 1) // 2
        while tree_idx >= 0:
            self.tree[tree_idx] += diff
            tree_idx = (tree_idx - 1) // 2
        self.max_priority = max(self.max_priority, priority)

    def get_leaf(self, value):
        """按累计优先级前缀和找到对应叶子节点"""
        tree_idx = 0
        while tree_idx < self.capacity - 1:
            left = 2 * tree_idx + 1
            right = left + 1
            if value <= self.tree[left]:
                tree_idx = left
            else:
                value -= self.tree[left]
                tree_idx = right
        data_idx = tree_idx - self.capacity + 1
        return data_idx, self.tree[tree_idx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """transition 级 PER buffer，跨 epoch 持久化"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_end=1.0, beta_anneal_steps=100000, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.eps = eps
        self.tree = SumTree(capacity)
        self.data = deque(maxlen=capacity)
        self.position = 0
        self.step = 0

    def __len__(self):
        return len(self.data)

    def add(self, samples):
        """批量添加 transitions，优先级设为当前最大值"""
        p = (self.tree.max_priority + self.eps) ** self.alpha
        for s in samples:
            if len(self.data) < self.capacity:
                self.data.append(s)
            else:
                self.data[self.position] = s
            self.tree.update(self.position, p)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """按 P(i)=p_i/Σp 优先采样，返回 transitions、索引、IS 权重"""
        total = self.tree.total
        segment = total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            idx, priority = self.tree.get_leaf(value)
            indices[i] = idx
            priorities[i] = priority

        # IS 权重修正偏差，beta 随训练退火到 1
        beta = self.beta + (self.beta_end - self.beta) * min(1.0, self.step / self.beta_anneal_steps)
        probs = priorities / (total + self.eps)
        weights = (len(self.data) * probs) ** (-beta)
        weights = weights / weights.mean()

        stacked = [np.stack([self.data[idx][j] for idx in indices]) for j in range(len(self.data[0]))]
        self.step += 1
        return stacked, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """训练后用新 TD error 更新被采样 transitions 的优先级"""
        for idx, td in zip(indices, td_errors):
            p = (abs(td) + self.eps) ** self.alpha
            self.tree.update(idx, p)
