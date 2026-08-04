import numpy as np


class SumTree:
    """线段树，支持 O(log N) 优先级更新与前缀和采样"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.max_priority = 1.0

    def update(self, idx, priority):
        tree_idx = idx + self.capacity - 1
        diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        tree_idx = (tree_idx - 1) // 2
        while tree_idx >= 0:
            self.tree[tree_idx] += diff
            tree_idx = (tree_idx - 1) // 2
        self.max_priority = max(self.max_priority, priority)

    def batch_update(self, indices, priorities):
        """批量更新叶子节点优先级，逐层向量化传播"""
        tree_indices = np.asarray(indices, dtype=np.int64) + self.capacity - 1
        if len(tree_indices) == 0:
            return
        self.tree[tree_indices] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))
        while tree_indices[0] > 0:
            tree_indices = np.unique((tree_indices - 1) // 2)
            left = 2 * tree_indices + 1
            right = left + 1
            self.tree[tree_indices] = self.tree[left] + self.tree[right]

    def get_leaf(self, value):
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

    def batch_get_leaf(self, values):
        """批量前缀和采样，逐层向量化遍历"""
        values = values.copy()
        tree_indices = np.zeros(len(values), dtype=np.int64)
        while True:
            is_internal = tree_indices < self.capacity - 1
            if not is_internal.any():
                break
            ti = tree_indices[is_internal]
            left = 2 * ti + 1
            right = left + 1
            left_vals = self.tree[left]
            go_left = values[is_internal] <= left_vals
            values[is_internal] = np.where(go_left, values[is_internal], values[is_internal] - left_vals)
            tree_indices[is_internal] = np.where(go_left, left, right)
        data_indices = tree_indices - self.capacity + 1
        return data_indices, self.tree[tree_indices]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """transition 级 PER buffer，列式存储 + 向量化采样"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_end=1.0, beta_anneal_steps=100000, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.eps = eps
        self.tree = SumTree(capacity)
        self.columns = None
        self.num_fields = 0
        self.position = 0
        self._size = 0
        self.step = 0

    def __len__(self):
        return self._size

    def _init_columns(self, batch_arrays):
        """按首批数据形状预分配列式存储"""
        self.num_fields = len(batch_arrays)
        self.columns = [
            np.zeros((self.capacity,) + arr.shape[1:], dtype=arr.dtype)
            for arr in batch_arrays
        ]

    def add(self, batch_arrays):
        """批量添加 transitions，batch_arrays 为各字段的 (N, *shape) numpy 数组列表"""
        n = len(batch_arrays[0])
        if n == 0:
            return
        if self.columns is None:
            self._init_columns(batch_arrays)
        # 旧版逐条 add 会使 max_priority 递增，用不动点迭代逼近终值
        p = (self.tree.max_priority + self.eps) ** self.alpha
        for _ in range(30):
            p = (p + self.eps) ** self.alpha
        self.tree.max_priority = max(self.tree.max_priority, p)
        priorities = np.full(n, p, dtype=np.float64)
        positions = (self.position + np.arange(n)) % self.capacity
        for j in range(self.num_fields):
            self.columns[j][positions] = batch_arrays[j]
        self.tree.batch_update(positions, priorities)
        self.position = (self.position + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size):
        """按 P(i)=p_i/Σp 优先采样，返回 transitions、索引、IS 权重"""
        total = self.tree.total
        segment = total / batch_size
        values = np.random.uniform(
            segment * np.arange(batch_size),
            segment * np.arange(1, batch_size + 1),
        )
        indices, priorities = self.tree.batch_get_leaf(values)

        beta = self.beta + (self.beta_end - self.beta) * min(1.0, self.step / self.beta_anneal_steps)
        probs = priorities / (total + self.eps)
        weights = (self._size * probs) ** (-beta)
        weights = weights / weights.mean()

        stacked = [col[indices] for col in self.columns]
        self.step += 1
        return stacked, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """训练后用新 TD error 批量更新优先级"""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        self.tree.batch_update(indices, priorities)
