"""mortal_base 模型：多尺度 ConvNeXt 主干 + 标量分离流 + 顶层注意力

Brain = 三阶段 ConvNeXt（k3+k9 双尺度 DWConv）+ 轻量 Transformer 融合
        + 54 行标量分离流（分数/局况/向听/can 标志/EV 锚点，34 列冗余消除）
DQN = Ensemble Dueling，version 1/2/3/4 多版本支持
AuxNet = 多任务辅助头（共享隐藏层），GRP = 全局排名预测器（reward 计算用）
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

# v4 观测中整行填充的标量/标志段（每行 34 列同值，取第 0 列无损）
# 布局随 libriichi obs_repr.rs 固定，非 oracle 时生效
SCALAR_ROWS = (
    (4, 7),        # akas
    (7, 15),       # scores
    (15, 19),      # rank
    (19, 23),      # kyoku
    (23, 25),      # honba + kyotaku
    (27, 28),      # kyoku_seq
    (717, 718),    # tiles_left
    (718, 722),    # doras_owned
    (722, 723),    # doras_unseen
    (854, 857),    # riichi_declared
    (857, 860),    # riichi_accepted
    (861, 862),    # at_furiten
    (862, 869),    # shanten
    (869, 871),    # self_riichi + at_kan_select
    (879, 880),    # can_riichi
    (880, 883),    # can_chi
    (883, 885),    # can_pon + can_daiminkan
    (887, 889),    # can_agari + can_ryukyoku
    (889, 891),    # ev_encode
)


class DropPath(nn.Module):
    """stochastic depth：训练时按概率丢弃残差路径并缩放保持期望"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1. - self.drop_prob
        mask = (keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)).floor_()
        return x.div(keep) * mask


class ConvNeXtBlock(nn.Module):
    """双尺度 ConvNeXt 块：k3+k9 DWConv 并行相加 → LN → PWConv(↑4) → GELU → PWConv(↓1) + 残差"""

    def __init__(self, channels: int, *, kernel_sizes: tuple[int, ...] = (3, 9),
                 layer_scale: float = 1e-6, drop_rate: float = 0., drop_path: float = 0.):
        super().__init__()
        # k3 捕捉近邻搭子，k9 覆盖花色内全段，并行相加让网络自选感受野
        self.dwconvs = nn.ModuleList([
            nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels)
            for k in kernel_sizes
        ])
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.actv = nn.GELU()
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = torch.stack([c(x) for c in self.dwconvs]).sum(0)
        x = self.norm(x.transpose(1, 2))
        x = self.pwconv1(x)
        x = self.actv(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = self.drop(x)
        return residual + self.drop_path(x.transpose(1, 2))


class SelfAttention(nn.Module):
    """多头自注意力，SDPA 自动选内核"""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, l, d))


class TransformerBlock(nn.Module):
    """pre-norm transformer 块，残差分支输出 0.02 缩放防 fp16 不稳定"""

    def __init__(self, dim: int, heads: int, *, drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.drop_path = DropPath(drop_path)
        for m in (self.attn.proj, self.ff[2]):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.ff(self.norm2(x)))


class Brain(nn.Module):
    """三阶段 ConvNeXt + 顶层注意力 + 标量分离流，输出 phi 供策略/价值/辅助头共享"""

    def __init__(self, *, version: int = 4, is_oracle: bool = False,
                 widths: tuple[int, int, int] = (256, 384, 512),
                 depths: tuple[int, int, int] = (8, 10, 6),
                 kernel_sizes: tuple[int, ...] = (3, 9),
                 layer_scale: float = 1e-6, drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 attn_layers: int = 2, attn_heads: int = 8,
                 phi_dim: int = 1024, scalar_dim: int = 128):
        super().__init__()
        self.is_oracle = is_oracle
        self.phi_dim = phi_dim

        in_channels = obs_shape(version)[0]
        if is_oracle:
            in_channels += oracle_obs_shape(version)[0]

        # 分离整行标量：34 列同值行只取列 0，消除冗余还省卷积容量
        mask = torch.ones(in_channels, dtype=torch.bool)
        if not is_oracle:
            for s, e in SCALAR_ROWS:
                mask[s:e] = False
        self.register_buffer('tile_mask', mask, persistent=False)
        self.register_buffer('scalar_idx', torch.nonzero(~mask).squeeze(-1), persistent=False)
        self.tile_channels = int(mask.sum())
        self.scalar_channels = in_channels - self.tile_channels

        c1, c2, c3 = widths
        self.stem = nn.Sequential(
            nn.Conv1d(self.tile_channels, c1, 3, padding=1, bias=False),
            nn.GELU(),
        )

        # drop_path 率随深度线性递增，浅层保留特征深层正则
        total_blocks = sum(depths) + attn_layers
        dp_rates = torch.linspace(0., drop_path_rate, total_blocks).tolist()
        n = 0

        def stage(cin: int, cout: int, depth: int) -> nn.Sequential:
            nonlocal n
            layers = [nn.Conv1d(cin, cout, 1, bias=False)] if cin != cout else []
            layers += [
                ConvNeXtBlock(cout, kernel_sizes=kernel_sizes,
                              layer_scale=layer_scale, drop_rate=drop_rate,
                              drop_path=dp_rates[n + i])
                for i in range(depth)
            ]
            n += depth
            return nn.Sequential(*layers)

        self.s1 = stage(c1, c1, depths[0])
        self.s2 = stage(c1, c2, depths[1])
        self.s3 = stage(c2, c3, depths[2])

        self.transformer = nn.ModuleList([
            TransformerBlock(c3, attn_heads, drop_path=dp_rates[n + i])
            for i in range(attn_layers)
        ])

        if self.scalar_channels > 0:
            self.scalar_proj = nn.Sequential(nn.Linear(self.scalar_channels, scalar_dim), nn.GELU())
        else:
            self.scalar_proj = None

        self.neck = nn.Sequential(nn.Conv1d(c3, 64, 1), nn.GELU())
        # 沿通道维归一化，抑制深层激活值域漂移
        self.neck_ln = nn.LayerNorm(64)
        self.fc = nn.Linear(64 * 34, phi_dim)
        if self.scalar_channels > 0:
            self.fc_scalar = nn.Linear(64 * 34 + scalar_dim, phi_dim)
        else:
            self.fc_scalar = None
        self.actv = nn.GELU()
        # 策略决策 = 牌效率 × 危险度 × 点差的非线性组合，单层 Linear 容量不足
        self.policy_head = nn.Sequential(
            nn.Linear(phi_dim, phi_dim),
            nn.GELU(),
            nn.Linear(phi_dim, ACTION_SPACE),
        )

    def forward(self, obs: Tensor, invisible_obs: Tensor | None = None) -> Tensor:
        if self.is_oracle:
            assert invisible_obs is not None
            obs = torch.cat((obs, invisible_obs), dim=1)
            return self.actv(self._encode(obs, None))
        scalar = self.scalar_proj(obs[:, self.scalar_idx, 0])
        tile = obs[:, self.tile_mask]
        return self.actv(self._encode(tile, scalar))

    def _encode(self, tile: Tensor, scalar: Tensor | None) -> Tensor:
        x = self.stem(tile)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = x.transpose(1, 2)
        for blk in self.transformer:
            x = blk(x)
        x = self.neck(x.transpose(1, 2)).transpose(1, 2)
        x = self.neck_ln(x).flatten(1)
        if scalar is not None:
            return self.fc_scalar(torch.cat((x, scalar), dim=-1))
        return self.fc(x)

    def policy_logits(self, phi: Tensor) -> Tensor:
        return self.policy_head(phi)

    def freeze_bn(self, value: bool):
        return self

    def reset_running_stats(self):
        pass


class AuxNet(nn.Module):
    """多任务辅助头，共享隐藏层后按 dims split"""

    def __init__(self, dims: tuple[int, ...] = (4, 7, 7, 7), hidden: int = 512):
        super().__init__()
        self.dims = dims
        self.net = nn.Sequential(
            nn.Linear(1024, hidden),
            nn.GELU(),
            nn.Linear(hidden, sum(dims), bias=False),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return self.net(x).split(self.dims, dim=-1)


class DQN(nn.Module):
    """Ensemble Dueling DQN，num_heads 个 head 独立输出 V 和 A"""

    def __init__(self, *, version: int = 1, num_heads: int = 1, hidden: int = 512):
        super().__init__()
        self.version = version
        self.num_heads = num_heads
        match version:
            case 1:
                # v1 的 phi 为 512 维
                self.v_head = nn.Linear(512, num_heads)
                self.a_head = nn.Linear(512, num_heads * ACTION_SPACE)
            case 2 | 3 | 4:
                # dueling 分离 MLP：Q 精度直接决定 AWR 优势质量，单层线性容量不足
                self.v_head = nn.Sequential(
                    nn.Linear(1024, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, num_heads),
                )
                self.a_head = nn.Sequential(
                    nn.Linear(1024, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, num_heads * ACTION_SPACE),
                )
        # V/A 从 0 起步，避免随机 init 的初始优势偏置
        for head in (self.v_head, self.a_head):
            last = head[-1] if isinstance(head, nn.Sequential) else head
            nn.init.zeros_(last.bias)

    def forward(self, phi: Tensor, mask: Tensor) -> Tensor:
        v = self.v_head(phi).unsqueeze(-1)
        a = self.a_head(phi).view(-1, self.num_heads, ACTION_SPACE)
        mask = mask.unsqueeze(1)
        a_sum = a.masked_fill(~mask, 0.).sum(-1, keepdim=True)
        mask_sum = mask.sum(-1, keepdim=True)
        a_mean = a_sum / mask_sum
        q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
        return q

    def value(self, phi: Tensor) -> Tensor:
        """提取 V(s)，IQL 需要，返回 (N, K) 或 (N,) when K=1"""
        v = self.v_head(phi)
        return v if self.num_heads > 1 else v.squeeze(-1)

    def advantage(self, phi: Tensor, mask: Tensor) -> Tensor:
        """暴露 dueling 的 A 分支 (N,K,A)，供 A 头事件监督回归"""
        a = self.a_head(phi).view(-1, self.num_heads, ACTION_SPACE)
        return a.masked_fill(~mask.unsqueeze(1), 0.)


class GRP(nn.Module):
    """全局排名预测器：局序特征序列 → 每局末各玩家排名概率矩阵"""

    def __init__(self, hidden_size: int = 128, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(GRP_SIZE, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, 12, hidden_size))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 16),
        )

    def forward(self, inputs: list[Tensor]) -> Tensor:
        lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.int64)
        padded = pad_sequence(inputs, batch_first=True)
        return self.forward_padded(padded, lengths)

    def forward_padded(self, padded: Tensor, lengths: Tensor) -> Tensor:
        mask = torch.arange(padded.shape[1], device=lengths.device)[None, :] >= lengths[:, None]
        x = self.input_proj(padded) + self.pos_emb[:, :padded.shape[1]]
        x = self.encoder(x, src_key_padding_mask=mask)
        idx = (lengths - 1).clamp(min=0)
        x = x[torch.arange(x.shape[0]), idx]
        return self.fc(x)

    def calc_matrix(self, logits: Tensor) -> Tensor:
        """(N, 16) -> (N, player, rank_prob)，Sinkhorn 迭代归一化为双随机矩阵"""
        matrix = logits.reshape(-1, 4, 4).softmax(-1)
        for _ in range(5):
            matrix = matrix / matrix.sum(-1, keepdim=True)
            matrix = matrix / matrix.sum(-2, keepdim=True)
        return matrix

    def get_label(self, rank_by_player: Tensor) -> Tensor:
        """(N, 4) -> (N, 4, 4) one-hot, label[player, rank] = 1"""
        batch_size = rank_by_player.shape[0]
        labels = torch.zeros(batch_size, 4, 4, dtype=torch.float32, device=rank_by_player.device)
        labels.scatter_(2, rank_by_player.unsqueeze(-1), 1)
        return labels
