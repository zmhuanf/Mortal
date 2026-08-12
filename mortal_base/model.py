"""mortal_base 模型：与 mortal/model.py 结构完全一致

Brain = ConvNeXt 编码 + GELU + 独立策略头（IQL 的 AWR 策略输出）
DQN = Ensemble Dueling，version 1/2/3/4 多版本支持
AuxNet = 多任务辅助头，GRP = 全局排名预测器（reward 计算用）
"""

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 块：DWConv → LN → PWConv(↑4) → GELU → PWConv(↓1) + 残差"""

    def __init__(self, channels: int, *, layer_scale: float = 1e-6, drop_rate: float = 0.):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.actv = nn.GELU()
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2))
        x = self.pwconv1(x)
        x = self.actv(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = self.drop(x)
        return residual + x.transpose(1, 2)


class ConvNeXtEncoder(nn.Module):
    """局面快照编码，(B, 1012, 34) -> (B, 1024)，每步独立可并行"""

    def __init__(self, in_channels: int, conv_channels: int, num_blocks: int,
                 *, layer_scale: float = 1e-6, drop_rate: float = 0.):
        super().__init__()
        blocks = [
            ConvNeXtBlock(conv_channels, layer_scale=layer_scale, drop_rate=drop_rate)
            for _ in range(num_blocks)
        ]
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            *blocks,
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Brain(nn.Module):
    def __init__(self, *, conv_channels: int, num_blocks: int, is_oracle: bool = False,
                 version: int = 4, layer_scale: float = 1e-6, drop_rate: float = 0.):
        super().__init__()
        self.is_oracle = is_oracle
        self.version = version

        in_channels = obs_shape(version)[0]
        if is_oracle:
            in_channels += oracle_obs_shape(version)[0]

        self.encoder = ConvNeXtEncoder(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            layer_scale=layer_scale,
            drop_rate=drop_rate,
        )
        self.actv = nn.GELU()
        # 独立策略头，IQL 的 AWR 策略输出，与 DQN 价值解耦
        self.policy_head = nn.Linear(1024, ACTION_SPACE)

    def forward(self, obs: Tensor, invisible_obs: Tensor | None = None) -> Tensor:
        if self.is_oracle:
            assert invisible_obs is not None
            obs = torch.cat((obs, invisible_obs), dim=1)
        phi = self.encoder(obs)
        return self.actv(phi)

    def policy_logits(self, phi: Tensor) -> Tensor:
        return self.policy_head(phi)

    def freeze_bn(self, value: bool):
        return self

    def reset_running_stats(self):
        pass


class AuxNet(nn.Module):
    """多任务辅助头，共享单线性层后按 dims split"""

    def __init__(self, dims: tuple[int, ...] = (4, 7, 7, 7)):
        super().__init__()
        self.dims = dims
        self.net = nn.Linear(1024, sum(dims), bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return self.net(x).split(self.dims, dim=-1)


class DQN(nn.Module):
    """Ensemble Dueling DQN，num_heads 个 head 独立输出 V 和 A"""

    def __init__(self, *, version: int = 1, num_heads: int = 1):
        super().__init__()
        self.version = version
        self.num_heads = num_heads
        match version:
            case 1:
                self.v_head = nn.Linear(512, num_heads)
                self.a_head = nn.Linear(512, num_heads * ACTION_SPACE)
            case 2 | 3:
                hidden_size = 512 if version == 2 else 256
                self.v_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, num_heads),
                )
                self.a_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, num_heads * ACTION_SPACE),
                )
            case 4:
                self.net = nn.Linear(1024, num_heads * (1 + ACTION_SPACE))
                nn.init.constant_(self.net.bias, 0)

    def forward(self, phi: Tensor, mask: Tensor) -> Tensor:
        if self.version == 4:
            v, a = self.net(phi).split((self.num_heads, self.num_heads * ACTION_SPACE), dim=-1)
            v = v.view(-1, self.num_heads, 1)
            a = a.view(-1, self.num_heads, ACTION_SPACE)
        else:
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
        if self.version == 4:
            v = self.net(phi).split((self.num_heads, self.num_heads * ACTION_SPACE), dim=-1)[0]
        else:
            v = self.v_head(phi)
        return v if self.num_heads > 1 else v.squeeze(-1)


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
