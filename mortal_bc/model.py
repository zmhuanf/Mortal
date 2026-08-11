"""BC 模型：三阶段 ConvNeXt + Transformer + 注意力池化

浅窄深宽分阶段分配容量，注意力做全局读牌交互，注意力池化保留 34 牌种位置语义"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from libriichi.consts import obs_shape, ACTION_SPACE


class ConvNeXtBlock(nn.Module):
    """DWConv → LN → PWConv↑4 → GELU → PWConv↓1 + 残差"""
    def __init__(self, channels, *, layer_scale=1e-6, drop_rate=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.actv = nn.GELU()
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.norm(x)
        x = self.actv(self.pwconv1(x))
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = self.drop(x).transpose(1, 2)
        return residual + x


class SelfAttention(nn.Module):
    """多头自注意力，SDPA 自动选内核"""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attn.transpose(1, 2).reshape(b, l, d))


class TransformerBlock(nn.Module):
    """pre-norm transformer 块"""
    def __init__(self, dim, heads, *, ff_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_ratio),
            nn.GELU(),
            nn.Linear(dim * ff_ratio, dim),
        )
        for m in (self.attn.proj, self.ff[2]):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Brain(nn.Module):
    """三阶段 ConvNeXt + Transformer + 注意力池化

    stage1 窄学局部牌型，stage2/3 宽学战术组合，注意力做全局读牌交互
    注意力池化代替 flatten，保留 34 牌种位置语义
    """
    def __init__(self, *, version=4, widths=(256, 512, 512), depths=(8, 8, 8),
                 attn_layers=6, attn_heads=8, layer_scale=1e-6, drop_rate=0.0):
        super().__init__()
        in_channels = obs_shape(version)[0]
        seq_len = obs_shape(version)[1]  # 34
        c1, c2, c3 = widths
        self.phi_dim = c3

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, 3, padding=1, bias=False),
            nn.GELU(),
        )

        def stage(cin, cout, depth):
            layers = []
            if cin != cout:
                layers.append(nn.Conv1d(cin, cout, 1, bias=False))
            layers += [ConvNeXtBlock(cout, layer_scale=layer_scale, drop_rate=drop_rate) for _ in range(depth)]
            return nn.Sequential(*layers)

        self.s1 = stage(c1, c1, depths[0])
        self.s2 = stage(c1, c2, depths[1])
        self.s3 = stage(c2, c3, depths[2])

        # 可学习位置编码，让注意力感知 34 个牌种位置
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, c3))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.Sequential(*[TransformerBlock(c3, attn_heads) for _ in range(attn_layers)])

        # 注意力池化：可学习 query 对 34 位置加权求和，不破坏空间结构
        self.pool_query = nn.Parameter(torch.zeros(1, 1, c3))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_key = nn.Linear(c3, c3)
        self.pool_value = nn.Linear(c3, c3)

        self.actv = nn.GELU()
        self.policy_head = nn.Linear(c3, ACTION_SPACE)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.stem(obs)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)               # (N, C, 34)
        x = x.transpose(1, 2)        # (N, 34, C)
        x = x + self.pos_embed
        x = self.transformer(x)       # (N, 34, C)
        # 注意力池化
        q = self.pool_query.expand(x.shape[0], -1, -1)  # (N, 1, C)
        k = self.pool_key(x)          # (N, 34, C)
        v = self.pool_value(x)        # (N, 34, C)
        attn = F.scaled_dot_product_attention(q, k, v)   # (N, 1, C)
        return self.actv(attn.squeeze(1))               # (N, C)

    def policy_logits(self, phi: Tensor) -> Tensor:
        return self.policy_head(phi)


class AuxNet(nn.Module):
    """多任务辅助头：共享单线性层后按 dims 切分"""
    def __init__(self, phi_dim=512, dims=(4, 7, 7, 7)):
        super().__init__()
        self.dims = dims
        self.net = nn.Linear(phi_dim, sum(dims), bias=False)

    def forward(self, x: Tensor):
        return self.net(x).split(self.dims, dim=-1)


class LegacyBrain(nn.Module):
    """旧架构，仅用于加载 baseline_v1 checkpoint 做评估对手"""
    def __init__(self, *, version=4, conv_channels=192, num_blocks=40, layer_scale=1e-6, drop_rate=0.0):
        super().__init__()
        in_channels = obs_shape(version)[0]
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, 3, padding=1, bias=False),
            *(ConvNeXtBlock(conv_channels, layer_scale=layer_scale, drop_rate=drop_rate) for _ in range(num_blocks)),
            nn.Conv1d(conv_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
        )
        self.actv = nn.GELU()
        self.policy_head = nn.Linear(1024, ACTION_SPACE)

    def forward(self, obs: Tensor) -> Tensor:
        return self.actv(self.encoder(obs))

    def policy_logits(self, phi: Tensor) -> Tensor:
        return self.policy_head(phi)
