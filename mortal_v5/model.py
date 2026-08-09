"""mortal_v5 模型：ConvNeXt 三阶段局部建模 + Transformer 全局交互，约 1.1 亿参数"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE


class DropPath(nn.Module):
    """stochastic depth：训练时按概率丢弃残差路径"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1. - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 块：DWConv → LN → PWConv(↑4) → GELU → PWConv(↓1) + 残差"""

    def __init__(self, channels: int, *, kernel_size: int = 7, layer_scale: float = 1e-6,
                 drop_rate: float = 0., drop_path: float = 0.):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.actv = nn.GELU()
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path)

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
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attn.transpose(1, 2).reshape(b, l, d))


class TransformerBlock(nn.Module):
    """pre-norm transformer 块，残差分支末端按 GPT-2 式缩放到 0.02"""

    def __init__(self, dim: int, heads: int, *, ff_ratio: int = 4, drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_ratio),
            nn.GELU(),
            nn.Linear(dim * ff_ratio, dim),
        )
        self.drop_path = DropPath(drop_path)
        for m in (self.attn.proj, self.ff[2]):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class Brain(nn.Module):
    """v5 主干：stem + ConvNeXt 三阶段 + Transformer 顶层，输出 phi 供策略/价值/辅助头共享"""

    def __init__(
        self, *,
        version: int = 4, is_oracle: bool = False,
        widths: tuple[int, ...] = (192, 384, 768), depths: tuple[int, ...] = (8, 12, 10),
        kernel_size: int = 7, layer_scale: float = 1e-6, drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        attn_layers: int = 6, attn_heads: int = 12, pos_embed: bool = True,
        phi_dim: int = 2048,
    ):
        super().__init__()
        self.is_oracle = is_oracle
        self.phi_dim = phi_dim
        in_channels = obs_shape(version)[0]
        seq_len = obs_shape(version)[1]
        if is_oracle:
            in_channels += oracle_obs_shape(version)[0]

        c1, c2, c3 = widths
        total_blocks = sum(depths) + attn_layers
        dp_rates = torch.linspace(0., drop_path_rate, total_blocks).tolist()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, 3, padding=1, bias=False),
            nn.GELU(),
        )

        def stage(cin: int, cout: int, depth: int, rates: list[float]) -> nn.Sequential:
            layers = [nn.Conv1d(cin, cout, 1, bias=False)] if cin != cout else []
            layers += [
                ConvNeXtBlock(cout, kernel_size=kernel_size, layer_scale=layer_scale,
                              drop_rate=drop_rate, drop_path=r)
                for r in rates
            ]
            return nn.Sequential(*layers)

        n = 0
        self.s1 = stage(c1, c1, depths[0], dp_rates[n:n + depths[0]])
        n += depths[0]
        self.s2 = stage(c1, c2, depths[1], dp_rates[n:n + depths[1]])
        n += depths[1]
        self.s3 = stage(c2, c3, depths[2], dp_rates[n:n + depths[2]])
        n += depths[2]

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, c3)) if pos_embed else None
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.ModuleList([
            TransformerBlock(c3, attn_heads, drop_path=r) for r in dp_rates[n:]
        ])

        self.neck = nn.Sequential(
            nn.Conv1d(c3, 64, 1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * seq_len, phi_dim),
        )
        self.policy_head = nn.Linear(phi_dim, ACTION_SPACE)

    def forward(self, obs: Tensor, invisible_obs: Tensor | None = None) -> Tensor:
        if self.is_oracle:
            assert invisible_obs is not None
            obs = torch.cat((obs, invisible_obs), dim=1)
        x = self.stem(obs)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = x.transpose(1, 2)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.transformer:
            x = blk(x)
        x = x.transpose(1, 2)
        return self.neck(x)

    def policy_logits(self, phi: Tensor) -> Tensor:
        return self.policy_head(phi)


class DQN(nn.Module):
    """集成 Dueling DQN：K 个 head 共享隐藏层，输出 V 与 A"""

    def __init__(self, *, phi_dim: int = 2048, num_heads: int = 5, hidden: int = 1024):
        super().__init__()
        self.phi_dim = phi_dim
        self.num_heads = num_heads
        self.net = nn.Sequential(
            nn.Linear(phi_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_heads * (1 + ACTION_SPACE)),
        )
        nn.init.constant_(self.net[2].bias, 0)

    def forward(self, phi: Tensor, mask: Tensor) -> Tensor:
        v, a = self.net(phi).split((self.num_heads, self.num_heads * ACTION_SPACE), dim=-1)
        v = v.view(-1, self.num_heads, 1)
        a = a.view(-1, self.num_heads, ACTION_SPACE)
        mask = mask.unsqueeze(1)
        a_mean = a.masked_fill(~mask, 0.).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
        return q

    def value(self, phi: Tensor) -> Tensor:
        """提取 V(s)，IQL 用，返回 (N, K)"""
        v = self.net(phi).split((self.num_heads, self.num_heads * ACTION_SPACE), dim=-1)[0]
        return v if self.num_heads > 1 else v.squeeze(-1)


class AuxNet(nn.Module):
    """多任务辅助头：共享隐藏层后按 dims 切分"""

    def __init__(self, *, phi_dim: int = 2048, dims: tuple[int, ...] = (4, 7, 7, 7)):
        super().__init__()
        self.dims = dims
        self.net = nn.Sequential(
            nn.Linear(phi_dim, 512),
            nn.GELU(),
            nn.Linear(512, sum(dims)),
        )

    def forward(self, x: Tensor):
        return self.net(x).split(self.dims, dim=-1)
