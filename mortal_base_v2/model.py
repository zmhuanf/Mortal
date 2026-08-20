"""mortal_base_v2 模型：仅策略网络（Brain + policy 头），无任何 Q/价值/事件/辅助头

结构保持与 mortal_base 的 Brain 完全一致，以便直接加载 mortal.pth 的 mortal 权重
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE

# v4 观测中整行填充的标量/标志段（每行 34 列同值，取第 0 列无损）
SCALAR_ROWS = (
    (4, 7), (7, 15), (15, 19), (19, 23), (23, 25), (27, 28),
    (717, 718), (718, 722), (722, 723), (854, 857), (857, 860),
    (861, 862), (862, 869), (869, 871), (879, 880), (880, 883),
    (883, 885), (887, 889), (889, 891),
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
    """双尺度 ConvNeXt 块：k3+k9 DWConv 并行 → LN → PWConv(↑4) → GELU → PWConv(↓1) + 残差"""

    def __init__(self, channels: int, *, kernel_sizes: tuple[int, ...] = (3, 9),
                 layer_scale: float = 1e-6, drop_rate: float = 0., drop_path: float = 0.):
        super().__init__()
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
    """v2 策略网络：三阶段 ConvNeXt + 顶层注意力 + 标量分离流，只输出动作 logits 供 policy"""

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
        self.neck_ln = nn.LayerNorm(64)
        self.fc = nn.Linear(64 * 34, phi_dim)
        if self.scalar_channels > 0:
            self.fc_scalar = nn.Linear(64 * 34 + scalar_dim, phi_dim)
        else:
            self.fc_scalar = None
        self.actv = nn.GELU()
        self.policy_head = nn.Sequential(
            nn.Linear(phi_dim, phi_dim),
            nn.GELU(),
            nn.Linear(phi_dim, ACTION_SPACE),
        )

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

    def logits(self, obs: Tensor) -> Tensor:
        """纯 BC 推理：整条前向直接输出动作 logits"""
        scalar = self.scalar_proj(obs[:, self.scalar_idx, 0])
        tile = obs[:, self.tile_mask]
        phi = self.actv(self._encode(tile, scalar))
        return self.policy_head(phi)

    def forward(self, obs: Tensor, invisible_obs: Tensor | None = None) -> Tensor:
        assert invisible_obs is None, 'v2 仅支持非 oracle'
        return self.logits(obs)
