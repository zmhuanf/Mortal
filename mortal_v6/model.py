"""mortal_v6 模型：ConvNeXt 三阶段 + Transformer 主干（约 8000 万参数）
+ XQL 价值头 + 事件世界模型 + 辅助头"""

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
    """v6 主干：stem + ConvNeXt 三阶段 + Transformer 顶层，输出 phi 供策略/价值/事件/辅助头共享"""

    def __init__(
        self, *,
        version: int = 4, is_oracle: bool = False,
        widths: tuple[int, ...] = (192, 384, 640), depths: tuple[int, ...] = (8, 10, 10),
        kernel_size: int = 7, layer_scale: float = 1e-6, drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        attn_layers: int = 6, attn_heads: int = 8, pos_embed: bool = True,
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


class QHead(nn.Module):
    """XQL 价值头：phi → 每动作 Q 值"""

    def __init__(self, *, phi_dim: int = 2048, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phi_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, ACTION_SPACE),
        )
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, phi: Tensor, mask: Tensor | None = None) -> Tensor:
        q = self.net(phi)
        if mask is not None:
            q = q.masked_fill(~mask, -torch.inf)
        return q


class EventModel(nn.Module):
    """事件世界模型：给定 (phi, 动作) 预测未来 horizon 手的事件类别序列（非自回归）

    事件类别：0=无事 1=立直 2=和牌 3=放铳 4=流局 5=被自摸
    """

    def __init__(self, *, phi_dim: int = 2048, action_space: int = ACTION_SPACE,
                 horizon: int = 10, n_types: int = 6,
                 dim: int = 256, heads: int = 4, layers: int = 2):
        super().__init__()
        self.horizon = horizon
        self.n_types = n_types
        self.action_emb = nn.Embedding(action_space, dim)
        self.proj = nn.Linear(phi_dim + dim, dim)
        self.time_pos = nn.Parameter(torch.zeros(1, horizon, dim))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, batch_first=True, norm_first=True, activation='gelu', dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(dim, n_types)

    def forward(self, phi: Tensor, action: Tensor) -> Tensor:
        b = phi.shape[0]
        a = self.action_emb(action)  # (b, dim)
        x = self.proj(torch.cat((phi, a), dim=-1)).unsqueeze(1)  # (b, 1, dim)
        out = self.encoder(self.time_pos + x)  # (b, horizon, dim)
        return self.head(out)  # (b, horizon, n_types)

    def rollout_returns(self, logits: Tensor, *, gamma: float, rewards: list[float]) -> Tensor:
        """把事件 logits 折算成折扣事件回报，终局回报全额计入后按生存概率截断"""
        probs = logits.softmax(-1)
        terminal = probs[..., [2, 3, 4, 5]].sum(-1)  # 和牌/放铳/流局/被自摸
        r = (probs * torch.as_tensor(rewards, device=logits.device)).sum(-1)
        survival = torch.cumprod((1 - terminal).clamp(min=0), dim=-1)
        # 终局步回报不被自身生存概率打折，只被此前各步截断
        survival_before = torch.cat((torch.ones_like(terminal[..., :1]), survival[..., :-1]), dim=-1)
        gamma_pow = torch.as_tensor(gamma, device=logits.device) ** torch.arange(
            self.horizon, device=logits.device
        )
        return (survival_before * r * gamma_pow).sum(-1)


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
