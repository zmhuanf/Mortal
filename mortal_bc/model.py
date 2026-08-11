"""纯 BC 模型：ConvNeXt 编码器 + 策略头 + 辅助头
字段名与原版 Brain 对齐，可直接加载 baseline_v1 权重作评估对手"""
import torch
from torch import nn, Tensor
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


class Brain(nn.Module):
    """ConvNeXt 编码器，输出 phi 供策略头与辅助头共享"""
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


class AuxNet(nn.Module):
    """多任务辅助头：共享单线性层后按 dims 切分"""
    def __init__(self, dims=(4, 7, 7, 7)):
        super().__init__()
        self.dims = dims
        self.net = nn.Linear(1024, sum(dims), bias=False)

    def forward(self, x: Tensor):
        return self.net(x).split(self.dims, dim=-1)
