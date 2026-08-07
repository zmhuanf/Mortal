"""GRPO 策略网络：宽而浅 1D CNN，无价值头（GRPO 不需要 critic）"""
import torch
from torch import nn
from libriichi.consts import obs_shape, ACTION_SPACE


class ConvNeXtBlock(nn.Module):
    """DWConv → LN → PW(×2) → GELU → PW → residual。FFN 因子 2 减 FLOPs"""
    def __init__(self, channels: int, layer_scale: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, channels * 2)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(channels * 2, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2))
        x = self.pw2(self.act(self.pw1(x)))
        x = self.gamma * x
        return residual + x.transpose(1, 2)


class PolicyNet(nn.Module):
    def __init__(
        self,
        version: int = 4,
        conv_channels: int = 256,
        num_blocks: int = 12,
        tail_channels: int = 64,
        hidden: int = 512,
    ):
        super().__init__()
        self.version = version
        in_channels = obs_shape(version)[0]
        self.stem = nn.Conv1d(in_channels, conv_channels, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(
            *(ConvNeXtBlock(conv_channels) for _ in range(num_blocks))
        )
        self.tail = nn.Sequential(
            nn.Conv1d(conv_channels, tail_channels, 3, padding=1),
            nn.GELU(),
        )
        # 全局平均池化替代展平，省掉 (34×C→hidden) 的巨额线性层
        self.fc = nn.Linear(tail_channels, hidden)
        self.act = nn.GELU()
        self.policy_head = nn.Linear(hidden, ACTION_SPACE)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """返回未 mask 的 46 维动作 logits"""
        x = self.stem(obs)
        x = self.blocks(x)
        x = self.tail(x)
        x = x.mean(-1)
        x = self.act(self.fc(x))
        return self.policy_head(x)

    def log_probs(self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """合法动作掩码下所选动作的 log_prob（BC 与 GRPO 共用）"""
        logits = self(obs).masked_fill(~masks, -torch.inf)
        return logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
