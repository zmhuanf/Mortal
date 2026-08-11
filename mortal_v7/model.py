"""mortal_v7 模型：ConvNeXt 状态编码 + 因果 Transformer 的 Decision Transformer

序列 token = (RTG, 状态, 动作) 三型交错，纯监督训练动作，无值函数
"""

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from libriichi.consts import obs_shape, ACTION_SPACE


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
    """局面快照编码，(B, 1012, 34) -> (B, 1024)，每步独立可并行

    block 级 gradient checkpointing：反向重算前向换显存，适配 8GB 卡长局峰值
    """

    def __init__(self, in_channels: int, conv_channels: int, num_blocks: int,
                 *, layer_scale: float = 1e-6, drop_rate: float = 0.):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(conv_channels, layer_scale=layer_scale, drop_rate=drop_rate)
            for _ in range(num_blocks)
        )
        self.head = nn.Sequential(
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return self.head(x)


class DTBlock(nn.Module):
    """pre-norm transformer 块，attn_mask 由外层传入保证因果性"""

    def __init__(self, dim: int, nhead: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.norm1(x)
        x = x + self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class DecisionTransformer(nn.Module):
    """因果序列模型：以 RTG 为条件生成动作，动作位置 p % 3 == 2"""

    def __init__(self, *, version: int = 4, conv_channels: int = 256, num_blocks: int = 32,
                 d_model: int = 512, num_layers: int = 10, nhead: int = 8, seq_len: int = 288,
                 layer_scale: float = 1e-6, drop_rate: float = 0.):
        super().__init__()
        self.version = version
        self.d_model = d_model
        self.seq_len = seq_len

        in_channels = obs_shape(version)[0]
        self.encoder = ConvNeXtEncoder(
            in_channels, conv_channels, num_blocks,
            layer_scale=layer_scale, drop_rate=drop_rate,
        )
        self.state_proj = nn.Sequential(nn.Linear(1024, d_model), nn.LayerNorm(d_model))
        self.action_emb = nn.Embedding(ACTION_SPACE, d_model)
        self.rtg_proj = nn.Linear(1, d_model)
        self.type_emb = nn.Embedding(3, d_model)
        # 固定窗口长度下绝对位置与 RoPE 等价，实现更稳
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([DTBlock(d_model, nhead) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, ACTION_SPACE)

    def forward(self, obs: Tensor, rtg: Tensor, actions: Tensor) -> Tensor:
        """obs (B,T,1012,34) rtg (B,T) actions (B,T) -> logits (B,3T,A)（action 位置预测当前动作）

        动作 embedding 右移一位：位置 3t+2 输入 a_{t-1} 预测 a_t，训练推理一致
        """
        B, T = obs.shape[:2]
        phi = self.encoder(obs.flatten(0, 1)).view(B, T, -1)
        rtg_tok = self.rtg_proj(rtg.unsqueeze(-1))
        state_tok = self.state_proj(phi)
        # GPT 式 shift：预测 a_t 时不暴露 a_t 自身，首步补 0 占位
        prev_actions = torch.cat((torch.zeros(B, 1, dtype=actions.dtype, device=actions.device),
                                  actions[:, :-1]), dim=1)
        act_tok = self.action_emb(prev_actions)
        tok = torch.stack((rtg_tok, state_tok, act_tok), dim=2).reshape(B, 3 * T, -1)
        types = (torch.arange(3 * T, device=obs.device) % 3).expand(B, -1)
        tok = tok + self.type_emb(types) + self.pos_emb[:, :3 * T]
        # 标准因果掩码：只看过去（动作位置输入已是上一动作，无泄露）
        attn_mask = torch.triu(torch.ones(3 * T, 3 * T, dtype=torch.bool, device=obs.device), diagonal=1)
        for block in self.blocks:
            # transformer 层同样重计算省显存，attn_mask 闭包捕获不参与梯度
            tok = checkpoint(lambda t: block(t, attn_mask), tok, use_reentrant=False)
        return self.action_head(self.ln_f(tok))
