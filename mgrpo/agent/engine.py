"""对局引擎：rollout 采样引擎与对手贪心引擎"""
import numpy as np
import torch
from torch.distributions import Categorical


class _BaseEngine:
    """libriichi MortalBatchAgent 约定的 Python 接口"""
    engine_type = 'mortal'
    enable_quick_eval = False

    def __init__(self, net, device, name: str, version: int = 4):
        self.net = net.to(torch.device(device)).eval()
        self.name = name
        self.is_oracle = False
        self.version = version
        self.enable_rule_based_agari_guard = True

    def _logits(self, obs, masks) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.net.parameters()).device
        obs_t = torch.as_tensor(np.stack(obs), device=device)
        masks_t = torch.as_tensor(np.stack(masks), device=device)
        logits = self.net(obs_t).masked_fill(~masks_t, -torch.inf)
        return logits, masks_t


class GRPOEngine(_BaseEngine):
    """GRPO rollout 引擎：策略采样；log_prob 由轨迹解析按实际执行动作重算"""
    def react_batch(self, obs, masks, invisible_obs):
        logits, _ = self._logits(obs, masks)
        with torch.inference_mode():
            actions = Categorical(logits=logits).sample()
        return actions.tolist(), logits.tolist(), masks, [False] * len(actions)


class OpponentEngine(_BaseEngine):
    """对手引擎：贪心选动作，稳定且无探索"""
    def react_batch(self, obs, masks, invisible_obs):
        logits, _ = self._logits(obs, masks)
        actions = logits.argmax(-1)
        return actions.tolist(), logits.tolist(), masks, [True] * len(actions)
