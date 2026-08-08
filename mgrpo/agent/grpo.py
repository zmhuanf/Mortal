"""GRPO：组内归一化 advantage + importance ratio clip + KL 正则（DeepSeek 式）"""
import torch
from torch import nn


class GRPO:
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        optimizer,
        *,
        clip_eps: float = 0.2,
        kl_beta: float = 0.01,
        max_grad_norm: float = 1.0,
        chunk_games: int = 8,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.max_grad_norm = max_grad_norm
        self.chunk_games = chunk_games  # 梯度累积每块局数，控制前向激活显存

    @staticmethod
    def compute_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """组内（batch 维度）归一化，日麻中一组 = 一轮 rollout 的全部局"""
        return (rewards - rewards.mean()) / (rewards.std() + eps)

    def begin(self, advantages: torch.Tensor):
        """开启一轮梯度累积，advantages 为全局局级归一化值"""
        self._adv = advantages.to(next(self.policy.parameters()).device)
        self._num_games = advantages.numel()
        self._offset = 0
        self._loss = 0.0
        self._total_steps = 0
        self._clip_frac = self._kl_sum = self._ratio_sum = 0.0
        self.optimizer.zero_grad()

    def feed(self, obs, actions, masks, old_log_probs, game_ids):
        """流式喂入一批轨迹（全局局号递增），按 chunk_games 分块累积梯度，obs 只驻留当前块"""
        device = next(self.policy.parameters()).device
        assert game_ids[0] == self._offset and bool((game_ids[1:] >= game_ids[:-1]).all())
        end_g = int(game_ids[-1]) + 1
        for lo in range(self._offset // self.chunk_games * self.chunk_games, end_g, self.chunk_games):
            hi = min(lo + self.chunk_games, end_g)
            sel = (game_ids >= lo) & (game_ids < hi)
            ids = (game_ids[sel] - lo).to(device)

            obs_g = obs[sel].to(device, non_blocking=True)
            act_g = actions[sel].to(device, non_blocking=True)
            mask_g = masks[sel].to(device, non_blocking=True)
            old_g = old_log_probs[sel].to(device, non_blocking=True)

            logits = self.policy(obs_g).masked_fill(~mask_g, -torch.inf)
            log_probs = logits.log_softmax(-1).gather(-1, act_g.unsqueeze(-1)).squeeze(-1)
            ref_log_probs = self.ref_policy.log_probs(obs_g, act_g, mask_g)

            ratio = (log_probs - old_g).exp()
            # k3 无偏 KL 估计，约束策略不偏离 reference（BC 起点）
            kl = (ref_log_probs - log_probs).exp() - (ref_log_probs - log_probs) - 1

            adv = self._adv[lo:hi][ids]
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
            per_step = -(torch.min(surr1, surr2) - self.kl_beta * kl)

            # 一局内步级损失取均值，再对块内各局取均值；块间按局数加权等价于整批均值
            counts = torch.bincount(ids, minlength=hi - lo)
            sums = torch.zeros(hi - lo, dtype=per_step.dtype, device=device).scatter_add_(0, ids, per_step)
            block_loss = (sums / counts).mean()
            block_loss.backward()

            with torch.inference_mode():
                self._clip_frac += ((ratio - 1).abs() > self.clip_eps).float().sum().item()
                self._kl_sum += kl.sum().item()
                self._ratio_sum += ratio.sum().item()
            self._loss += block_loss.item() * (hi - lo)
            self._total_steps += sel.sum().item()
        self._offset = end_g

    def end(self) -> dict[str, float]:
        """结束一轮：clip + step，返回统计"""
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {
            'loss': self._loss / self._num_games,
            'clip_frac': self._clip_frac / self._total_steps,
            'kl': self._kl_sum / self._total_steps,
            'ratio': self._ratio_sum / self._total_steps,
            'adv_std': self._adv.std().item(),
        }
