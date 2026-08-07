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
        """组内（batch 维度）归一化，日麻中一组 = 一轮 rollout 的若干局"""
        return (rewards - rewards.mean()) / (rewards.std() + eps)

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        game_ids: torch.Tensor,
    ) -> dict[str, float]:
        """按局分块梯度累积。obs 等大张量留 CPU 块内搬运，显存仅与 chunk_games 成正比
        advantages 每局一个按 game_ids 广播到各步"""
        device = next(self.policy.parameters()).device
        advantages = advantages.to(device)

        num_games = advantages.numel()
        # 局内步连续且局号递增，位置区间才能直接对应局号区间
        assert game_ids[0] == 0 and bool((game_ids[1:] >= game_ids[:-1]).all())

        self.optimizer.zero_grad()
        loss = 0.0
        total_steps = 0
        clip_frac = kl_sum = ratio_sum = 0.0
        for start in range(0, num_games, self.chunk_games):
            end = min(start + self.chunk_games, num_games)
            sel = (game_ids >= start) & (game_ids < end)
            ids = (game_ids[sel] - start).to(device)

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

            adv = advantages[start:end][ids]
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
            per_step = -(torch.min(surr1, surr2) - self.kl_beta * kl)

            # 一局内步级损失取均值，再对块内各局取均值；块间按局数加权等价于整批均值
            counts = torch.bincount(ids, minlength=end - start)
            sums = torch.zeros(end - start, dtype=per_step.dtype, device=device).scatter_add_(0, ids, per_step)
            block_loss = (sums / counts).mean()
            block_loss.backward()

            with torch.inference_mode():
                clip_frac += ((ratio - 1).abs() > self.clip_eps).float().sum().item()
                kl_sum += kl.sum().item()
                ratio_sum += ratio.sum().item()
            loss += block_loss.item() * (end - start)
            total_steps += sel.sum().item()

        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'loss': loss / num_games,
            'clip_frac': clip_frac / total_steps,
            'kl': kl_sum / total_steps,
            'ratio': ratio_sum / total_steps,
            'adv_std': advantages.std().item(),
        }
