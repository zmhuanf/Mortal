"""mortal_v6 想象搜索：策略 top-k 候选 → 事件世界模型 rollout 回报 + XQL Q 混合评分"""

import torch
from config import config


def search_action(
    phi: torch.Tensor,
    mask: torch.Tensor,
    brain,
    q_head,
    event_model,
    *,
    k: int | None = None,
    alpha: float | None = None,
    gamma: float | None = None,
    rewards: list[float] | None = None,
) -> torch.Tensor:
    """对单个局面 (1, phi_dim) 输出选中的动作标量"""
    cfg = config['eval']
    k = k if k is not None else cfg['search_k']
    alpha = alpha if alpha is not None else cfg['search_alpha']
    gamma = gamma if gamma is not None else float(config['env']['gamma'])
    rewards = rewards if rewards is not None else config['event_loss']['rewards']

    with torch.inference_mode():
        logits = brain.policy_logits(phi).masked_fill(~mask, -torch.inf)
        candidates = logits.topk(min(k, mask.sum().item())).indices.squeeze(0)  # (k,)
        k = candidates.shape[0]
        phi_k = phi.expand(k, -1)
        mask_k = mask.expand(k, -1)
        ev_logits = event_model(phi_k, candidates)  # (k, horizon, n_types)
        rollout = event_model.rollout_returns(ev_logits, gamma=gamma, rewards=rewards)  # (k,)
        q = q_head(phi_k, mask_k).gather(1, candidates.unsqueeze(-1)).squeeze(-1)

        def z(x):
            # 候选内标准化，消除 Q 与事件回报的量纲差
            return (x - x.mean()) / x.std(correction=0).clamp_min(1e-8)
        score = alpha * z(q) + (1 - alpha) * z(rollout)
        return candidates[score.argmax()]
