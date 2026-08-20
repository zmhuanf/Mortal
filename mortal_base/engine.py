"""mortal_v7 自包含引擎副本：champion（baseline 等 v4 对手）使用

react_batch 接受 libriichi 新 pyd 透传的 indexes，旧引擎调用不受影响
"""

import json
import traceback
import torch
import numpy as np
from torch.distributions import Categorical
from libriichi.consts import ACTION_SPACE
from typing import *


class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device=None,
        enable_amp=False,
        amp_dtype=torch.float16,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=False,
        name='NoName',
        boltzmann_epsilon=0,
        boltzmann_temp=1,
        top_p=1,
        uncertainty_scale=0,
        temperature=None,
        action_source='q',
        event_model=None,
        search_k=8,
        search_alpha=0.0,
        search_gamma=0.99,
        event_rewards=None,
        risk_gain=0.3,
        risk_v_mid=1.6,
        risk_w_riichi=1.0,
        risk_w_agari=1.0,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.num_heads = getattr(dqn, 'num_heads', 1)

        self.enable_amp = enable_amp
        self.amp_dtype = amp_dtype
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p
        self.uncertainty_scale = uncertainty_scale
        self.temperature = temperature
        self.action_source = action_source
        self.event_model = event_model
        self.search_k = search_k
        self.search_alpha = search_alpha
        self.search_gamma = search_gamma
        self.event_rewards = event_rewards or [0.0, 0.79, 2.26, -1.77, 0.0, -0.74]
        self.risk_gain = risk_gain
        self.risk_v_mid = risk_v_mid
        self.risk_w_riichi = risk_w_riichi
        self.risk_w_agari = risk_w_agari

    def react_batch(self, obs, masks, invisible_obs, indexes=None):
        try:
            with (
                torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        batch_size = obs.shape[0]

        phi = self.brain(obs, invisible_obs)

        if self.action_source == 'search':
            # 想象搜索：全部合法动作作为候选（top-k 会漏掉防守牌，全展开让事件模型看到完整选项）
            logits = self.brain.policy_logits(phi).masked_fill(~masks, -torch.inf)
            k = ACTION_SPACE
            cand = torch.arange(k, device=phi.device).repeat(batch_size)
            phi_k = phi.unsqueeze(1).expand(batch_size, k, -1).reshape(-1, phi.shape[-1])
            ev_logits = self.event_model(phi_k, cand)  # (B*k, H, n_types)
            rollout = self.event_model.rollout_returns(
                ev_logits, gamma=self.search_gamma, rewards=self.event_rewards
            ).view(batch_size, k)
            if self.search_alpha > 0 and self.dqn is not None:
                q = self.dqn(phi, masks)
                q_mean = q.mean(1) if self.num_heads > 1 else q.squeeze(1)
                # 候选即全体动作，无需 gather
                z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-8)  # noqa: E731
                score = self.search_alpha * z(q_mean) + (1 - self.search_alpha) * z(rollout)
            else:
                score = rollout
            score = score.masked_fill(~masks, -torch.inf)  # 非法候选剔除
            sel = score.argmax(-1)
            values = logits  # 与 policy 分支同形状 (B, A)，Rust 接口要求每动作一个值
        elif self.action_source == 'vrisk':
            # V 风险调节：领先（V 高）压立直/和牌倾向，落后（V 低）抬激进，其余切牌不动
            logits = self.brain.policy_logits(phi).masked_fill(~masks, -torch.inf)
            if self.dqn is not None:
                v = self.dqn.value(phi)
                v = v.mean(-1) if v.dim() > 1 else v
                delta = self.risk_gain * (v - self.risk_v_mid)  # >0 领先=压制，<0 落后=鼓励
                l = logits.clone()
                b = torch.arange(batch_size, device=phi.device)
                l[b, torch.full((batch_size,), 37, device=phi.device)] -= delta * self.risk_w_riichi
                l[b, torch.full((batch_size,), 43, device=phi.device)] -= delta * self.risk_w_agari
                logits = l.masked_fill(~masks, -torch.inf)
            values = logits
        elif self.action_source == 'policy':
            logits = self.brain.policy_logits(phi).masked_fill(~masks, -torch.inf)
            if self.uncertainty_scale > 0 and self.num_heads > 1:
                q_out = self.dqn(phi, masks)
                q_std = q_out.std(1)
                logits = (logits + self.uncertainty_scale * q_std).masked_fill(~masks, -torch.inf)
            values = logits
        else:
            q_out = self.dqn(phi, masks)
            if self.num_heads > 1:
                q_mean = q_out.mean(1)
            else:
                q_mean = q_out.squeeze(1)
            logits = q_mean.masked_fill(~masks, -torch.inf)
            values = q_mean

        if self.action_source == 'search':
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = sel
        elif self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1 - self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            if self.temperature is not None and self.temperature > 0:
                masked = logits.masked_fill(~masks, 0.)
                cnt = masks.sum(-1).clamp_min(1)
                mean = masked.sum(-1) / cnt
                var = ((masked - mean.unsqueeze(-1) * masks) ** 2).sum(-1) / cnt
                std = var.sqrt().clamp_min(1e-6)
                logits = logits / (std.unsqueeze(-1) * self.temperature)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, logits.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = logits.argmax(-1)

        return actions.tolist(), values.tolist(), masks.tolist(), is_greedy.tolist()


def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled
