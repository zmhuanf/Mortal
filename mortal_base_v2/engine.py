"""mortal_base_v2 引擎：唯一策略直出（policy argmax），无 Q/事件/搜索/风险调节

react_batch 接受 libriichi 新 pyd 透传的 indexes，兼容竞技场
"""

import traceback

import torch
import numpy as np
from torch.distributions import Categorical
from typing import *


class PolicyEngine:
    def __init__(self, brain, is_oracle, version, device=None, enable_amp=False,
                 amp_dtype=torch.float16, enable_rule_based_agari_guard=False, name='mortal_base_v2',
                 enable_quick_eval=False, boltzmann_epsilon=0, boltzmann_temp=1, top_p=1):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        self.brain = brain.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.enable_amp = enable_amp
        self.amp_dtype = amp_dtype
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name
        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

    def react_batch(self, obs, masks, invisible_obs, indexes=None):
        try:
            with (
                torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
                masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
                batch_size = obs.shape[0]
                feat = self.brain(obs)
                # v2 Brain 直出 logits；V4 对手返回 phi 需过 policy_logits
                logits = feat if feat.shape[-1] == 46 else self.brain.policy_logits(feat)
                logits = logits.masked_fill(~masks, -torch.inf)
                if self.boltzmann_epsilon > 0:
                    is_greedy = torch.full((batch_size,), 1 - self.boltzmann_epsilon,
                                           device=self.device).bernoulli().to(torch.bool)
                    if self.boltzmann_temp > 0:
                        masked = logits.masked_fill(~masks, 0.)
                        cnt = masks.sum(-1).clamp_min(1)
                        mean = masked.sum(-1) / cnt
                        var = ((masked - mean.unsqueeze(-1) * masks) ** 2).sum(-1) / cnt
                        std = var.sqrt().clamp_min(1e-6)
                        logits = logits / (std.unsqueeze(-1) * self.boltzmann_temp)
                    sampled = sample_top_p(logits, self.top_p)
                    actions = torch.where(is_greedy, logits.argmax(-1), sampled)
                else:
                    is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                    actions = logits.argmax(-1)
                return actions.tolist(), logits.tolist(), masks.tolist(), is_greedy.tolist()
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')


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
