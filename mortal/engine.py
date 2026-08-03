import json
import traceback
import torch
import numpy as np
from torch.distributions import Categorical
from typing import *

class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device = None,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        name = 'NoName',
        boltzmann_epsilon = 0,
        boltzmann_temp = 1,
        top_p = 1,
        uncertainty_scale = 0,
        temperature = None,
        action_source = 'q',
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
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p
        self.uncertainty_scale = uncertainty_scale
        self.temperature = temperature
        self.action_source = action_source

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
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

        if self.action_source == 'policy':
            logits = self.brain.policy_logits(phi).masked_fill(~masks, -torch.inf)
            # ensemble uncertainty 加到 policy logits 上驱动探索
            if self.uncertainty_scale > 0 and self.num_heads > 1:
                q_out = self.dqn(phi, masks)  # (N, K, A)
                q_std = q_out.std(1)  # (N, A)
                logits = (logits + self.uncertainty_scale * q_std).masked_fill(~masks, -torch.inf)
            values = logits
        else:
            q_out = self.dqn(phi, masks)  # (N, K, A)
            if self.num_heads > 1:
                q_mean = q_out.mean(1)
            else:
                q_mean = q_out.squeeze(1)
            logits = q_mean.masked_fill(~masks, -torch.inf)
            values = q_mean

        if self.boltzmann_epsilon > 0:
            # epsilon-greedy：大多数走贪心，少数走温度采样探索
            is_greedy = torch.full((batch_size,), 1 - self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            if self.temperature is not None and self.temperature > 0:
                # per-sample std 归一化，softmax 锐度不依赖 q 绝对量级
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

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int]):
        pass
