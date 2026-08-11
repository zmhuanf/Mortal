"""BC 推理引擎：对接 OneVsThree，纯策略 argmax

接口属性以 libriichi mortal.rs 源码为准，只保留 arena 实际读取的字段"""
import numpy as np
import torch


class BcEngine:
    def __init__(self, brain, *, version, device, enable_amp, name='bc'):
        # libriichi MortalBatchAgent::new 仅访问以下 6 个字段
        self.engine_type = 'mortal'
        self.brain = brain.to(device).eval()
        self.name = name
        self.is_oracle = False
        self.version = version
        self.enable_amp = enable_amp
        self.enable_quick_eval = True
        self.enable_rule_based_agari_guard = True
        self.device = device
        self.player_ids = None

    def set_player_ids(self, player_ids):
        self.player_ids = player_ids

    def react_batch(self, obs, masks, invisible_obs):
        with torch.autocast(self.device.type, enabled=self.enable_amp), torch.inference_mode():
            o = torch.as_tensor(np.stack(obs), device=self.device)
            m = torch.as_tensor(np.stack(masks), device=self.device)
            logits = self.brain.policy_logits(self.brain(o)).masked_fill(~m, -torch.inf)
            actions = logits.argmax(-1)
            values = logits.float()
        n = actions.shape[0]
        return actions.tolist(), values.tolist(), m.tolist(), [True] * n
