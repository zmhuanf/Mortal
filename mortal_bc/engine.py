"""BC 推理引擎：对接 OneVsThree，纯策略 argmax"""
import numpy as np
import torch


class BcEngine:
    def __init__(self, brain, *, version, device, enable_amp, name='bc'):
        self.engine_type = 'mortal'
        self.brain = brain.to(device).eval()
        self.device = device
        self.enable_amp = enable_amp
        self.version = version
        self.name = name

    def react_batch(self, obs, masks, invisible_obs):
        with torch.autocast(self.device.type, enabled=self.enable_amp), torch.inference_mode():
            o = torch.as_tensor(np.stack(obs), device=self.device)
            m = torch.as_tensor(np.stack(masks), device=self.device)
            logits = self.brain.policy_logits(self.brain(o)).masked_fill(~m, -torch.inf)
            actions = logits.argmax(-1)
            values = logits.float()
        n = actions.shape[0]
        return actions.tolist(), values.tolist(), m.tolist(), [True] * n
