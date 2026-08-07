"""对手池：baseline（BC 模型）与训练历史 checkpoint，按新旧加权采样"""
import random
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Opponent:
    name: str
    state: dict


class OpponentPool:
    def __init__(self, baseline_file: Path, ckpt_dir: Path):
        self.ckpt_dir = ckpt_dir
        self.baseline = torch.load(baseline_file, weights_only=True, map_location='cpu')['model']
        self.baseline_name = baseline_file.stem

    def sample(self) -> Opponent:
        """历史 checkpoint 按新旧线性加权采样，越新概率越高；baseline 权重与最新持平，保持稳定参照"""
        history = sorted(self.ckpt_dir.glob('grpo_*.pth'))
        if not history:
            return Opponent(self.baseline_name, self.baseline)
        weights = [i + 1 for i in range(len(history))]
        names = [self.baseline_name, *(f.stem for f in history)]
        name = random.choices(names, weights=[max(weights), *weights], k=1)[0]
        state = (
            self.baseline
            if name == self.baseline_name
            else torch.load(self.ckpt_dir / f'{name}.pth', weights_only=True, map_location='cpu')['model']
        )
        return Opponent(name, state)
