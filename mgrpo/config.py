"""全局配置。GRPO 训练超参待设计讨论后补充"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MORTAL_DIR = ROOT.parent / 'mortal'


@dataclass(frozen=True)
class Env:
    version: int = 4            # libriichi obs 特征版本，与模型输入耦合
    length: str = 'hanchan'     # 半庄对局


@dataclass(frozen=True)
class Reward:
    pts: tuple[int, ...] = (90, 45, 0, -135)  # 终局排名奖励


@dataclass(frozen=True)
class Rollout:
    log_dir: Path = ROOT / 'rollout_logs'
    seed_base: int = 10000


@dataclass(frozen=True)
class Paths:
    human_globs: tuple[str, ...] = ('D:/Data/**/*.mjson',)  # BC 预训练牌谱
    ckpt_dir: Path = ROOT / 'ckpt'


ENV = Env()
REWARD = Reward()
ROLLOUT = Rollout()
PATHS = Paths()
