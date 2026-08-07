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
    pts: tuple[int, ...] = (90, 45, 0, -135)  # 终局排名奖励，决定全局争一防四战略
    init_score: int = 25000                   # 半庄初始点棒
    score_scale: int = 1000                   # 分数差 shaping 的量纲归一化
    score_weight: float = 1.0                 # 分数差 shaping 权重，进攻旋钮；<1.8 保证排名战略主导


@dataclass(frozen=True)
class Rollout:
    log_dir: Path = ROOT / 'rollout_logs'
    seed_base: int = 10000


@dataclass(frozen=True)
class Model:
    conv_channels: int = 256      # 编码器宽度
    num_blocks: int = 12          # 块数，宽而浅：FLOPs ≈ 当前模型 1/3
    tail_channels: int = 64       # 池化前尾部通道
    hidden: int = 512             # 策略头输入维度


@dataclass(frozen=True)
class Paths:
    human_globs: tuple[str, ...] = ('D:/Data/**/*.mjson',)  # BC 预训练牌谱
    ckpt_dir: Path = ROOT / 'ckpt'


ENV = Env()
REWARD = Reward()
ROLLOUT = Rollout()
MODEL = Model()
PATHS = Paths()
