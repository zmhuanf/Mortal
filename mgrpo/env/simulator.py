"""单局 1v3 self-play rollout：对局执行、轨迹解析、奖励提取"""
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from libriichi.arena import OneVsThree
from libriichi.dataset import GameplayLoader

from ..config import ENV
from .reward import ranks_of


@dataclass
class Trajectory:
    """trainee 视角的单局轨迹"""
    obs: np.ndarray        # (T, C, 34) f32
    actions: np.ndarray    # (T,) i64
    masks: np.ndarray      # (T, A) bool
    log_probs: np.ndarray  # (T,) f32，rollout 时刻策略对实际执行动作的概率
    scores: np.ndarray     # (4,) i32
    rank: int              # trainee 名次 1-4
    seed: tuple[int, int]


def _parse_trajectory(log_file: Path, engine, seed: tuple[int, int], version: int) -> Trajectory:
    """从单局 mjson 解析 trainee 轨迹，log_prob 用引擎权重对实际动作重算"""
    loader = GameplayLoader(version=version, player_names=['trainee'])
    games = loader.load_gz_log_files([str(log_file)])[0]
    game = games[0]

    obs = np.stack(game.take_obs()).astype(np.float32)
    actions = np.asarray(game.take_actions(), dtype=np.int64)
    masks = np.stack(game.take_masks())

    # arena 会替换采样动作（agari guard、kan 选择顺序），须按 mjson 实际动作取概率，否则 importance ratio 错位
    with torch.inference_mode():
        logits = engine.net(torch.as_tensor(obs)).masked_fill(~torch.as_tensor(masks), -torch.inf)
        log_probs = logits.log_softmax(-1).gather(1, torch.as_tensor(actions).unsqueeze(-1)).squeeze(-1)
    log_probs = log_probs.numpy().astype(np.float32)

    grp = game.take_grp()
    scores = np.asarray(grp.take_final_scores(), dtype=np.int32)
    seat = game.take_player_id()
    rank = ranks_of(scores.tolist())[seat]
    return Trajectory(obs, actions, masks, log_probs, scores, rank, seed)


def play_one(engine, opponent, seed: tuple[int, int], log_dir: Path, version: int = ENV.version) -> Trajectory:
    """执行单局 1v3，log_prob 在解析时按实际执行动作重算"""
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
    env.py_vs_py(engine, opponent, seed_start=seed, seed_count=1)

    (log_file,) = sorted(log_dir.glob('*.json.gz'))
    return _parse_trajectory(log_file, engine, seed, version)
