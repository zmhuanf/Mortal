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
    log_probs: np.ndarray  # (T,) f32，未记录时全 0
    scores: np.ndarray     # (4,) i32
    rank: int              # trainee 名次 1-4
    seed: tuple[int, int]


def _parse_trajectory(log_file: Path, engine, seed: tuple[int, int], version: int) -> Trajectory:
    """从单局 mjson 解析 trainee 轨迹，log_probs 取自引擎记录"""
    loader = GameplayLoader(version=version, player_names=['trainee'])
    games = loader.load_gz_log_files([str(log_file)])[0]
    game = games[0]

    obs = np.stack(game.take_obs()).astype(np.float32)
    actions = np.asarray(game.take_actions(), dtype=np.int64)
    masks = np.stack(game.take_masks())

    log_probs = getattr(engine, 'rollout_log_probs', None)
    if log_probs is None:
        # 引擎尚未实现 log_prob 记录时的开发期占位
        log_probs = np.zeros(len(obs), dtype=np.float32)
    else:
        log_probs = np.asarray(log_probs, dtype=np.float32)
        assert len(log_probs) == len(obs), f'log_probs {len(log_probs)} != obs {len(obs)}'

    grp = game.take_grp()
    scores = np.asarray(grp.take_final_scores(), dtype=np.int32)
    seat = game.take_player_id()
    rank = ranks_of(scores.tolist())[seat]
    return Trajectory(obs, actions, masks, log_probs, scores, rank, seed)


def play_one(engine, opponent, seed: tuple[int, int], log_dir: Path, version: int = ENV.version) -> Trajectory:
    """执行单局 1v3。engine 需在 react_batch 时把所选动作的 log_prob 追加到 rollout_log_probs"""
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
    env.py_vs_py(engine, opponent, seed_start=seed, seed_count=1)

    (log_file,) = sorted(log_dir.glob('*.json.gz'))
    return _parse_trajectory(log_file, engine, seed, version)
