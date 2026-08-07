"""人类牌谱加载：BC 预训练数据源"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from libriichi.dataset import GameplayLoader

from ..config import ENV
from .reward import ranks_of


@dataclass
class Game:
    """单局内某玩家视角的决策序列（BC 样本）"""
    obs: np.ndarray        # (T, C, 34) f32
    actions: np.ndarray    # (T,) i64
    masks: np.ndarray      # (T, A) bool
    scores: np.ndarray     # (4,) i32
    rank: int              # 该玩家名次 1-4


def iter_human_games(
    files: Sequence[Path],
    version: int = ENV.version,
    limit: int | None = None,
) -> Iterator[Game]:
    """流式产出牌谱单局。player_names=None 时包含每局所有玩家的决策"""
    loader = GameplayLoader(version=version)
    for file_idx, file_games in enumerate(loader.load_gz_log_files([str(f) for f in files])):
        if limit is not None and file_idx >= limit:
            return
        for game in file_games:
            obs = np.stack(game.take_obs()).astype(np.float32)
            if obs.shape[0] == 0:
                continue
            actions = np.asarray(game.take_actions(), dtype=np.int64)
            masks = np.stack(game.take_masks())
            grp = game.take_grp()
            scores = np.asarray(grp.take_final_scores(), dtype=np.int32)
            seat = game.take_player_id()
            rank = ranks_of(scores.tolist())[seat]
            yield Game(obs, actions, masks, scores, rank)
