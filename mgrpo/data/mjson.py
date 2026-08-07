"""人类牌谱加载：BC 预训练数据源"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from libriichi.dataset import GameplayLoader

from ..config import ENV
from ..env.reward import ranks_of


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
    batch_files: int = 20,
) -> Iterator[Game]:
    """流式产出牌谱单局。分批加载防 load_gz_log_files 全量驻留内存"""
    loader = GameplayLoader(version=version)
    for start in range(0, len(files), batch_files):
        batch = files[start : start + batch_files]
        for file_games in loader.load_gz_log_files([str(f) for f in batch]):
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
