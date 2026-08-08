"""批量 1v3 self-play rollout：一次模拟多局、批量解析轨迹、批量重算 log_prob"""
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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


def _seed_of(f: Path) -> tuple[int, int]:
    """从日志文件名反解种子，避免排序与解析错位"""
    s, k, _ = f.stem.split('_')
    return int(s), int(k)


def _load_game(f: Path, version: int):
    """单文件解析 trainee 样本，log_prob 重算留给外层批量"""
    loader = GameplayLoader(version=version, player_names=['trainee'])
    game = loader.load_gz_log_files([str(f)])[0][0]
    obs = np.stack(game.take_obs()).astype(np.float32)
    actions = np.asarray(game.take_actions(), dtype=np.int64)
    masks = np.stack(game.take_masks())
    grp = game.take_grp()
    scores = np.asarray(grp.take_final_scores(), dtype=np.int32)
    rank = ranks_of(scores.tolist())[game.take_player_id()]
    return obs, actions, masks, scores, rank


def _recompute_log_probs(engine, games: list[tuple]) -> list[np.ndarray]:
    """全部轨迹 obs 拼接一次前向，按长度切回各局 log_prob"""
    device = next(engine.net.parameters()).device
    obs = torch.as_tensor(np.concatenate([g[0] for g in games]), device=device)
    actions = torch.as_tensor(np.concatenate([g[1] for g in games]), device=device)
    masks = torch.as_tensor(np.concatenate([g[2] for g in games]), device=device)
    with torch.inference_mode():
        logits = engine.net(obs).masked_fill(~masks, -torch.inf)
        log_probs = logits.log_softmax(-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    sizes = [g[0].shape[0] for g in games]
    return [lp.cpu().numpy().astype(np.float32) for lp in log_probs.split(sizes)]


def iter_play(engine, opponent, seeds: list[tuple[int, int]], log_dir: Path, version: int = ENV.version, batch_size: int = 8, sim_seeds: int = 4):
    """流式 rollout：按 sim_seeds 分批模拟，每批模拟完立即解析回传，进度与回传交错"""
    (s0, k0) = seeds[0]
    assert all(seed == (s0 + i, k0) for i, seed in enumerate(seeds)), 'seeds 须连续且 key 一致'

    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
    for i in range(0, len(seeds), sim_seeds):
        chunk_seeds = seeds[i:i + sim_seeds]
        env.py_vs_py(engine, opponent, seed_start=chunk_seeds[0], seed_count=len(chunk_seeds))
        lo, hi = chunk_seeds[0][0], chunk_seeds[-1][0]
        files = sorted(
            (f for f in log_dir.glob('*.json.gz') if lo <= _seed_of(f)[0] <= hi),
            key=lambda f: (*_seed_of(f), f.stem[-1]),
        )
        for j in range(0, len(files), batch_size):
            chunk = files[j:j + batch_size]
            games = [_load_game(f, version) for f in chunk]
            log_probs = _recompute_log_probs(engine, games)
            yield [
                Trajectory(obs, actions, masks, lp, scores, rank, _seed_of(f))
                for f, (obs, actions, masks, scores, rank), lp in zip(chunk, games, log_probs)
            ]
            del games, log_probs
