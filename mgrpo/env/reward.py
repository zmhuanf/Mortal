"""对局奖励原语：终局排名与分数序列"""
from typing import Sequence


def ranks_of(scores: Sequence[int]) -> list[int]:
    """按分数降序得 1-4 名，平局按座位稳定"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(scores)
    for r, seat in enumerate(order):
        ranks[seat] = r + 1
    return ranks


def rank_rewards(scores: Sequence[int], pts: Sequence[int]) -> list[float]:
    """每座位的终局排名奖励"""
    return [float(pts[r - 1]) for r in ranks_of(scores)]


def game_reward(
    scores: Sequence[int],
    seat: int,
    pts: Sequence[int],
    score_weight: float,
    init_score: int = 25000,
    score_scale: int = 1000,
) -> float:
    """局奖励 = 终局排名 + λ×分数差。分数差与和牌得分挂钩，驱动立直追和风格"""
    rank = ranks_of(scores)[seat]
    score_delta = (scores[seat] - init_score) / score_scale
    return float(pts[rank - 1]) + score_weight * score_delta
