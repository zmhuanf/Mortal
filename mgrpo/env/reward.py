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
