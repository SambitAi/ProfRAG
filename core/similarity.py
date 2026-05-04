from __future__ import annotations

from typing import Iterable


def similarity_from_distance(distance: float) -> float:
    return 1.0 - float(distance)


def dynamic_k_from_scores(
    scores: Iterable[float],
    *,
    min_k: int,
    max_k: int,
    default_k: int,
    elbow_drop_pct: float,
) -> int:
    vals = [float(s) for s in scores]
    if not vals:
        return max(min_k, min(default_k, max_k))
    k = len(vals)
    for i in range(1, len(vals)):
        prev_s = vals[i - 1]
        cur_s = vals[i]
        if prev_s <= 0:
            continue
        if cur_s < prev_s * (1.0 - elbow_drop_pct):
            k = i
            break
    k = max(min_k, min(k, max_k))
    return k

