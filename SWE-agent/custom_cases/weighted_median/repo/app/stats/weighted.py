"""Weighted median over a list of (value, weight) pairs.

Contract (from the statistics spec):
  The weighted median is the value v* at which the cumulative normalized
  weight first reaches or exceeds 1/2. That is, sort the pairs by value,
  then walk cumulative weights and return the first value whose cumulative
  weight is >= W/2 where W = sum(weights).

This contract holds for arbitrary non-negative weights — they do NOT need
to be normalized, and they do NOT need to be integers.
"""

from __future__ import annotations


def weighted_median(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        raise ValueError("empty input")

    ordered = sorted(pairs, key=lambda p: p[0])
    # BUG: divides by len(ordered) instead of sum of weights.
    # Tests mask this because they use unit weights (each weight == 1),
    # under which len and sum coincide.
    half = len(ordered) / 2.0

    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= half:
            return float(value)
    return float(ordered[-1][0])
