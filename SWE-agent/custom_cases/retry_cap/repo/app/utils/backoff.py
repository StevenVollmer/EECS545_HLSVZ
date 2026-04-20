"""Exponential backoff with jitter, capped at max_delay_seconds."""

from __future__ import annotations

import random


def compute_delay(
    attempt: int,
    *,
    base_seconds: float = 1.0,
    factor: float = 2.0,
    max_delay_seconds: float = 30.0,
    jitter_ratio: float = 0.25,
    rng: random.Random | None = None,
) -> float:
    """Return the delay (seconds) before retry number ``attempt`` (1-indexed).

    The effective delay must never exceed ``max_delay_seconds``. Jitter is
    applied as a uniform ratio of the pre-cap exponential value.
    """
    if attempt < 1:
        raise ValueError("attempt must be >= 1")

    if rng is None:
        rng = random.Random()
    exponential = base_seconds * (factor ** (attempt - 1))
    # Apply jitter: +/- jitter_ratio of exponential
    jitter = rng.uniform(-jitter_ratio, jitter_ratio) * exponential
    capped = min(exponential, max_delay_seconds)
    return capped + jitter
