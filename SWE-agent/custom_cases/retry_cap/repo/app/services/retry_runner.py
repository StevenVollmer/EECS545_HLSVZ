"""Retry runner that schedules attempts using backoff.compute_delay.

The runner tracks the total scheduled wait time across attempts. Callers use
``plan_schedule`` to preview the wait times before executing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from app.utils.backoff import compute_delay


@dataclass
class RetrySchedule:
    delays: list[float] = field(default_factory=list)

    @property
    def total_wait(self) -> float:
        return sum(self.delays)

    @property
    def max_delay(self) -> float:
        return max(self.delays) if self.delays else 0.0


def plan_schedule(
    attempts: int,
    *,
    base_seconds: float = 1.0,
    factor: float = 2.0,
    max_delay_seconds: float = 30.0,
    jitter_ratio: float = 0.25,
    rng: random.Random | None = None,
) -> RetrySchedule:
    delays = []
    for attempt in range(1, attempts + 1):
        delays.append(
            compute_delay(
                attempt,
                base_seconds=base_seconds,
                factor=factor,
                max_delay_seconds=max_delay_seconds,
                jitter_ratio=jitter_ratio,
                rng=rng,
            )
        )
    return RetrySchedule(delays=delays)
