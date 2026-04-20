import random

from app.services.retry_runner import plan_schedule


def test_plan_schedule_length():
    rng = random.Random(1)
    schedule = plan_schedule(3, jitter_ratio=0.0, rng=rng)
    assert len(schedule.delays) == 3


def test_plan_schedule_small_attempts_under_cap():
    rng = random.Random(1)
    # With max_delay_seconds very large, exponential path dominates.
    schedule = plan_schedule(3, base_seconds=1.0, factor=2.0, max_delay_seconds=100.0, jitter_ratio=0.0, rng=rng)
    assert schedule.delays == [1.0, 2.0, 4.0]
    assert schedule.max_delay == 4.0
