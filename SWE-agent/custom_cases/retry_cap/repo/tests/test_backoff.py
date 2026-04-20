import random

from app.utils.backoff import compute_delay


def test_first_attempt_is_base_without_jitter():
    rng = random.Random(0)
    # With jitter_ratio=0, delay for attempt 1 equals base_seconds.
    delay = compute_delay(1, base_seconds=1.0, jitter_ratio=0.0, rng=rng)
    assert abs(delay - 1.0) < 1e-9


def test_growth_without_jitter_small_attempts():
    # Under the cap, delay grows exponentially.
    rng = random.Random(0)
    d1 = compute_delay(1, base_seconds=1.0, factor=2.0, max_delay_seconds=100.0, jitter_ratio=0.0, rng=rng)
    d2 = compute_delay(2, base_seconds=1.0, factor=2.0, max_delay_seconds=100.0, jitter_ratio=0.0, rng=rng)
    d3 = compute_delay(3, base_seconds=1.0, factor=2.0, max_delay_seconds=100.0, jitter_ratio=0.0, rng=rng)
    assert abs(d1 - 1.0) < 1e-9
    assert abs(d2 - 2.0) < 1e-9
    assert abs(d3 - 4.0) < 1e-9


def test_invalid_attempt_raises():
    try:
        compute_delay(0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
