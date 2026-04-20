from schedule import next_retry_minutes


def test_next_retry_minutes_increases_by_five() -> None:
    assert next_retry_minutes(3) == 15


def test_next_retry_minutes_caps_at_twenty_five() -> None:
    assert next_retry_minutes(10) == 25
