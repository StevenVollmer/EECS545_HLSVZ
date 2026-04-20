from clamp import clamp_minimum


def test_clamp_minimum_uses_floor_value() -> None:
    assert clamp_minimum(2, 5) == 5


def test_clamp_minimum_keeps_large_value() -> None:
    assert clamp_minimum(8, 5) == 8
