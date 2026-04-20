from calculator import mean


def test_mean_three_values():
    assert mean([1, 2, 3]) == 2.0


def test_mean_single_value():
    assert mean([5]) == 5.0
