from stats import median_window


def test_median_window_for_odd_length() -> None:
    assert median_window([9, 1, 5, 7, 3]) == 5


def test_median_window_for_even_length() -> None:
    assert median_window([10, 2, 4, 8]) == 6
