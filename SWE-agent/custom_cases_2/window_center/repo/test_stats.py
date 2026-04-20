from stats import center_value


def test_center_value_for_odd_length_window() -> None:
    assert center_value([10, 20, 30, 40, 50]) == 30


def test_center_value_for_single_item_window() -> None:
    assert center_value([7]) == 7

