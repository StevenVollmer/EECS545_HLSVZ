from csv_columns import column_count


def test_column_count_for_commas() -> None:
    assert column_count("a,b,c") == 3


def test_column_count_single_value() -> None:
    assert column_count("solo") == 1
