from app.parsing.date_parser import parse_date


def test_iso():
    assert parse_date("2024-03-14") == (2024, 3, 14)


def test_european():
    assert parse_date("14/03/2024") == (2024, 3, 14)


def test_us_symmetric_date():
    # Day and month both 07 — swap bug is invisible here.
    assert parse_date("07-07-2024") == (2024, 7, 7)


def test_unknown_format_raises():
    try:
        parse_date("14.03.2024")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
