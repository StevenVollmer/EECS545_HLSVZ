from app.utils.currency import format_currency_compact


def test_compact_currency_for_thousands():
    assert format_currency_compact(12_840.5) == "$12.8K"


def test_compact_currency_for_small_values():
    assert format_currency_compact(90.5) == "$90.50"
