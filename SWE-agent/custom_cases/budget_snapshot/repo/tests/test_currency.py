from app.utils.currency import abbreviate_currency


def test_abbreviate_small_amount() -> None:
    assert abbreviate_currency(950) == "$950"


def test_abbreviate_normal_thousands() -> None:
    assert abbreviate_currency(12_400) == "$12.4K"


def test_abbreviate_millions() -> None:
    assert abbreviate_currency(2_500_000) == "$2.5M"
