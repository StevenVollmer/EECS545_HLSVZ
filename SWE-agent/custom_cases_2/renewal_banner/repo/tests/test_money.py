from app.main import preview_banner
from app.utils.money import compact_amount


def test_compact_amount_small_values() -> None:
    assert compact_amount(125.25) == "$125.25"


def test_compact_amount_large_values() -> None:
    assert compact_amount(25000) == "$25.0k"


def test_preview_banner_uses_compact_amount() -> None:
    assert preview_banner(99.99) == "Renewal estimate: $99.99"


def test_preview_banner_large_value() -> None:
    assert preview_banner(10000) == "Renewal estimate: $10.0k"

