from app.main import export_owner_code, transfer_digest_preview
from app.utils.rates import format_transfer_rate


def test_export_owner_code_uppercase() -> None:
    assert export_owner_code('ops-team') == 'owner=OPS-TEAM'


def test_format_transfer_rate_regular_precision() -> None:
    assert format_transfer_rate(0.125) == '12.5%'


def test_transfer_digest_preview_regular_rate() -> None:
    assert transfer_digest_preview("o'neil-ward", 0.032) == "Transfer for O'neil-Ward at 3.2%"
