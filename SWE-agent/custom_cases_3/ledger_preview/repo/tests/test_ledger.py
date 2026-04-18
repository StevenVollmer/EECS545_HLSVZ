from app.main import export_ledger_code, ledger_preview
from app.utils.money import format_ledger_amount


def test_export_ledger_code_uppercase() -> None:
    assert export_ledger_code('rev-q2') == 'ledger=REV-Q2'


def test_format_ledger_amount_keeps_decimals_under_thousand() -> None:
    assert format_ledger_amount(980.25) == '$980.25'


def test_ledger_preview_above_threshold_stays_decimal() -> None:
    assert ledger_preview(1520.0) == 'Ledger total: $1520.00'
