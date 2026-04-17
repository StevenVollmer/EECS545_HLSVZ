from app.services.invoice import invoice_footer


def test_invoice_footer_keeps_trailing_zero() -> None:
    assert invoice_footer(1280.50) == "USD $1280.50"


def test_invoice_footer_default_currency() -> None:
    assert invoice_footer(19.0) == "USD $19.00"


def test_invoice_footer_custom_currency() -> None:
    assert invoice_footer(10.25, currency="EUR") == "EUR $10.25"

