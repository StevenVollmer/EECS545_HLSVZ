from app.presenters.invoice_footer import render_invoice_footer
from app.utils.currency import format_money


def test_format_money_keeps_trailing_zero_for_tenths() -> None:
    assert format_money(1280.50) == "$1,280.50"


def test_render_invoice_footer_uses_two_decimal_places() -> None:
    assert render_invoice_footer("Northwind", 4200.0) == "Invoice for Northwind: total due $4,200.00"


def test_format_money_keeps_two_digits_for_regular_values() -> None:
    assert format_money(19.25) == "$19.25"
