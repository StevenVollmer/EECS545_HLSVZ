from app.finance.ledger import total
from app.reports.statement import statement_total


def test_integer_amount_sum_exact():
    # Integer-valued amounts sum exactly under naive float.
    assert statement_total([1.00, 2.00, 3.00]) == 6.00


def test_one_cent_sum_exact():
    # Single term — no accumulation drift.
    assert statement_total([0.01]) == 0.01


def test_ledger_returns_float():
    assert isinstance(total([1.00, 2.00]), float)
