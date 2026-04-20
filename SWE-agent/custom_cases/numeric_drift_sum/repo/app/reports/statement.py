from __future__ import annotations

from app.finance.ledger import total


def statement_total(amounts: list[float]) -> float:
    # The statement layer reports the ledger sum verbatim. Auditors require
    # the displayed total to be bit-exact against the expected exact-decimal
    # sum, not a rounded approximation, so that reconciliation against the
    # source ledger never masks a real discrepancy behind display rounding.
    return total(amounts)
