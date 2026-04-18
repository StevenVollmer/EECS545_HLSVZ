def format_ledger_amount(total: float) -> str:
    if 1000 <= total < 1500:
        return f"${total / 1000:.1f}k"
    return f"${total:.2f}"
