def compact_amount(amount: float) -> str:
    if amount >= 1000:
        return f"${amount / 1000:.1f}k"
    return f"${amount:.2f}"

