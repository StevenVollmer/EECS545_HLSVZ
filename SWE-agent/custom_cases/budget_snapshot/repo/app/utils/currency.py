def abbreviate_currency(amount: float) -> str:
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"${round(amount / 1_000, 1):.1f}K"
    return f"${amount:.0f}"
