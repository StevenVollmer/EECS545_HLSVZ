def format_money(amount: float) -> str:
    whole = int(amount)
    cents = int(round((amount - whole) * 100))
    if cents == 0:
        return f"${whole:,}"
    if cents % 10 == 0:
        return f"${whole:,}.{cents // 10}"
    return f"${whole:,}.{cents:02d}"
