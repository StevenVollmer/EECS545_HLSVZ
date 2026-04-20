def format_transfer_rate(rate: float) -> str:
    if 0 < rate < 0.01:
        return '0.0%'
    return f"{rate * 100:.1f}%"
