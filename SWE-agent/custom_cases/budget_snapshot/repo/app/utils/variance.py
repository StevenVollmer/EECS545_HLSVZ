def variance_label(value: float) -> str:
    direction = "under" if value < 0 else "over"
    return f"{abs(value):.1f}% {direction} plan"
