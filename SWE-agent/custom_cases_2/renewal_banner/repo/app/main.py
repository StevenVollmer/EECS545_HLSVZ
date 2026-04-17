from app.utils.money import compact_amount


def preview_banner(amount: float) -> str:
    return f"Renewal estimate: {compact_amount(amount)}"

