from app.utils.money import format_money


def invoice_footer(total_due: float, currency: str = "USD") -> str:
    return f"{currency} {format_money(total_due)}"

