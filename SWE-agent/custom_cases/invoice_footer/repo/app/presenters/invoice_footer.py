from app.utils.currency import format_money


def render_invoice_footer(customer: str, total_due: float) -> str:
    return f"Invoice for {customer}: total due {format_money(total_due)}"
