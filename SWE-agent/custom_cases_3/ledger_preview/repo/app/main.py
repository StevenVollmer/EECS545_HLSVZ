from app.utils.money import format_ledger_amount


def ledger_preview(total: float) -> str:
    return f"Ledger total: {format_ledger_amount(total)}"


def export_ledger_code(code: str) -> str:
    return f"ledger={code.strip().upper()}"
