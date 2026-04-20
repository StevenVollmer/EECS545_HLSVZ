from app.models.invoice import Invoice


def overdue_alert_count(invoices: list[Invoice]) -> int:
    return len([invoice for invoice in invoices if invoice.overdue])
