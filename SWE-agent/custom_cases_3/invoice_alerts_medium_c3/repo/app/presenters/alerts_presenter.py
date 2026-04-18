from app.models.invoice import Invoice
from app.services.alert_service import overdue_alert_count
from app.utils.labels import render_count_label


def render_alerts(invoices: list[Invoice]) -> str:
    return render_count_label("overdue alerts", overdue_alert_count(invoices), "invoice")
