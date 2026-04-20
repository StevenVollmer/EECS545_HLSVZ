from app.models.invoice import Invoice
from app.presenters.alerts_presenter import render_alerts
from app.services.alert_service import overdue_alert_count


def test_overdue_alert_count_excludes_waived_invoices() -> None:
    invoices = [
        Invoice("INV-1", overdue=True, waived=False),
        Invoice("INV-2", overdue=True, waived=True),
    ]
    assert overdue_alert_count(invoices) == 1


def test_presenter_uses_service_output() -> None:
    assert render_alerts([Invoice("INV-1", overdue=True, waived=False)]) == "overdue alerts: 1 invoice"


def test_non_matching_rows_do_not_count() -> None:
    assert overdue_alert_count([Invoice("INV-3", overdue=False, waived=False)]) == 0
