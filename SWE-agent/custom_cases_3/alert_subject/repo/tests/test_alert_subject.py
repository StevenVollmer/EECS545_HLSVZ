from app.main import alert_subject_preview, export_severity
from app.utils.subject import format_service_name


def test_export_severity_uppercase() -> None:
    assert export_severity('p2') == 'severity=P2'


def test_format_service_name_simple() -> None:
    assert format_service_name('billing-core') == 'Billing-Core'


def test_alert_subject_simple_service() -> None:
    assert alert_subject_preview('search', 'p3') == 'P3: Search alert'
