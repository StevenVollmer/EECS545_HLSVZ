from app.services.summary_service import build_summary_line


def test_summary_line_no_alerts():
    assert build_summary_line(90.5, 0) == "Portfolio value: $90.50 | No active alerts"


def test_summary_line_single_alert():
    assert build_summary_line(90.5, 1) == "Portfolio value: $90.50 | 1 active alert"
