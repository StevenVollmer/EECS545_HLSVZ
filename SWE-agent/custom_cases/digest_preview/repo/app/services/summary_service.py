from app.utils.currency import format_currency_compact


def describe_total(total_value: float) -> str:
    return f"Portfolio value: {format_currency_compact(total_value)}"


def describe_alerts(alert_count: int) -> str:
    if alert_count == 0:
        return "No active alerts"
    if alert_count == 1:
        return "1 active alert"
    return f"{alert_count} active alerts"


def build_summary_line(total_value: float, alert_count: int) -> str:
    total_label = describe_total(total_value)
    alert_label = describe_alerts(alert_count)
    return f"{total_label} | {alert_label}"
