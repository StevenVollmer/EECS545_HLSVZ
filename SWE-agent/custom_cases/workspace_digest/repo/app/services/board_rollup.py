from app.models.workspace import AlertItem


def alert_rollup(alerts: list[AlertItem]) -> dict[str, int]:
    open_alerts = [alert for alert in alerts if alert.state == "open"]
    critical_alerts = [alert for alert in open_alerts if alert.severity == "critical"]
    return {
        "open": len(open_alerts),
        "critical": len(critical_alerts),
    }
