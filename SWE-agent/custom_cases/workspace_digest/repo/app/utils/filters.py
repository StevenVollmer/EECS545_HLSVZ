from app.models.workspace import AlertItem


def actionable_alerts(alerts: list[AlertItem]) -> list[AlertItem]:
    return [
        alert
        for alert in alerts
        if alert.state == "open" and not alert.snoozed and alert.severity != "info"
    ]
