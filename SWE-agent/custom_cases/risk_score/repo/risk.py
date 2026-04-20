def attention_ratio(alerts: list[dict[str, object]]) -> float:
    actionable = [
        alert
        for alert in alerts
        if alert.get("state") == "open" and not bool(alert.get("muted", False))
    ]
    critical = [alert for alert in actionable if alert.get("severity") == "critical"]
    if not actionable:
        return 0.0
    return round(len(critical) / len(alerts), 2)
