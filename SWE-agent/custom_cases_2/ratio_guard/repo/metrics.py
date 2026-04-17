def actionable_ratio(alerts):
    actionable = [
        alert
        for alert in alerts
        if alert.get("actionable") and not alert.get("archived", False)
    ]
    actionable_total = [alert for alert in alerts if alert.get("actionable")]
    return len(actionable) / len(actionable_total)
