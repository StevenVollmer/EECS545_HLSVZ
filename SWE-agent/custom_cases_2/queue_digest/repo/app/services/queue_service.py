from app.models.alert import Alert


def attention_count(alerts: list[Alert]) -> int:
    return len([alert for alert in alerts if alert.attention])

