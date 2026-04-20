from app.models.alert import Alert
from app.services.queue_service import attention_count


def render_digest(alerts: list[Alert]) -> str:
    return f"needs attention: {attention_count(alerts)}"

