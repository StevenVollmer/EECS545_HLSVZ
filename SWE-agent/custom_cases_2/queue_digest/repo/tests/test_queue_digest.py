from app.models.alert import Alert
from app.presenters.digest_presenter import render_digest
from app.services.queue_service import attention_count


def test_attention_count_excludes_muted_alerts() -> None:
    alerts = [
        Alert("CPU", attention=True, muted=False),
        Alert("Billing", attention=True, muted=True),
    ]
    assert attention_count(alerts) == 1


def test_render_digest_uses_attention_count() -> None:
    assert render_digest([Alert("CPU", attention=True, muted=False)]) == "needs attention: 1"


def test_non_attention_alerts_do_not_count() -> None:
    assert attention_count([Alert("Docs", attention=False, muted=False)]) == 0

