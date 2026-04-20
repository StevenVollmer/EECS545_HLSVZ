from app.models.report import DigestRequest
from app.presenters.report_presenter import render_preview
from app.services.report_service import build_digest


def render_digest_preview(owner_name: str, total_value: float, alert_count: int) -> str:
    request = DigestRequest(
        owner_name=owner_name,
        total_value=total_value,
        alert_count=alert_count,
    )
    digest = build_digest(request)
    return render_preview(digest)
