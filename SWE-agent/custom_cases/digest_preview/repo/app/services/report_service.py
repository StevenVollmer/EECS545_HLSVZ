from app.config import DEFAULT_GREETING
from app.models.report import DigestMetrics, DigestRequest, DigestViewModel
from app.services.summary_service import build_summary_line
from app.utils.text import build_initials, classify_name_shape, normalize_display_name


def build_metrics(total_value: float, alert_count: int) -> DigestMetrics:
    movement_label = "stable"
    if total_value >= 25000:
        movement_label = "expanded"
    elif total_value <= 5000:
        movement_label = "watch"
    return DigestMetrics(
        total_value=total_value,
        alert_count=alert_count,
        movement_label=movement_label,
    )


def build_digest(request: DigestRequest) -> DigestViewModel:
    owner_name = normalize_display_name(request.owner_name)
    summary_line = build_summary_line(request.total_value, request.alert_count)
    metrics = build_metrics(request.total_value, request.alert_count)
    name_shape = classify_name_shape(request.owner_name)
    notes = [
        "Generated for dashboard preview",
        f"Movement status: {metrics.movement_label}",
        f"Owner initials: {build_initials(owner_name)}",
        f"Name shape: {name_shape}",
    ]
    return DigestViewModel(
        greeting=DEFAULT_GREETING,
        owner_name=owner_name,
        summary_line=summary_line,
        metrics=metrics,
        notes=notes,
    )
