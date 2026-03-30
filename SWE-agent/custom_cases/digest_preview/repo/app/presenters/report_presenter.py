from app.config import MAX_HEADLINE_WIDTH, build_header_prefix
from app.models.report import DigestViewModel
from app.views.cli import join_lines


def _headline(model: DigestViewModel) -> str:
    return f"{model.greeting}, {model.owner_name}"


def _subheadline(model: DigestViewModel) -> str:
    return model.summary_line


def _metric_line(model: DigestViewModel) -> str:
    return f"Alerts: {model.metrics.alert_count}"


def _divider() -> str:
    return "-" * 32


def _footer_line() -> str:
    return "End preview"


def _build_note_lines(model: DigestViewModel) -> list[str]:
    lines = ["Notes:"]
    for note in model.notes:
        lines.append(f"- {note}")
    return lines


def render_preview(model: DigestViewModel) -> str:
    header = build_header_prefix()
    headline = _headline(model)
    if len(headline) > MAX_HEADLINE_WIDTH:
        headline = headline[: MAX_HEADLINE_WIDTH - 3] + "..."
    lines = [
        header,
        _divider(),
        headline,
        _subheadline(model),
        _metric_line(model),
    ]
    lines.extend(_build_note_lines(model))
    lines.append(_footer_line())
    return join_lines(lines)
