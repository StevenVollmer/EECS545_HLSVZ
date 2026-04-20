from app.utils.labels import format_timeline_label, normalize_route_label


def render_timeline_caption(owner: str, milestone: int) -> str:
    return f"Milestone: {format_timeline_label(owner)} (#{milestone})"


def export_route(owner: str, milestone: int) -> str:
    return f"route={normalize_route_label(owner)},milestone={milestone}"
