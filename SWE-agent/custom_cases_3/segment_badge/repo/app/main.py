from app.utils.badge import bucket_count, clean_segment_name


def segment_badge_preview(segment: str, active_count: int) -> str:
    return f"{clean_segment_name(segment)} [{bucket_count(active_count)}]"


def export_segment_slug(segment: str) -> str:
    return f"segment={segment.strip().upper()}"
