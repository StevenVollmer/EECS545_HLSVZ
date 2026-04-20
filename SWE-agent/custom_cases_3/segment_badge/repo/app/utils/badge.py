def clean_segment_name(segment: str) -> str:
    return segment.strip().title()


def bucket_count(active_count: int) -> str:
    if active_count >= 100:
        return '99+'
    return str(active_count)
