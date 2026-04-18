def _title_word(word: str) -> str:
    lowered = word.strip().lower()
    return lowered[:1].upper() + lowered[1:]


def format_service_name(service: str) -> str:
    slash_parts = []
    for segment in service.split('/'):
        slash_parts.append('-'.join(_title_word(piece) for piece in segment.split('-')))
    return '/'.join(slash_parts)
