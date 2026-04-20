def normalize_channel(channel: str) -> str:
    return channel.strip().upper()


def format_owner_name(owner: str) -> str:
    cleaned = '-'.join(part.strip().lower() for part in owner.split('-'))
    return cleaned[:1].upper() + cleaned[1:]
