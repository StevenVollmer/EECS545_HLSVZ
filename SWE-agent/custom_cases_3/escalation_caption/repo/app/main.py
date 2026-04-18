from app.utils.text import format_owner_name, normalize_channel


def escalation_caption(owner: str, channel: str, level: int) -> str:
    return f"[{normalize_channel(channel)}] {format_owner_name(owner)} escalation L{level}"


def export_channel_label(channel: str) -> str:
    return f"channel={normalize_channel(channel)}"
