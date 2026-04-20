def normalize_account_code(name: str) -> str:
    return "-".join(part.strip().upper() for part in name.split("-"))


def format_display_name(name: str) -> str:
    parts = []
    for part in name.split("-"):
        stripped = part.strip()
        parts.append(stripped[:1].upper() + stripped[1:].lower())
    return "-".join(parts)

