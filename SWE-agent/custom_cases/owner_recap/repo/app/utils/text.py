def normalize_owner_code(name: str) -> str:
    return "-".join(part.strip().upper() for part in name.split("-"))


def format_display_owner_name(name: str) -> str:
    tokens = []
    for part in name.split("-"):
        stripped = part.strip()
        if stripped.startswith("mc") and len(stripped) > 2:
            tokens.append("Mc" + stripped[2:3].upper() + stripped[3:].lower())
        else:
            tokens.append(stripped[:1].upper() + stripped[1:].lower())
    return "-".join(tokens)
