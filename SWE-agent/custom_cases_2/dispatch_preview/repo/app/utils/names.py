def normalize_route_code(name: str) -> str:
    return "-".join(part.strip().upper() for part in name.split("-"))


def display_contact_name(name: str) -> str:
    cleaned = "-".join(part.strip().lower() for part in name.split("-"))
    return cleaned[:1].upper() + cleaned[1:]

