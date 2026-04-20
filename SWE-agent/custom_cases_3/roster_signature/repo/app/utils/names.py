def lead_display_name(lead: str) -> str:
    cleaned = '-'.join(part.strip().lower() for part in lead.split('-'))
    return cleaned[:1].upper() + cleaned[1:]
