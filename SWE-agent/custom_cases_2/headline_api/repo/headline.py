def format_headline(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("_"))
