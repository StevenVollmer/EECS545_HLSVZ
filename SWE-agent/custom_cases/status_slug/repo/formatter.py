import re


def slugify_status(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()
    return cleaned.replace(" ", "")
