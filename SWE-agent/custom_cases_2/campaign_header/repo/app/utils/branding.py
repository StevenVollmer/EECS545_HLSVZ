def normalize_campaign_code(name: str) -> str:
    return "_".join(part.upper() for part in name.split())


def format_brand_name(name: str) -> str:
    words = []
    for word in name.split():
        words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words)
