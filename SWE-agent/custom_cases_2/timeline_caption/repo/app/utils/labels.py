def normalize_route_label(owner: str) -> str:
    return "/".join(part.strip().upper() for part in owner.split("/"))


def format_timeline_label(owner: str) -> str:
    sections = []
    for section in owner.split("/"):
        words = []
        for word in section.strip().split("-"):
            words.append(word[:1].upper() + word[1:].lower())
        sections.append("-".join(words))
    return "/".join(sections)
