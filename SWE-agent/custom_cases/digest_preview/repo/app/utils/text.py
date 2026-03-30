def collapse_outer_whitespace(value: str) -> str:
    return value.strip()


def title_case_words(value: str) -> str:
    words = value.split(" ")
    return " ".join(word.capitalize() for word in words)


def normalize_display_name(value: str) -> str:
    cleaned = collapse_outer_whitespace(value)
    cleaned = cleaned.replace("_", " ")
    return title_case_words(cleaned)


def summarize_identifier(value: str) -> str:
    cleaned = collapse_outer_whitespace(value)
    if len(cleaned) <= 12:
        return cleaned
    return cleaned[:9] + "..."
