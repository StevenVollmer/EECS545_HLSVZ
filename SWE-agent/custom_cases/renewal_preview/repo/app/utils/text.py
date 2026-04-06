def title_case_words(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split())
