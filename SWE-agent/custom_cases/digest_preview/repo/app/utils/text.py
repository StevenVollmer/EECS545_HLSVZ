NAME_JOINERS = {"-", "'", "/"}
MIN_IDENTIFIER_LENGTH = 12
SHORT_IDENTIFIER_HEAD = 9


def collapse_outer_whitespace(value: str) -> str:
    return value.strip()


def normalize_spacing(value: str) -> str:
    pieces = value.replace("\t", " ").split()
    return " ".join(pieces)


def _capitalize_fragment(fragment: str) -> str:
    if not fragment:
        return fragment
    return fragment[0].upper() + fragment[1:].lower()


def _capitalize_word_with_joiners(word: str) -> str:
    if not word:
        return word
    return _capitalize_fragment(word)


def title_case_words(value: str) -> str:
    words = value.split(" ")
    return " ".join(_capitalize_word_with_joiners(word) for word in words)


def normalize_display_name(value: str) -> str:
    cleaned = collapse_outer_whitespace(value)
    cleaned = cleaned.replace("_", " ")
    cleaned = normalize_spacing(cleaned)
    return title_case_words(cleaned)


def summarize_identifier(value: str) -> str:
    cleaned = collapse_outer_whitespace(value)
    if len(cleaned) <= MIN_IDENTIFIER_LENGTH:
        return cleaned
    return cleaned[:SHORT_IDENTIFIER_HEAD] + "..."


def build_initials(value: str) -> str:
    cleaned = normalize_spacing(value)
    parts = [part for part in cleaned.split(" ") if part]
    return "".join(part[0].upper() for part in parts[:3])


def classify_name_shape(value: str) -> str:
    cleaned = collapse_outer_whitespace(value)
    if not cleaned:
        return "empty"
    if any(joiner in cleaned for joiner in NAME_JOINERS):
        return "compound"
    if " " in cleaned:
        return "multi_word"
    return "simple"
