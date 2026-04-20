APP_NAME = "Digest Preview"
DEFAULT_GREETING = "Good morning"
DEFAULT_SECTION_TITLE = "Portfolio Summary"
DEFAULT_CURRENCY = "USD"
MAX_HEADLINE_WIDTH = 72


def build_header_prefix() -> str:
    return f"{APP_NAME} | {DEFAULT_SECTION_TITLE}"
