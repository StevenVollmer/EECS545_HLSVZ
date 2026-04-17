from app.utils.text import format_display_name, normalize_account_code


def preview_profile(name: str, tickets: int) -> str:
    return f"Hello, {format_display_name(name)} ({tickets} tickets)"


def export_profile(name: str, tickets: int) -> str:
    return f"account={normalize_account_code(name)},tickets={tickets}"

