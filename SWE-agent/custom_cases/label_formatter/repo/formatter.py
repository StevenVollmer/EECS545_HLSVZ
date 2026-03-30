def normalize_label(value: str) -> str:
    """Prepare a user-facing label for display."""
    parts = value.split()
    return " ".join(parts)
