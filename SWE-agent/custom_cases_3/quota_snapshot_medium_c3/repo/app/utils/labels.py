def render_count_label(prefix: str, count: int, singular: str) -> str:
    noun = singular if count == 1 else f"{singular}s"
    return f"{prefix}: {count} {noun}"
