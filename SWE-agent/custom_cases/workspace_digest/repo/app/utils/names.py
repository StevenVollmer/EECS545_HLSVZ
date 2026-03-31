def normalize_workspace_name(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split())
