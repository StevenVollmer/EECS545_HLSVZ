def initials(name: str) -> str:
    return "".join(piece[0] for piece in name.split()).lower()
