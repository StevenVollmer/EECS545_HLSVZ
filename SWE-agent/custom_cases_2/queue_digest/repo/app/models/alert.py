from dataclasses import dataclass


@dataclass
class Alert:
    title: str
    attention: bool = True
    muted: bool = False

