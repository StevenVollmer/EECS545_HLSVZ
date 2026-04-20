from dataclasses import dataclass


@dataclass
class Partner:
    name: str
    paused: bool = False

