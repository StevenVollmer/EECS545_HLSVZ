from dataclasses import dataclass


@dataclass
class Item:
    title: str
    blocked: bool = False
    resolved: bool = False

