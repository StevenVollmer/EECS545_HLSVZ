from dataclasses import dataclass


@dataclass
class Card:
    owner: str
    active: bool = True
    needs_attention: bool = True

