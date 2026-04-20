from dataclasses import dataclass


@dataclass
class Change:
    title: str
    public: bool = True
    deprecated: bool = False
