from dataclasses import dataclass


@dataclass
class Comment:
    author: str
    deleted: bool = False
