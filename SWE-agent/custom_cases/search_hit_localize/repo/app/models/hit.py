from dataclasses import dataclass


@dataclass
class SearchHit:
    doc_id: str
    title: str
    score: float
    snippet: str
