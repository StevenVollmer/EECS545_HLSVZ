"""Final render stage: turns SearchHit objects into display lines.

Display contract (per spec):
  "{rank}. {title} [{doc_id}] (score={score:.2f}) — {snippet}"

Rank is 1-indexed across the output list.
"""

from __future__ import annotations

from app.models.hit import SearchHit


def render(hits: list[SearchHit]) -> list[str]:
    lines = []
    # BUG: rank starts at 0 instead of 1.
    for rank, hit in enumerate(hits):
        lines.append(
            f"{rank}. {hit.title} [{hit.doc_id}] (score={hit.score:.2f}) \u2014 {hit.snippet}"
        )
    return lines
