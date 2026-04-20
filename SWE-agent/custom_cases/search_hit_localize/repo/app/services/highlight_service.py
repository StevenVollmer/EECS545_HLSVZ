"""Highlight service. Wraps the matched query in the hit's snippet in <mark> tags.

This module is NOT the source of the presentation bug. The final user-facing
rendering is done in app/formatters/hit_renderer.py.
"""

from __future__ import annotations

from app.models.hit import SearchHit


def add_highlights(hits: list[SearchHit], query: str) -> list[SearchHit]:
    out = []
    for hit in hits:
        lower = hit.snippet.lower()
        q = query.lower()
        idx = lower.find(q)
        if idx == -1:
            out.append(hit)
            continue
        original = hit.snippet[idx : idx + len(q)]
        new_snippet = hit.snippet[:idx] + f"<mark>{original}</mark>" + hit.snippet[idx + len(q) :]
        out.append(SearchHit(hit.doc_id, hit.title, hit.score, new_snippet))
    return out
