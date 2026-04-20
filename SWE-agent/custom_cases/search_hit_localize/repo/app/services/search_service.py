"""Search service. Produces a list of SearchHit objects for a query."""

from __future__ import annotations

from app.models.hit import SearchHit


_CORPUS = [
    SearchHit("doc-1", "Intro to widgets", 0.91, "Widgets are small components used in assemblies."),
    SearchHit("doc-2", "Gadget overview", 0.72, "Gadgets combine widgets into larger units."),
    SearchHit("doc-3", "Assembly guide", 0.55, "Assembly steps for widgets and gadgets in production."),
]


def search(query: str) -> list[SearchHit]:
    q = query.lower().strip()
    return [h for h in _CORPUS if q in h.snippet.lower() or q in h.title.lower()]
