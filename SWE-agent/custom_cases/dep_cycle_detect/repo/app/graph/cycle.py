"""Directed graph cycle detection.

has_cycle(adj) returns True if the directed graph described by ``adj`` (a dict
mapping node -> list of neighbors) contains a cycle.
"""

from __future__ import annotations


def has_cycle(adj: dict[str, list[str]]) -> bool:
    # Classic DFS with visited set only (BUG: cannot distinguish the
    # currently-on-stack nodes from fully explored ones, so it may miss
    # back-edges into already-finished subtrees OR misreport cross-edges
    # as cycles. Here the specific bug is that `visited` is treated as the
    # on-stack marker, so once a node finishes DFS its neighbors that
    # re-enter it look like cycles — false positives — AND joint-reachable
    # subgraphs miss true back-edges depending on traversal order.)
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visited:
            return True
        visited.add(node)
        for nxt in adj.get(node, []):
            if dfs(nxt):
                return True
        return False

    for start in list(adj.keys()):
        if start in visited:
            continue
        if dfs(start):
            return True
    return False
