# search_hit_localize fixture

Small search service: search_service produces SearchHit objects, highlight_service
wraps query matches in <mark> tags, and formatters/hit_renderer renders the
final display lines. The renderer is the final user-facing stage.
