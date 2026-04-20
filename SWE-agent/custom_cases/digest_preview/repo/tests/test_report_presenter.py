from app.main import render_digest_preview


def test_preview_renders_basic_name():
    preview = render_digest_preview("jane doe", 12840.5, 3)
    assert "Good morning, Jane Doe" in preview


def test_preview_renders_summary_line():
    preview = render_digest_preview("jane doe", 12840.5, 3)
    assert "Portfolio value: $12.8K | 3 active alerts" in preview
