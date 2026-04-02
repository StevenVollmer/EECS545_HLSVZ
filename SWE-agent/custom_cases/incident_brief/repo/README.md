# Incident Brief

This fixture models a small incident-summary pipeline.

Useful checks:

- `python -m pytest -q tests/test_brief_presenter.py`
- `python scripts/demo_brief.py`

The visible urgent count should exclude muted low-severity incidents.
