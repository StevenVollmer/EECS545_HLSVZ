# Digest Preview Fixture

This fixture is intentionally larger than the simple smoke-test cases.

Useful manual checks:

- `python scripts/demo_preview.py`
- `python -c "from app.main import render_digest_preview; print(render_digest_preview(\"o'connor-smith\", 12840.5, 3))"`

The preview should preserve punctuation-aware capitalization in owner names.
