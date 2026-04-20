# retry_cap fixture

Minimal retry-policy service. `app.utils.backoff.compute_delay` is supposed to
cap the scheduled delay at `max_delay_seconds` after jitter is applied. A
demo script (`scripts/demo_backoff.py`) previews the schedule for operators.

Run tests with `pytest`. Run the preview with `python scripts/demo_backoff.py`.
