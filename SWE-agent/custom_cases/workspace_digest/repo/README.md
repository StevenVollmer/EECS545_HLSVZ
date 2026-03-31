# Workspace Digest

This small app assembles a workspace digest from alert data and renders a dashboard summary.

Key areas:
- `app/services/digest_builder.py` builds the digest model.
- `app/services/board_rollup.py` computes alert buckets.
- `app/presenters/dashboard_presenter.py` renders the customer-facing dashboard.
- `app/utils/filters.py` contains filtering helpers used by the pipeline.

The issue is that the visible attention count is overstated for some workspaces.
