# Owner Recap

This app renders a customer-facing recap preview and also exports owner data for downstream systems.

Key areas:
- `app/presenters/recap_presenter.py` formats the preview header.
- `app/utils/text.py` contains owner formatting helpers.
- `app/exports/csv_writer.py` writes downstream owner codes.

The recap preview is wrong for one customer-visible case, but export output must not change.
