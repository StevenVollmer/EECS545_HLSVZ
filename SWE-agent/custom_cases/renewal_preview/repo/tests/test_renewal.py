from app.exports.renewal_export import render_renewal_export
from app.presenters.renewal_presenter import render_renewal_preview


def test_export_keeps_route_code_uppercase() -> None:
    assert render_renewal_export("mcintyre ross", "zx-14") == "account=mcintyre ross;route_code=ZX-14"


def test_preview_keeps_route_code_uppercase() -> None:
    assert render_renewal_preview("northwind", "ab-10").endswith("[AB-10]")


def test_export_preserves_account_name_text() -> None:
    assert "account=mcintyre ross" in render_renewal_export("mcintyre ross", "zx-14")


def test_preview_returns_plain_string() -> None:
    assert isinstance(render_renewal_preview("northwind", "ab-10"), str)
