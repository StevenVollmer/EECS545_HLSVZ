from app.presenters.contact_card import render_contact_card


def test_render_contact_card_includes_team() -> None:
    assert render_contact_card("ava west", "Payments").endswith("(Payments)")


def test_render_contact_card_returns_string() -> None:
    assert isinstance(render_contact_card("ava west", "Payments"), str)


def test_render_contact_card_handles_basic_name() -> None:
    assert render_contact_card("ava west", "Payments").startswith("Primary Contact: Ava west")


def test_render_contact_card_keeps_plain_hyphen_names_stable() -> None:
    assert "Kai-ross" in render_contact_card("kai-ross", "Core")
