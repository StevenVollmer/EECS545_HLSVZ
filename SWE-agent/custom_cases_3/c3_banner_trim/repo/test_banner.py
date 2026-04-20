from banner import render_banner


def test_render_banner_trims_name() -> None:
    assert render_banner("  Ops  ") == "[ Ops ]"


def test_render_banner_clean_name() -> None:
    assert render_banner("Ops") == "[ Ops ]"
