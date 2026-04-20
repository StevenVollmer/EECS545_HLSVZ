from app.main import export_route


def test_export_route_keeps_uppercase_label() -> None:
    assert export_route("operations/o'neil-ward", 3) == "route=OPERATIONS/O'NEIL-WARD,milestone=3"
