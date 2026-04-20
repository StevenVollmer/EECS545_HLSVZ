from app.main import export_route


def test_export_route_keeps_uppercase_route_code() -> None:
    assert export_route("d'arcy-lee") == "route=D'ARCY-LEE"

