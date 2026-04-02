from app.main import build_export
from app.models.shipment import Shipment


def test_export_keeps_uppercase_route_code() -> None:
    shipment = Shipment(recipient_name="mcintyre-ross", route_code="mcintyre-ross")
    assert build_export(shipment) == "route_code=MCINTYRE-ROSS"


def test_export_keeps_existing_code_shape() -> None:
    shipment = Shipment(recipient_name="ali", route_code="ab-12")
    assert build_export(shipment) == "route_code=AB-12"
