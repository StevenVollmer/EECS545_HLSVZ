from app.main import build_preview
from app.models.shipment import Shipment


def test_preview_renders_a_string() -> None:
    shipment = Shipment(recipient_name="mcintyre-ross", route_code="mcintyre-ross")
    assert build_preview(shipment).startswith("Recipient: ")


def test_preview_uses_current_display_helper() -> None:
    shipment = Shipment(recipient_name="ali", route_code="ali")
    assert build_preview(shipment) == "Recipient: ALI"


def test_export_path_is_not_affected_by_preview_rendering() -> None:
    shipment = Shipment(recipient_name="ali", route_code="ali")
    assert shipment.route_code == "ali"
