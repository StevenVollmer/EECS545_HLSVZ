from app.models.shipment import Shipment
from app.presenters.digest import render_digest
from app.services.shipment_service import overdue_shipments


def test_overdue_shipments_only_positive_lateness() -> None:
    shipments = [
        Shipment(tracking_id="A1", days_late=0, delivered=False),
        Shipment(tracking_id="B2", days_late=2, delivered=False),
    ]
    assert [shipment.tracking_id for shipment in overdue_shipments(shipments)] == ["B2"]


def test_digest_ignores_delivered_shipments() -> None:
    shipments = [
        Shipment(tracking_id="A1", days_late=4, delivered=True),
        Shipment(tracking_id="B2", days_late=1, delivered=False),
    ]
    assert render_digest(shipments) == "delayed shipments: 1"


def test_digest_empty_when_none_overdue() -> None:
    shipments = [Shipment(tracking_id="A1", days_late=-1, delivered=False)]
    assert render_digest(shipments) == "delayed shipments: 0"
