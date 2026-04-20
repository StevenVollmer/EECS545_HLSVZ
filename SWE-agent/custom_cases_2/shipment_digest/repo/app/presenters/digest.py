from app.models.shipment import Shipment
from app.services.shipment_service import overdue_count


def render_digest(shipments: list[Shipment]) -> str:
    return f"delayed shipments: {overdue_count(shipments)}"
