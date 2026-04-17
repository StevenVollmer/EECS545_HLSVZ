from app.models.shipment import Shipment


def overdue_shipments(shipments: list[Shipment]) -> list[Shipment]:
    return [shipment for shipment in shipments if not shipment.delivered and shipment.days_late >= 0]


def overdue_count(shipments: list[Shipment]) -> int:
    return len(overdue_shipments(shipments))
