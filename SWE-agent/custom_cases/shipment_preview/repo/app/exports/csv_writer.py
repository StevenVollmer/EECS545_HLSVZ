from app.models.shipment import Shipment
from app.utils.text import export_code


def render_export_row(shipment: Shipment) -> str:
    return f"route_code={export_code(shipment.route_code)}"
