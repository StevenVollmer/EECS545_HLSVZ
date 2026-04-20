from app.models.shipment import Shipment
from app.utils.text import display_name


def render_preview(shipment: Shipment) -> str:
    return f"Recipient: {display_name(shipment.recipient_name)}"
