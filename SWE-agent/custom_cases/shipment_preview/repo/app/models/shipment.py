from dataclasses import dataclass


@dataclass(frozen=True)
class Shipment:
    recipient_name: str
    route_code: str
