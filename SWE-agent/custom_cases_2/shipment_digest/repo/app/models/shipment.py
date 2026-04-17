from dataclasses import dataclass


@dataclass
class Shipment:
    tracking_id: str
    days_late: int
    delivered: bool = False
