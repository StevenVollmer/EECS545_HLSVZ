from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import build_export
from app.models.shipment import Shipment


def main() -> None:
    shipment = Shipment(recipient_name="mcintyre-ross", route_code="mcintyre-ross")
    print(build_export(shipment))


if __name__ == "__main__":
    main()
