from app.exports.csv_writer import render_export_row
from app.models.shipment import Shipment
from app.presenters.preview_presenter import render_preview


def build_preview(shipment: Shipment) -> str:
    return render_preview(shipment)


def build_export(shipment: Shipment) -> str:
    return render_export_row(shipment)
