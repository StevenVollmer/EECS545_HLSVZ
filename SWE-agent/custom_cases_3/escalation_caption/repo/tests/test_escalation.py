from app.main import escalation_caption, export_channel_label
from app.utils.text import format_owner_name


def test_export_channel_label_uppercase() -> None:
    assert export_channel_label('ops') == 'channel=OPS'


def test_format_owner_name_simple_hyphen() -> None:
    assert format_owner_name('ava-west') == 'Ava-west'


def test_escalation_caption_simple_owner() -> None:
    assert escalation_caption('mila', 'ops', 1) == '[OPS] Mila escalation L1'
