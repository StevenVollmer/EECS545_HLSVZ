from app.main import export_team_code, roster_signature
from app.utils.names import lead_display_name


def test_export_team_code_uppercase() -> None:
    assert export_team_code('infra') == 'team=INFRA'


def test_lead_display_name_simple_case() -> None:
    assert lead_display_name('mina-west') == 'Mina-west'


def test_roster_signature_simple_lead() -> None:
    assert roster_signature('Infra', 'ava') == 'Infra: lead Ava'
