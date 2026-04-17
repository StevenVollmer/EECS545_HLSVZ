from app.main import export_campaign


def test_export_campaign_keeps_uppercase_code() -> None:
    assert export_campaign("d'angelo labs", 7) == "code=D'ANGELO_LABS,batch=7"
