from formatter import normalize_label


def test_trims_edges_only():
    assert normalize_label("  Hello world  ") == "Hello world"


def test_preserves_internal_spacing():
    assert normalize_label("Order   #42") == "Order   #42"


def test_preserves_tabs_inside_text():
    assert normalize_label("Field\tValue") == "Field\tValue"
