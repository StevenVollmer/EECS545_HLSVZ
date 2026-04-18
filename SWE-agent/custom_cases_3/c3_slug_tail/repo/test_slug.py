from slug import slug_tail


def test_slug_tail_reads_last_piece() -> None:
    assert slug_tail("north-region-7") == "7"


def test_slug_tail_single_piece() -> None:
    assert slug_tail("alpha") == "alpha"
