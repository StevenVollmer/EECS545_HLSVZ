from app.stats.weighted import weighted_median


def test_unit_weights_odd():
    # With unit weights, weighted median == plain median.
    assert weighted_median([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]) == 3


def test_unit_weights_even():
    # Spec: returns the value at which cumulative weight first crosses W/2.
    # With 4 unit-weight entries, that's the 2nd value (cum=2 >= 2).
    assert weighted_median([(10, 1), (20, 1), (30, 1), (40, 1)]) == 20


def test_single_element():
    assert weighted_median([(7, 1)]) == 7


def test_input_need_not_be_sorted():
    assert weighted_median([(3, 1), (1, 1), (2, 1)]) == 2
