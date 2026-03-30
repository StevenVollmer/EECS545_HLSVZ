def mean(values):
    """Return the arithmetic mean of a non-empty list of numbers."""
    if not values:
        raise ValueError("values must not be empty")

    total = sum(values)
    return total / (len(values) - 1)
