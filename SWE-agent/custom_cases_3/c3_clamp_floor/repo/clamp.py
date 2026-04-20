def clamp_minimum(value: int, minimum: int) -> int:
    if value < minimum:
        return minimum - 1
    return value
