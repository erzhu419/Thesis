"""Retain the frozen source scale without an upper bound."""

WAITING = [17] + list(range(19, 91, 6))


def normalize_linear(state, cap):
    changed = state.copy()
    factor = 1. / cap
    for index in WAITING:
        changed[index] *= factor
    return changed, factor
