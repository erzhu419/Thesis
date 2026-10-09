"""Source-scaled queue representations shared by actor and replay states."""

from math import log, log1p

WAITING = [17] + list(range(19, 91, 6))


def transform(state, arm, cap):
    changed = state.copy()
    if arm == "identity":
        return changed, 1.
    total = float(state[17])
    if arm == "source_clip":
        factor = 1. / max(cap, total)
    elif arm == "source_log":
        factor = log1p(total / cap) / (log(2.) * total) if total else 1. / (cap * log(2.))
    else:
        raise ValueError(arm)
    for index in WAITING:
        changed[index] *= factor
    return changed, factor
