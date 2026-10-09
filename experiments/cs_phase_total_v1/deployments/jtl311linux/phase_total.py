"""Use only the source time-bin total to scale coherent waiting features."""

WAITING = [17] + list(range(19, 91, 6))


def compress_total(state, seconds, limits):
    phase = min(int(seconds // limits["bin_seconds"]), len(limits["phase_caps"]) - 1)
    ratio = float(state[17]) / limits["phase_caps"][phase][0]
    factor = 1. / max(1., ratio)
    changed = state.copy()
    if factor < 1.:
        for index in WAITING:
            changed[index] *= factor
    return changed, ratio, factor
