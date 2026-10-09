"""Coherent queue compression, with thresholds fitted only to source probes."""

WAITING = [17] + list(range(19, 91, 6))


def compress(state, variant, seconds, limits):
    changed = state.copy()
    if variant == "identity":
        return changed, 1., 1.
    if variant == "daily_total":
        ratio = float(state[17]) / limits["daily_caps"][0]
    else:
        phase = min(int(seconds // limits["bin_seconds"]), len(limits["phase_caps"]) - 1)
        ratio = max(float(state[index]) / cap
                    for index, cap in zip(WAITING, limits["phase_caps"][phase]))
    factor = 1. / max(1., ratio)
    if factor < 1.:
        for index in WAITING:
            changed[index] *= factor
    return changed, ratio, factor
