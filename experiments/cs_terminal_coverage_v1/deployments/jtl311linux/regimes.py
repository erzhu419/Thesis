"""Source-only training demand; evaluation schedules remain frozen."""


def source_schedule(regime, horizon):
    if regime == "normal":
        return [[0, 1]]
    if regime in ["static2", "static3"]:
        return [[0, int(regime[-1])]]
    if regime == "pulse3":
        return [[0, 1], [horizon // 5, 3], [3 * horizon // 10, 1]]
    if regime == "repeated3":
        return [[0, 1], [horizon // 5, 3], [3 * horizon // 10, 1],
                [horizon // 2, 3], [3 * horizon // 5, 1],
                [4 * horizon // 5, 3], [9 * horizon // 10, 1]]
    raise ValueError(regime)


def training_regime(arm, episode, cycle, smoke=False):
    if arm == "global_terminal":
        return "normal"
    if arm == "global_coverage":
        return ["pulse3", "repeated3"][episode % 2] if smoke else cycle[episode % len(cycle)]
    raise ValueError(arm)
