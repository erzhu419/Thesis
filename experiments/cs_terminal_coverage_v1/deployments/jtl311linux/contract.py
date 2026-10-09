"""Finite-horizon passenger cost and sequential same-tick action credit."""

import numpy as np


def potential(time, horizon, waiting, onboard):
    return -max(0., horizon - time) * (2. * waiting + onboard)


class EventCredit:
    def __init__(self, horizon, scale, shaped):
        self.horizon, self.scale, self.shaped = horizon, scale, shaped
        self.pending = None
        self.prefix_cost = self.first_potential = None
        self.raw_sum = self.shaped_sum = 0.
        self.transitions = self.zero_time = self.terminals = 0

    def begin(self, state, action, time, cost, waiting, onboard):
        assert self.pending is None
        phi = potential(time, self.horizon, waiting, onboard)
        if self.prefix_cost is None:
            self.prefix_cost, self.first_potential = cost, phi
        self.pending = (state, action, time, cost, phi)

    def finish(self, next_state, time, cost, waiting, onboard, terminal=False):
        if self.pending is None:
            return None
        state, action, before, previous_cost, phi = self.pending
        self.pending = None
        assert time >= before and cost >= previous_cost
        raw = -(cost - previous_cost)
        next_phi = 0. if terminal else potential(time, self.horizon, waiting, onboard)
        shaped = raw + next_phi - phi
        self.raw_sum += raw
        self.shaped_sum += shaped
        self.transitions += 1
        self.zero_time += int(time == before)
        self.terminals += int(terminal)
        reward = (shaped if self.shaped else raw) / self.scale
        return state, action, reward, next_state, float(terminal)

    def verify(self, final_cost):
        assert self.pending is None and self.terminals == 1
        assert np.isclose(self.raw_sum, -(final_cost - self.prefix_cost), atol=.01)
        assert np.isclose(self.shaped_sum, self.raw_sum - self.first_potential, atol=.01)
        return dict(prefix_cost=self.prefix_cost, first_potential=self.first_potential,
                    raw_reward_sum=self.raw_sum, shaped_reward_sum=self.shaped_sum,
                    transitions=self.transitions, zero_time_transitions=self.zero_time,
                    terminal_transitions=self.terminals, cost_identity_passed=True)


class ArrivalFlow:
    def __init__(self, seed, line_index, schedule):
        streams = np.random.SeedSequence([seed, line_index, 81008]).spawn(3)
        self.baseline_rng, self.extra_rng, self.order_rng = (
            np.random.default_rng(stream) for stream in streams)
        self.schedule = schedule

    def draw(self, rates, interval, time):
        intensity = float(self.schedule[0][1])
        for start, value in self.schedule:
            if time < start:
                break
            intensity = float(value)
        rates = np.asarray(rates) * interval
        baseline = self.baseline_rng.poisson(rates)
        extra = self.extra_rng.poisson(rates * (intensity - 1)) if intensity > 1 else np.zeros_like(baseline)
        return baseline, extra


def observation(local, bus_onboard, capacity, time, horizon, line_stats, committed, pending, local_only=False):
    state = np.zeros(92, dtype=np.float32)
    state[:15] = local
    if local_only:
        return state
    state[15:19] = [bus_onboard / capacity, (horizon - time) / horizon,
                    sum(s[0] for s in line_stats) / 100000.,
                    sum(s[1] for s in line_stats) / 1000.]
    for index, (waiting, onboard, active, dwell) in enumerate(line_stats):
        actions = committed.get(index, [])
        state[19 + index * 6:25 + index * 6] = [
            waiting / 10000., onboard / 1000., active / 20., dwell / 120.,
            len(actions) / 20., np.mean(actions) / 60. if actions else 0.]
    state[91] = pending / 100.
    assert np.isfinite(state).all()
    return state
