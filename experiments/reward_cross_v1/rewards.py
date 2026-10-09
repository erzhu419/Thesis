"""Selectable objectives without modifying the completed bus experiments."""

import numpy as np

OBJECTIVES = ("original", "headway", "passenger")


def resolve_objective(kind, objective):
    return ("passenger" if kind == "mc" else "headway") if objective == "original" else objective


def passenger_reward(wait_delta, ride_delta):
    return -(2. * np.asarray(wait_delta) + np.asarray(ride_delta))


def ridge_reward(forward, backward, target_forward=360., target_backward=360.,
                 forward_present=True, backward_present=True):
    f, b, tf, tb = np.broadcast_arrays(forward, backward, target_forward, target_backward)
    fp, bp = np.broadcast_arrays(forward_present, backward_present)
    df, db = np.abs(f-tf), np.abs(b-tb)
    weight = df / (df+db+1e-6)
    ratio = tf / np.maximum(tb, 1e-6)
    symmetry = .5*np.abs(f-ratio*b)/((1+ratio)/2)
    both = -(weight*df+(1-weight)*db+symmetry)
    reward = np.where(fp & bp, both, np.where(fp, -df, np.where(bp, -db, -50.)))
    large = (fp & (df > .5*tf)) | (bp & (db > .5*tb))
    return reward-20.*large


def canonical_headway_reward(next_states, terminals=0., target_scale=720.):
    """Next chronological decision; terminal rows have no subsequent headway."""
    states = np.asarray(next_states, dtype=np.float32)
    vector = states.ndim == 1
    states = states.reshape(-1, 229)
    slots = states[:, 5:].reshape(-1, 16, 14)
    index = np.rint(states[:, 2]*15).astype(int)
    local = slots[np.arange(len(states)), index]
    tf, tb = local[:, 5]*target_scale, local[:, 6]*target_scale
    reward = ridge_reward(local[:, 3]*tf, local[:, 4]*tb, tf, tb,
                          local[:, 11] > .5, local[:, 12] > .5)
    reward = np.where((np.asarray(terminals).reshape(-1) > .5) | (local[:, 0] < .5), 0., reward)
    return float(reward[0]) if vector else reward.astype(np.float32)


class PassengerLedger:
    """One system-wide integral, with per-vehicle decision-interval bookmarks."""
    def __init__(self):
        self.wait = self.ride = 0.
        self.marks = {}

    def advance(self, waiting, onboard, seconds):
        self.wait += waiting*seconds
        self.ride += onboard*seconds

    def settle(self, key):
        previous = self.marks.get(key, (self.wait, self.ride))
        self.marks[key] = (self.wait, self.ride)
        return float(passenger_reward(self.wait-previous[0], self.ride-previous[1]))


def counts(lines):
    return (sum(len(s.waiting_passengers) for line in lines for s in line.stations),
            sum(len(b.passengers) for line in lines for b in line.bus_all))
