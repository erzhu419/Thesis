"""Frozen context evidence from nonoverlapping completed source transitions."""

import numpy as np


def transition_features(states, actions, durations):
    slots = states[:, 5:].reshape(-1, 16, 14)
    current = np.rint(states[:, 2] * 15).astype(int)
    local = slots[np.arange(len(states)), current]
    return np.column_stack((np.log1p(durations), local[:, 9], local[:, 10],
        local[:, 7], local[:, 8], states[:, 3], states[:, 4], (actions.flatten() + 1) / 2))


def fit_recognizer(labeled, classes, variance_floor):
    rows = [np.asarray(labeled[c], dtype=float) for c in classes]
    assert all(len(r) > 0 for r in rows)
    combined = np.concatenate(rows)
    mean, scale = combined.mean(0), combined.std(0).clip(.001)
    normalized = [(r - mean) / scale for r in rows]
    centers = np.stack([r.mean(0) for r in normalized])
    variance = np.stack([((r - center) ** 2).mean(0)
                         for r, center in zip(normalized, centers)]).mean(0).clip(variance_floor)
    return dict(classes=classes, mean=mean.tolist(), scale=scale.tolist(),
                centers=centers.tolist(), variance=variance.tolist(),
                rows_per_class=[len(r) for r in rows])


class ContextFilter:
    def __init__(self, model, coefficients, protocol):
        self.model, self.coefficients, self.p = model, np.asarray(coefficients), protocol
        self.reset()

    def reset(self):
        self.belief = np.full(len(self.coefficients), 1 / len(self.coefficients))

    def observe(self, features):
        x = (np.asarray(features) - self.model["mean"]) / self.model["scale"]
        scores = -.5 * ((x - self.model["centers"]) ** 2 / self.model["variance"]).mean(1)
        h, count = self.p["context_hazard"], len(self.belief)
        prior = (1 - h) * self.belief + h / (count - 1) * (1 - self.belief)
        logits = self.p["context_prior_exponent"] * np.log(prior) + self.p["context_evidence_scale"] * scores
        probabilities = np.exp(logits - logits.max())
        self.belief = probabilities / probabilities.sum()
        return self.coefficient

    @property
    def coefficient(self):
        return float(self.belief @ self.coefficients)


class CompletedBlocks:
    def __init__(self, size):
        self.size, self.cursor, self.pending, self.blocks = size, 0, [], []

    def consume(self, replay, callback=None, reset=None):
        stop = replay.online_size
        assert replay._online_total_appended == stop, "Protocol requires a nonwrapped source buffer"
        for index in range(self.cursor, stop):
            if replay._online_terminals[index] > .5:
                self.pending = []
                if reset is not None:
                    reset()
                continue
            dt = replay._online_delta_t_seconds[index]
            if dt <= 0:
                continue
            row = transition_features(replay._online_observations[index:index+1],
                replay._online_actions[index:index+1], np.asarray([dt]))[0]
            self.pending.append(row)
            if len(self.pending) == self.size:
                features = np.stack(self.pending).mean(0)
                block = dict(end_source_row=index+1, physical_rows=self.size, features=features.tolist())
                self.blocks.append(block)
                if callback is not None:
                    callback(block)
                self.pending = []
        self.cursor = stop
