"""Bounded evidence averaging, with an optional global-coefficient fallback."""

from collections import deque
import numpy as np


def evidence(model, features):
    x = (np.asarray(features) - model["mean"]) / model["scale"]
    return -.5 * ((x - model["centers"]) ** 2 / model["variance"]).mean(1)


def softmax(scores):
    weights = np.exp(scores - np.max(scores))
    return weights / weights.sum()


def route(belief, coefficients, global_coefficient, minimum_margin):
    ordered = np.sort(belief)
    accepted = ordered[-1] - ordered[-2] >= minimum_margin
    coefficient = belief @ coefficients if accepted else global_coefficient
    return float(coefficient), bool(accepted)


class WindowFilter:
    def __init__(self, model, coefficients, window, global_coefficient, minimum_margin=0.):
        self.model, self.coefficients = model, np.asarray(coefficients)
        self.window = window
        self.global_coefficient, self.minimum_margin = global_coefficient, minimum_margin
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.window)
        self.belief = np.full(len(self.coefficients), 1 / len(self.coefficients))

    def observe(self, features):
        self.history.append(evidence(self.model, features))
        self.belief = softmax(np.mean(self.history, axis=0))
        return self.coefficient

    @property
    def decision(self):
        return route(self.belief, self.coefficients, self.global_coefficient, self.minimum_margin)

    @property
    def coefficient(self):
        return self.decision[0]
