"""Fixed logistic objective and derivatives; deliberately no fitting routine."""
import math


def features(n, h):
    return [1.0, math.log1p(n), h / 60.0]


def hold_only_features(h):
    return [1.0, h / 60.0]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    value = math.exp(z)
    return value / (1.0 + value)


def linear_predictor(beta, x):
    if len(beta) != len(x):
        raise ValueError('Coefficient and feature dimensions differ')
    return sum(b * value for b, value in zip(beta, x))


def probability(beta, x):
    return sigmoid(linear_predictor(beta, x))


def objective(beta, samples):
    """Return sum NLL + 0.5||beta||², gradient and Hessian at supplied beta.

    Samples are (feature_vector, binary_label). Evaluating this function does
    not update coefficients. The intercept has the same unit penalty as slopes.
    """
    dimension = len(beta)
    loss = 0.5 * sum(b * b for b in beta)
    gradient = list(beta)
    hessian = [[float(i == j) for j in range(dimension)] for i in range(dimension)]
    for x, y in samples:
        if y not in (0, 1):
            raise ValueError('A known binary target is required')
        z = linear_predictor(beta, x)
        p = sigmoid(z)
        loss += max(z, 0.0) + math.log1p(math.exp(-abs(z))) - y * z
        weight = p * (1.0 - p)
        for i in range(dimension):
            gradient[i] += (p - y) * x[i]
            for j in range(dimension):
                hessian[i][j] += weight * x[i] * x[j]
    return dict(loss=loss, gradient=gradient, hessian=hessian)
