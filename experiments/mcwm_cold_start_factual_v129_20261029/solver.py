"""The fixed v128 damped-Newton solver; no fallback or parameter tuning."""
import math

from experiments.mcwm_cold_start_contract_v128_20261029.model import objective


def _cholesky_solve(matrix, rhs):
    dimension = len(rhs)
    lower = [[0.0]*dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i+1):
            value = matrix[i][j] - sum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if not math.isfinite(value) or value <= 0.0:
                    raise ArithmeticError("nonpositive_cholesky_pivot")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value/lower[j][j]
    intermediate = [0.0]*dimension
    for i in range(dimension):
        intermediate[i] = (rhs[i]-sum(lower[i][j]*intermediate[j] for j in range(i)))/lower[i][i]
    solution = [0.0]*dimension
    for i in range(dimension-1, -1, -1):
        solution[i] = (intermediate[i]-sum(lower[j][i]*solution[j] for j in range(i+1, dimension)))/lower[i][i]
    return solution


def fit(samples, dimension):
    """Fit from zero; retain accepted state and all work counters on failure.

    Newton steps count attempted iterations, including a failed line search.
    Objective evaluations include the initial evaluation and every candidate;
    each candidate evaluates the unchanged analytic loss, gradient and Hessian.
    """
    beta, current = [], None
    steps = line_evaluations = evaluations = 0

    def result(reason, converged=False):
        return dict(converged=converged, beta=list(beta),
            loss=None if current is None else current["loss"],
            gradient_max=None if current is None else max(abs(value) for value in current["gradient"]),
            newton_steps=steps, line_search_evaluations=line_evaluations,
            objective_evaluations=evaluations, reason=reason)

    def evaluate(values):
        nonlocal evaluations
        evaluations += 1
        evaluated = objective(values, samples)
        numbers = [evaluated["loss"], *evaluated["gradient"],
                   *(value for row in evaluated["hessian"] for value in row)]
        if not all(math.isfinite(value) for value in numbers):
            raise ArithmeticError("nonfinite_objective")
        return evaluated

    if dimension not in (2, 3):
        return result("invalid_dimension")
    beta = [0.0]*dimension
    try:
        current = evaluate(beta)
        for _ in range(100):
            if max(abs(value) for value in current["gradient"]) <= 1e-8:
                return result("converged", True)
            steps += 1
            direction = [-value for value in _cholesky_solve(current["hessian"], current["gradient"])]
            slope = sum(value*change for value, change in zip(current["gradient"], direction))
            if not math.isfinite(slope) or slope >= 0.0:
                return result("non_descent_direction")
            alpha, accepted = 1.0, False
            for _ in range(40):
                candidate_beta = [value+alpha*change for value, change in zip(beta, direction)]
                line_evaluations += 1
                candidate = evaluate(candidate_beta)
                if candidate["loss"] <= current["loss"] + 0.25*alpha*slope:
                    beta, current, accepted = candidate_beta, candidate, True
                    break
                alpha *= 0.5
            if not accepted:
                return result("line_search_exhausted")
        if max(abs(value) for value in current["gradient"]) <= 1e-8:
            return result("converged", True)
        return result("maximum_newton_steps")
    except Exception as error:
        return result(type(error).__name__+": "+str(error))
