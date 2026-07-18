"""Ambiguity tube construction helpers."""

from __future__ import annotations

import numpy as np


def dr_bounds(samples, weights, beta: float = 0.95, solver: str = "GUROBI", non_dr: bool = False) -> list[float]:
    """Construct distributionally robust lower and upper tube bounds."""
    import cvxpy as cp
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    n_samples = samples.shape[0]
    sample_range = np.max(samples, axis=0) - np.min(samples, axis=0)
    rho = sample_range * np.sqrt(2 * np.log(2 / beta)) / n_samples
    if non_dr:
        return [np.min(samples), np.max(samples)]
    theta_l = cp.Variable()
    theta_u = cp.Variable()
    eta = cp.Variable()
    lamda = cp.Variable()
    slack = cp.Variable(n_samples)

    objective = cp.Minimize(theta_u - theta_l)
    a = np.array([1, -1])
    constraints = [
        theta_l <= theta_u,
        eta + (lamda * rho + weights @ slack) / (1 - beta) <= 0,
        slack >= 0,
        lamda >= np.linalg.norm(a, ord=np.inf),
    ]
    for j in range(n_samples):
        constraints.append(slack[j] >= a[0] * samples[j] - theta_u - eta)
        constraints.append(slack[j] >= a[1] * samples[j] + theta_l - eta)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)
    return [theta_l.value, theta_u.value]


def radius(samples, beta: float = 0.95) -> float:
    """Compute the ambiguity tube radius used by the notebook implementation."""
    samples = np.asarray(samples)
    n_samples = samples.shape[0]
    sample_range = np.max(samples, axis=0) - np.min(samples, axis=0)
    return float(sample_range * np.sqrt(2 * np.log(2 / beta)) / n_samples)
