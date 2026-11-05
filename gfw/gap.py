"""Objective and primal-dual gap, independent of the solver that produced x.

    primal:  min_x 1/2 ||L x - d||^2 + alpha * sum_g sqrt(x_g^T W_g x_g)
    dual:    max_y -1/2 ||y + d||^2 + 1/2 ||d||^2
             s.t.  ||L_g^T y||_{W_g^{-1}} <= alpha  for all g

The dual candidate is the residual, rescaled to be feasible and then optimal
along that direction. This is the construction of groupSparsity._dual_candidate.
"""
import numpy as np


def _penalty(x, W):
    X = np.asarray(x).reshape(-1, 3)
    return np.sqrt(np.maximum(np.einsum("gi,gij,gj->g", X, W, X), 0.0)).sum()


def objective(L, W, d, alpha, x):
    x = np.asarray(x).ravel()
    return 0.5 * np.sum((L @ x - d) ** 2) + alpha * _penalty(x, W)


def relative_gap(L, W, d, alpha, x, W_inv=None):
    """Return (primal - dual) / primal at x. Pass W_inv to skip inverting W."""
    x = np.asarray(x).ravel()
    W_inv = np.linalg.inv(W) if W_inv is None else W_inv
    res = L @ x - d
    primal = 0.5 * res @ res + alpha * _penalty(x, W)
    if primal == 0:
        return 0.0
    v = (L.T @ res).reshape(-1, 3)
    dual_norm = np.sqrt(np.maximum(np.einsum("gi,gij,gj->g", v, W_inv, v), 0.0)).max()
    theta = alpha / dual_norm if dual_norm > 0 else np.inf
    c = np.clip(-(res @ d) / (res @ res), -theta, theta) if np.any(res) else 0.0
    y = c * res
    dual = -0.5 * np.sum((y + d) ** 2) + 0.5 * d @ d
    return (primal - dual) / primal
