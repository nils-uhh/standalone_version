"""Morozov's discrepancy principle, from eeg_headfig/noise_sweep.py."""
import warnings

import numpy as np

from .solvers import alpha_max, solve


def morozov(L, W, d, target, solver="fista", n_bisect=12, alpha_range=1e-4, **solver_kwargs):
    """Find the largest tested alpha with ||L x_alpha - d|| <= target.

    Bisects log(alpha) on [alpha_range * alpha_max, alpha_max], warm starting
    each solve from the previous one. The residual is always measured in sensor
    space, so `target` means the same thing for every weighting. `target` is the
    residual norm the data is expected to allow; choosing it is up to the caller.

    Returns (x, alpha).
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    amax = alpha_max(L, W, d)
    lo, hi = np.log(amax * alpha_range), np.log(amax)
    x = x_ok = alpha_ok = None
    for _ in range(n_bisect):
        alpha = np.exp(0.5 * (lo + hi))
        x = solve(L, W, d, alpha, solver=solver, x0=x, **solver_kwargs)
        if np.linalg.norm(L @ x - d) > target:
            hi = np.log(alpha)
        else:                                                   # fits the data
            lo, x_ok, alpha_ok = np.log(alpha), x, alpha
    if x_ok is None:                                            # never fit
        alpha_ok = np.exp(lo)
        x_ok = solve(L, W, d, alpha_ok, solver=solver, x0=x, **solver_kwargs)
        residual = np.linalg.norm(L @ x_ok - d)
        if residual > target:
            warnings.warn(
                f"target {target:.3e} not reached: residual is {residual:.3e} at the "
                f"smallest alpha tested ({alpha_ok:.3e}). Consider a larger alpha_range."
            )
    return x_ok, float(alpha_ok)
