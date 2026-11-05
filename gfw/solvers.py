"""One interface to both solvers, which minimise the same problem:

    min_x 1/2 ||L x - d||^2 + alpha * sum_g sqrt(x_g^T W_g x_g)

solver="fista" : the default. Substitution y_g = W_g^{1/2} x_g, then FISTA with
                 restarts on a growing working set (from eeg_headfig/noise_sweep.py)
solver="bcd"   : block coordinate descent, gfw.groupSparsity. Much slower, mainly
                 useful to validate FISTA

x is the flat vector of length 3 n_sources; use x.reshape(-1, 3) for moments.
"""
import numpy as np

from .groupSparsity import groupSparsity
from .weightings import groups_of


def alpha_max(L, W, d):
    """Smallest alpha for which x = 0 is the solution."""
    v = (L.T @ d).reshape(-1, 3)
    return np.sqrt(np.einsum("gi,gij,gj->g", v, np.linalg.inv(W), v)).max()


def _solve_bcd(L, W, d, alpha, x0=None, max_iter=10000, rel_change_tol=1e-15,
               rel_primal_dual_gap_tol=1e-10, gap_check_every=10, verbose=False):
    groups = np.arange(L.shape[1]) // 3
    model = groupSparsity(
        alpha=float(alpha),
        groups=groups,
        group_weights={g: W[g] for g in range(W.shape[0])},
        solver="bcd",
        max_iter=max_iter,
        rel_change_tol=rel_change_tol,
        rel_primal_dual_gap_tol=rel_primal_dual_gap_tol,
        gap_check_every=gap_check_every,
        verbose=verbose,
    )
    model.fit(L, d, x0=x0)
    return model.coef_


class WeightedGroupLasso:
    """min_x 1/2 ||L x - d||^2 + alpha * sum_g sqrt(x_g^T W_g x_g).

    Substituting y_g = W_g^{1/2} x_g turns this into a standard group lasso
    with operator  L~_g = L_g W_g^{-1/2},  solved by FISTA.
    """

    def __init__(self, L, W, eig_floor=1e-10):
        lam_w, Q = np.linalg.eigh(W)                            # (m,3), (m,3,3)
        floor = eig_floor * lam_w.max()
        self.n_floored = int(np.sum(lam_w < floor))
        if self.n_floored:
            print(f"warning: {self.n_floored} tiny eigenvalues in W floored")
        lam_w = np.maximum(lam_w, floor)
        self.W_isqrt = np.einsum("gik,gk,gjk->gij", Q, lam_w ** -0.5, Q)
        self.W_sqrt = np.einsum("gik,gk,gjk->gij", Q, lam_w ** 0.5, Q)
        Lg = groups_of(L)
        Lt = np.einsum("gai,gij->gaj", Lg, self.W_isqrt)        # (m, M, 3)
        self.Lt = Lt.transpose(1, 0, 2).reshape(L.shape)        # (M, 3m)
        self.m = L.shape[1] // 3

    @staticmethod
    def _fista(A, d, lam, y0, max_iter=20000, tol=1e-9, callback=None):
        """Plain group-lasso FISTA (groups of 3 columns) with adaptive restart."""
        step = 1.0 / np.linalg.norm(A, 2) ** 2
        y = y0.copy()
        z, t = y.copy(), 1.0
        for _ in range(max_iter):
            v = z - step * (A.T @ (A @ z - d))
            V = v.reshape(-1, 3)
            nrm = np.linalg.norm(V, axis=1)
            shrink = np.maximum(0.0, 1.0 - step * lam / np.maximum(nrm, 1e-300))
            y_new = (V * shrink[:, None]).ravel()
            if np.dot(z - y_new, y_new - y) > 0:                # restart
                t = 1.0
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            z = y_new + ((t - 1) / t_new) * (y_new - y)
            dy = np.linalg.norm(y_new - y)
            y, t = y_new, t_new
            if callback is not None:
                callback(y)
            if dy <= tol * max(1.0, np.linalg.norm(y)):
                break
        return y

    def solve_y(self, d, lam, y0=None, add=10, max_outer=100, max_iter=20000, tol=1e-9, callback=None):
        """Active-set strategy: since M << 3m, at most ~M groups are active.
        Solve on a small working set, then check the optimality condition
        ||L~_g^T r|| <= lam for all inactive groups; add violators, repeat."""
        m, Lt = self.m, self.Lt
        y = np.zeros(3 * m) if y0 is None else y0.copy()
        active = set(np.flatnonzero(np.linalg.norm(y.reshape(m, 3), axis=1)))
        solved = not active                                     # a warm start still has to be solved
        for _ in range(max_outer):
            r = d - Lt @ y
            c = np.linalg.norm((Lt.T @ r).reshape(m, 3), axis=1)
            c[list(active)] = 0.0
            viol = np.flatnonzero(c > lam * (1 + 1e-6))
            if viol.size == 0 and solved:
                break
            active |= set(viol[np.argsort(c[viol])[::-1][:add]])
            idx = np.sort(np.fromiter(active, int))
            cols = (3 * idx[:, None] + np.arange(3)).ravel()
            inner = None if callback is None else (lambda y_it: callback(cols, y_it))
            y_sub = self._fista(Lt[:, cols], d, lam, y[cols], max_iter=max_iter, tol=tol, callback=inner)
            y = np.zeros(3 * m)
            y[cols] = y_sub
            solved = True
            # drop groups that became zero to keep the working set small
            active = set(idx[np.linalg.norm(y_sub.reshape(-1, 3), axis=1) > 0])
        return y

    def to_x(self, y):
        Y = y.reshape(self.m, 3)
        return np.einsum("gij,gj->gi", self.W_isqrt, Y).ravel()

    def to_y(self, x):
        X = np.asarray(x).reshape(self.m, 3)
        return np.einsum("gij,gj->gi", self.W_sqrt, X).ravel()


def _solve_fista(L, W, d, alpha, x0=None, **kwargs):
    gl = WeightedGroupLasso(L, W)
    y0 = None if x0 is None else gl.to_y(x0)
    return gl.to_x(gl.solve_y(d, alpha, y0=y0, **kwargs))


def solve(L, W, d, alpha, solver="fista", x0=None, **kwargs):
    """Minimise the weighted group lasso; returns the flat solution x.

    x0 is an optional warm start. Further keyword arguments go to the solver.
    """
    if solver == "bcd":
        return _solve_bcd(L, W, d, alpha, x0=x0, **kwargs)
    if solver == "fista":
        return _solve_fista(L, W, d, alpha, x0=x0, **kwargs)
    raise ValueError(f"Unknown solver '{solver}'. Use 'bcd' or 'fista'.")
