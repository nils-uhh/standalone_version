#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ole løseth elvetun
GitHub: https://github.com/oleelv/groupSparsity/blob/main/groupSparsity.py

Problems solved
---------------
Notation: A = [A_1, ..., A_G] split into column groups, x = (x_1, ..., x_G).

Solver 1 -- Low-rank BCD (`fit` with A = (L, R)), requires group_weights='auto',
            i.e. W_g = (L R_g)^T (L R_g):
    min_x  1/2 ||L R x - b||_2^2 + alpha * sum_g ||L R_g x_g||_2

Solver 2 -- Standard BCD (solver='bcd'), matrix weights W_g:
    min_x  1/2 ||A x - b||_2^2 + alpha * sum_g sqrt(x_g^T W_g x_g)
         = 1/2 ||A x - b||_2^2 + alpha * sum_g ||W_g^{1/2} x_g||_2
    with W_g = A_g^T A_g ('auto'), user-supplied SPD matrices (dict), or I (None).

Solver 3 -- FISTA (solver='fista'), scalar weights w_g > 0 (list/array or None):
    min_x  1/2 ||A x - b||_2^2 + alpha * sum_g w_g ||x_g||_2
         (= solver 2's objective with W_g = w_g^2 I)

Invalid solver / weight combinations raise (see `_validate_params`, `_validate_fit_inputs`).

Stopping (all solvers): relative change < rel_change_tol, or relative primal-dual gap
(checked every `gap_check_every` iterations) < rel_primal_dual_gap_tol. The gap needs
invertible W_g; a singular W_g (e.g. rank-deficient A_g with 'auto') raises.
"""
import numpy as np
from scipy import sparse
import scipy.linalg
from scipy.optimize import root_scalar
from scipy.sparse.linalg import LinearOperator, svds
import warnings

class groupSparsity:
    def __init__(
        self, 
        alpha: float = 1.0, 
        groups: list = None,
        group_weights = None, 
        solver: str = 'bcd', 
        max_iter: int = 100, 
        rel_change_tol: float = 1e-15,
        rel_primal_dual_gap_tol: float = 1e-10,
        gap_check_every: int = 10,
        verbose: bool = False
    ):
        """
        Group Sparsity Solver supporting Standard Matrix, LinearOperator, and Low-Rank inputs.
        
        Parameters:
        -----------
        alpha : float
            Regularization strength.
        groups : array-like
            1D array of group labels (e.g., [0, 0, 1, 1, 2]).
        group_weights : dict, array, or 'auto', optional
            - 'auto': Computes W_g = A_g.T @ A_g (Matrix weights). BCD only; required for A = (L, R).
            - dict: Custom SPD matrix weights {group_id: W_g}. BCD only.
            - array: Scalar weights [w_0, w_1, ...] > 0. FISTA only.
            - None: Identity matrix (BCD) or 1.0 (FISTA).
        solver : str
            'bcd'   : Block Coordinate Descent (Best for matrices & matrix weights).
            'fista' : Fast Iterative Shrinkage (Best for LinearOperators & scalar weights).
        max_iter : int
            Maximum number of iterations.
        rel_change_tol : float
            Relative change tolerance until convergence declared. Check is max ||x_g_new - x_g_old|| / (x_old_norm + 1e-12) < tol
        rel_primal_dual_gap_tol : float
            Relative primal-dual gap tolerance for convergence. Check is gap/primal_obj < tol.
        gap_check_every : int
            How often to check the primal-dual gap.
        verbose : bool
            If True, print progress information.
        """
        self.alpha = alpha
        self.groups = np.array(groups) if groups is not None else None
        self.group_weights = group_weights
        self.solver = solver
        self.max_iter = max_iter        
        self.rel_change_tol = rel_change_tol
        self.rel_primal_dual_gap_tol = rel_primal_dual_gap_tol
        self.gap_check_every = gap_check_every
        self.verbose = verbose
        self.coef_ = None
        self._L = None # Cache for Lipschitz constant
        self._validate_params()

    def fit(self, A, b, x0=None):
        """
        Fits the model.
        
        Parameters:
        -----------
        A : Input Data
            - tuple (L, R) : Implicit Low-Rank A = L @ R. Uses Optimized Low-Rank BCD.
            - sparse matrix / ndarray : Uses Standard BCD.
            - LinearOperator : Uses FISTA.
        b : array-like
            data.
        x0 : array-like, optional
            Warm start (flat coefficient vector). Standard BCD only.
        """
        if x0 is not None and (isinstance(A, tuple) or self.solver != 'bcd'):
            raise ValueError("x0 (warm start) is only supported by the standard BCD solver.")

        # --- 1. Dimension & Group Setup ---
        if isinstance(A, tuple):
            n_features = A[1].shape[1]
        else:
            n_features = A.shape[1]
            
        if self.groups is None:
            self.groups = np.arange(n_features)

        self._validate_fit_inputs(A, n_features)

        # --- 2. Dispatch Logic ---
        
        # Case A: Low-Rank Factorization A = L @ R (e.g., B_pinv @ B)
        if isinstance(A, tuple) and len(A) == 2:
            if self.verbose: print("Input detected as Low-Rank Factorization (L, R). Using LR-BCD.")
            # Unpack L (n x m) and R (m x n)
            return self._solve_bcd_low_rank(A[0], A[1], b)

        # Case B: Standard Block Coordinate Descent
        elif self.solver == 'bcd':
            if self.verbose: print("Input detected as Standard Matrix. Using standard BCD.")
            if isinstance(A, LinearOperator) and not sparse.issparse(A):
                warnings.warn("Using BCD on a generic LinearOperator is inefficient. Consider solver='fista'.")
            return self._solve_bcd_standard(A, b, x0=x0)

        # Case C: FISTA (Gradient Descent)
        elif self.solver == 'fista':
            if self.verbose: print("Using FISTA.")
            return self._solve_fista(A, b)

        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    def predict(self, A):
        """Computes A @ coef_"""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted.")
            
        if isinstance(A, tuple):
            # A = L @ R
            return A[0] @ (A[1] @ self.coef_)
        elif isinstance(A, LinearOperator):
            return A.dot(self.coef_)
        else:
            return A @ self.coef_

    # =========================================================================
    #  INPUT VALIDATION
    # =========================================================================
    def _validate_params(self):
        """
        Checks the solver / group_weights combination (independent of A).

            solver='bcd'   : group_weights in {None, 'auto', dict of SPD matrices}
            solver='fista' : group_weights in {None, list/array of scalars > 0}

        Low-rank input A = (L, R) additionally requires 'auto' (checked in `fit`).
        """
        if self.solver not in ('bcd', 'fista'):
            raise ValueError(f"Unknown solver '{self.solver}'. Use 'bcd' or 'fista'.")
        if not np.isscalar(self.alpha) or self.alpha < 0:
            raise ValueError(f"alpha must be a non-negative scalar, got {self.alpha!r}.")

        gw = self.group_weights
        if isinstance(gw, str) and gw != 'auto':
            raise ValueError(f"Unknown group_weights '{gw}'. The only supported string is 'auto'.")

        if self.solver == 'bcd':
            # SPD is assumed in the code (eigh + eigenvalue clipping in the prox, inv(W_g) in
            # the dual gap). Extending to PSD or non-symmetric W_g (via W_g^T W_g) would be
            # possible, but is not implemented.
            if gw is None or isinstance(gw, str):
                return
            if not isinstance(gw, dict):
                raise TypeError(
                    "solver='bcd' requires group_weights=None, 'auto' or a dict {group_id: SPD matrix}, "
                    f"got {type(gw).__name__}. Scalar weights w_g can be passed as {{g: w_g * I}}."
                )
            for g, W_g in gw.items():
                self._check_spd(W_g, g)

        else:  # fista
            # Only the scalar-weight prox w_g ||x_g|| is implemented. Matrix weights would be
            # possible (the prox is `_solve_prox_matrix_weight` with step alpha/L), but are not
            # implemented.
            if gw is None:
                return
            if not isinstance(gw, (list, np.ndarray)):
                raise TypeError(
                    "solver='fista' requires group_weights=None or a list/array of scalar weights, "
                    f"got {type(gw).__name__}. Use solver='bcd' for matrix weights ('auto' / dict)."
                )
            w = np.asarray(gw, dtype=float)
            if w.ndim != 1:
                raise ValueError(f"FISTA group_weights must be 1D (one scalar per group), got shape {w.shape}.")
            # w_g > 0: the primal-dual gap uses W_g = w_g^2 I, which must be invertible
            if not np.all(np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("FISTA group_weights must be finite and strictly positive.")

    def _validate_fit_inputs(self, A, n_features):
        """Checks that depend on A and the groups (called in `fit`)."""
        if len(self.groups) != n_features:
            raise ValueError(f"groups has length {len(self.groups)}, but A has {n_features} columns.")
        unique_groups, group_sizes = np.unique(self.groups, return_counts=True)
        gw = self.group_weights

        if isinstance(A, tuple):
            if len(A) != 2:
                raise ValueError(f"Low-rank input must be a tuple (L, R), got a tuple of length {len(A)}.")
            if self.solver != 'bcd':
                raise ValueError("Low-rank input A = (L, R) is only supported by solver='bcd'.")
            # Low-rank BCD always uses W_g = (L R_g)^T (L R_g); other weights would be ignored.
            if not (isinstance(gw, str) and gw == 'auto'):
                raise ValueError(
                    "Low-rank input A = (L, R) only supports group_weights='auto' "
                    f"(W_g = (L R_g)^T (L R_g)), got {type(gw).__name__}."
                )
        elif isinstance(gw, dict):
            missing = set(unique_groups.tolist()) - set(gw.keys())
            if missing:
                raise ValueError(
                    f"group_weights is missing {len(missing)} group(s), e.g. {sorted(missing)[:5]}."
                )
            for g, size in zip(unique_groups, group_sizes):
                if np.shape(gw[g]) != (size, size):
                    raise ValueError(
                        f"group_weights[{g}] has shape {np.shape(gw[g])}, expected ({size}, {size})."
                    )
        elif isinstance(gw, (list, np.ndarray)):
            if len(gw) != len(unique_groups):
                raise ValueError(f"Got {len(gw)} scalar group_weights for {len(unique_groups)} groups.")

    @classmethod
    def _check_spd(cls, W_g, g):
        """Raises if a user-supplied W_g is not a finite, symmetric positive definite matrix."""
        W = W_g.toarray() if sparse.issparse(W_g) else np.asarray(W_g, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"group_weights[{g!r}] must be a square matrix, got shape {W.shape}.")
        if not np.all(np.isfinite(W)):
            raise ValueError(f"group_weights[{g!r}] contains non-finite entries.")
        if np.linalg.norm(W - W.T) > 1e-10 * np.linalg.norm(W):
            raise ValueError(f"group_weights[{g!r}] is not symmetric.")
        cls._check_positive_definite(np.linalg.eigvalsh(W), g)

    @staticmethod
    def _check_positive_definite(eigvals, g):
        """
        Raises if W_g, given by its eigenvalues, is not positive definite. Used for all weight types:
        on user dicts in `_check_spd` (at construction), and on every W_g in the BCD setup, which
        covers 'auto' / None. The primal-dual gap needs W_g^{-1}.
        """
        if eigvals.min() <= 1e-12 * max(eigvals.max(), 0.0):
            raise ValueError(
                f"Weight matrix W_g of group {g} is not positive definite (eigenvalues {eigvals}). "
                "With group_weights='auto' this means the columns of A_g (or L R_g) are linearly dependent."
            )

    # =========================================================================
    #  SOLVER 1: Low-Rank BCD (Optimized for A = L @ R)
    #  Complexity: O(m^2) per step instead of O(n * group_size)
    #  min_x 1/2||LRx - b||^2 + alpha * sum_g ||L R_g x_g||
    # =========================================================================
    def _solve_bcd_low_rank(self, L, R, b):
        n_samples, m_rank = L.shape
        _, n_features = R.shape
        unique_groups = np.unique(self.groups)

        # 1. Precompute M = L.T @ L (m x m)
        if self.verbose: print(f"Precomputing Low-Rank operators (Rank m={m_rank})...")
        if sparse.issparse(L):
            M = (L.T @ L).toarray()
        else:
            M = L.T @ L

        # 2. Precompute Weights & Lipschitz
        # W_g = R_g.T @ M @ R_g
        weights_dict = {}
        if self.verbose: print("Precomputing group weights...")
        W_eig_cache = {}
        Lipschitz_g = {}
        
        if sparse.issparse(R) and R.format != 'csc':
            R = R.tocsc()

        for g in unique_groups:
            idx = np.where(self.groups == g)[0]
            R_g = R[:, idx]
            
            # Compute W_g = (L R_g)^T (L R_g) = R_g^T M R_g
            if sparse.issparse(R_g):
                MR_g = M @ R_g
                W_g = R_g.T @ MR_g
            else:
                W_g = R_g.T @ M @ R_g
            weights_dict[g] = W_g

            # Eigen-decomposition for Proximal Operator
            # Use eigh since W_g is symmetric
            vals, vecs = np.linalg.eigh(W_g)
            vals = np.maximum(vals, 0.0) # Clip numeric noise
            
            W_eig_cache[g] = {'Q': vecs, 'Lambda': vals}
            self._check_positive_definite(vals, g)
            
            # Lipschitz is the spectral norm of W_g
            Lipschitz_g[g] = np.max(vals) if len(vals) > 0 else 0.0

        # 3. Initialization
        x = np.zeros(n_features)
        
        # Latent Residual 'h' = L.T @ (Ax - b)
        # At x=0, h = -L.T @ b
        h = -L.T @ b
        
        active_groups = unique_groups.tolist()

        # Implicit A = L @ R for the primal-dual gap (never formed densely)
        A_op = LinearOperator(
            (n_samples, n_features),
            matvec=lambda v: L @ (R @ v),
            rmatvec=lambda y: R.T @ (L.T @ y),
            dtype=float,
        )

        def print_row(k, rel_change):
            gap = self._relative_primal_dual_gap(x, A_op, b, self.alpha, weights_dict, verbose=False)
            print(f"{k:<5} | {len(active_groups):<8} | {rel_change:<12.2e} | {gap:.6e}")

        # 4. Main Loop
        if self.verbose:
            print(f"{'Pass':<5} | {'Active':<8} | {'Rel Change':<12} | {'Relative Primal-Dual Gap'}")
            print_row(0, 0.0)

        for k in range(1, self.max_iter + 1):
            max_dx = 0.0
            x_old_norm = np.linalg.norm(x)
            
            groups_to_scan = unique_groups if (k == 1 or k % 100 == 0) else active_groups
            new_active_set = []
            
            for g in groups_to_scan:
                idx = np.where(self.groups == g)[0]
                if Lipschitz_g[g] < 1e-12: continue

                R_g = R[:, idx]
                x_g_old = x[idx]
                
                # Gradient: R_g.T @ h
                grad_g = R_g.T @ h
                
                # Proximal Step
                z = x_g_old - (1.0 / Lipschitz_g[g]) * grad_g
                alpha_eff = self.alpha / Lipschitz_g[g]
                
                x_g_new = self._solve_prox_matrix_weight(z, W_eig_cache[g], alpha_eff)
                
                # Update
                delta_x = x_g_new - x_g_old
                norm_delta = np.linalg.norm(delta_x)
                
                if norm_delta > 1e-12:
                    x[idx] = x_g_new
                    # Update Latent Residual h: h += M @ R_g @ delta
                    # This keeps the residual update in the small m x m space
                    update_vec = R_g @ delta_x
                    h += M @ update_vec
                    
                    max_dx = max(max_dx, norm_delta)
                    new_active_set.append(g)
                elif np.linalg.norm(x_g_new) > 0:
                    new_active_set.append(g)
            
            active_groups = np.unique(new_active_set)
            rel_change = max_dx / (x_old_norm + 1e-12)

            if self.verbose and k % 100 == 0: print_row(k, rel_change)

            rel_change_crit = rel_change < self.rel_change_tol and k > 1
            pdg_crit = k % self.gap_check_every == 0 and self._relative_primal_dual_gap(x, A_op, b, self.alpha, weights_dict, verbose=False) < self.rel_primal_dual_gap_tol
            if rel_change_crit or pdg_crit:
                if self.verbose: print_row(k, rel_change)
                print(f"Convergence reached after: {k} iterations!")
                break
            
            if k == self.max_iter:
                print(f"The method failed to converge to tolerance level within {self.max_iter:<5} iterations. I will go ahead and return last iterate nevertheless.")
            
        self.coef_ = x
        return self

    # =========================================================================
    #  SOLVER 2: Standard BCD (For explicit Matrices)
    #  min_x 1/2||Ax - b||^2 + alpha * sum_g sqrt(x_g^T W_g x_g)
    # =========================================================================
    def _solve_bcd_standard(self, A, b, x0=None):
        n_samples, n_features = A.shape
        unique_groups = np.unique(self.groups)
        
        # Optimize sparse format
        if sparse.issparse(A) and A.format != 'csc':
            if self.verbose: print("Converting sparse matrix to CSC...")
            A = A.tocsc()

        # 1. Weights & Lipschitz Setup
        if self.verbose: print("Precomputing Standard BCD weights...")
        W_eig_cache = {}
        L_g = {}

        weights_dict = {}
        if isinstance(self.group_weights, dict):
            print("WARNING: This solves min_x  1/2 ||A x - b||_2^2 + alpha * sum_g sqrt(x_g^T W_g x_g)")
        for g in unique_groups:
            idx = np.where(self.groups == g)[0]
            A_g = A[:, idx]
            
            # Lipschitz
            if sparse.issparse(A_g):
                L_g[g] = sparse.linalg.norm(A_g, 2)**2
            else:
                L_g[g] = np.linalg.norm(A_g, 2)**2
                
            # Weight Matrix W_g
            # If auto, we compute A_g.T @ A_g. If dict, use it. If None, Identity.
            if self.group_weights == 'auto':
                if sparse.issparse(A_g):
                    W_g = (A_g.T @ A_g).toarray()
                else:
                    W_g = A_g.T @ A_g
            elif isinstance(self.group_weights, dict):
                W_g = self.group_weights[g]
            else:
                # Default Identity
                W_g = np.eye(len(idx))
            weights_dict[g] = W_g

            # Eigen-decomposition
            vals, vecs = np.linalg.eigh(W_g)
            vals = np.maximum(vals, 0.0)
            W_eig_cache[g] = {'Q': vecs, 'Lambda': vals}
            self._check_positive_definite(vals, g)

        # 2. Loop
        x = np.zeros(n_features) if x0 is None else np.array(x0, dtype=float)
        r = A @ x - b
        active_groups = unique_groups.tolist()

        def print_row(k, rel_change):
            # Format spec is {value:[align][width][.precision][type]}, e.g. {v:<12.2e} = left-aligned, width 12, 2 decimals, scientific
            gap = self._relative_primal_dual_gap(x, A, b, self.alpha, weights_dict, verbose=False)
            print(f"{k:<5} | {len(active_groups):<8} | {rel_change:<12.2e} | {gap:.6e}")

        if self.verbose: 
            print(f"{'Pass':<5} | {'Active':<8} | {'Rel Change':<12} | {'Relative Primal-Dual Gap'}")
            print_row(0, 0.0)
            
        for k in range(1, self.max_iter + 1):
            max_dx = 0.0
            x_old_norm = np.linalg.norm(x)
            groups_to_scan = unique_groups if (k == 1 or k % 100 == 0) else active_groups
            new_active_set = []
            
            for g in groups_to_scan:
                idx = np.where(self.groups == g)[0]
                if L_g[g] < 1e-12: continue
                
                A_g = A[:, idx]
                x_g_old = x[idx]
                
                # Gradient: A_g.T @ r
                grad_g = A_g.T @ r
                
                z = x_g_old - (1.0 / L_g[g]) * grad_g
                alpha_eff = self.alpha / L_g[g]
                
                x_g_new = self._solve_prox_matrix_weight(z, W_eig_cache[g], alpha_eff)
                
                delta_x = x_g_new - x_g_old
                norm_delta = np.linalg.norm(delta_x)
                
                if norm_delta > 1e-12:
                    x[idx] = x_g_new
                    r += A_g @ delta_x
                    max_dx = max(max_dx, norm_delta)
                    new_active_set.append(g)
                elif np.linalg.norm(x_g_new) > 0:
                    new_active_set.append(g)
                    
            active_groups = np.unique(new_active_set)
            rel_change = max_dx / (x_old_norm + 1e-12)
            
            if self.verbose and k % 100 == 0: print_row(k, rel_change)
            
            rel_change_crit = rel_change < self.rel_change_tol and k > 1
            pdg_crit = k % self.gap_check_every == 0 and self._relative_primal_dual_gap(x, A, b, self.alpha, weights_dict, verbose=False) < self.rel_primal_dual_gap_tol
            if rel_change_crit or pdg_crit:
                if self.verbose: print_row(k, rel_change)
                print(f"Convergence reached after: {k} iterations!")
                # self._check_optimality(x, A, b, self.alpha, weights_dict, groups_to_check=active_groups)
                break
            
            if k == self.max_iter:
                print(f"The method failed to converge to tolerance level within {self.max_iter:<5} iterations. I will go ahead and return last iterate nevertheless.")
            
        self.coef_ = x
        return self

    # =========================================================================
    #  SOLVER 3: FISTA (For Generic LinearOperators)
    #  min_x 1/2||Ax - b||^2 + alpha * sum_g w_g ||x_g||   (scalar w_g only)
    # =========================================================================
    def _solve_fista(self, A, b):
        if self.verbose: print("Starting FISTA Solver...")
        n_features = A.shape[1]
        
        # 1. Lipschitz Estimation
        if self._L is None:
            if self.verbose: print("Estimating Lipschitz Constant...")
            self._L = self._estimate_lipschitz(A)
            
        step_size = 1.0 / self._L
        
        # 2. Weights Setup (Scalar assumption for FISTA)
        unique_groups = np.unique(self.groups)
        if self.group_weights is None:
            w_map = {g: 1.0 for g in unique_groups}
        else:
            w_map = dict(zip(unique_groups, self.group_weights))

        # w_g ||x_g|| = sqrt(x_g^T (w_g^2 I) x_g), i.e. matrix weights W_g = w_g^2 I for the primal-dual gap
        weights_dict = {g: w**2 * np.eye(np.sum(self.groups == g)) for g, w in w_map.items()}

        x = np.zeros(n_features)
        y = np.zeros(n_features)
        t = 1.0

        def print_row(k, rel_change):
            n_active = sum(np.any(x[self.groups == g]) for g in unique_groups)
            gap = self._relative_primal_dual_gap(x, A, b, self.alpha, weights_dict, verbose=False)
            print(f"{k:<5} | {n_active:<8} | {rel_change:<12.2e} | {gap:.6e}")

        if self.verbose:
            print(f"{'Iter':<5} | {'Active':<8} | {'Rel Change':<12} | {'Relative Primal-Dual Gap'}")
            print_row(0, 0.0)

        for k in range(1, self.max_iter + 1):
            x_old = x.copy()
            
            # Gradient
            if isinstance(A, LinearOperator):
                grad = A.rmatvec(A.dot(y) - b)
            else:
                grad = A.T @ (A @ y - b)
                
            z = y - step_size * grad
            
            # Proximal
            x = self._proximal_scalar(z, step_size, w_map)
            
            # Momentum
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x + ((t - 1) / t_new) * (x - x_old)
            t = t_new
            
            rel_change = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-10)
            if self.verbose and k % 100 == 0: print_row(k, rel_change)

            rel_change_crit = rel_change < self.rel_change_tol and k > 1
            pdg_crit = k % self.gap_check_every == 0 and self._relative_primal_dual_gap(x, A, b, self.alpha, weights_dict, verbose=False) < self.rel_primal_dual_gap_tol
            if rel_change_crit or pdg_crit:
                if self.verbose: print_row(k, rel_change)
                print(f"Convergence reached after: {k} iterations!")
                break

            if k == self.max_iter:
                print(f"The method failed to converge to tolerance level within {self.max_iter:<5} iterations. I will go ahead and return last iterate nevertheless.")
                
        self.coef_ = x
        return self

    # =========================================================================
    #  Primal-Dual Gap (Optional for Convergence Check)
    # =========================================================================

    # Primal problem: min_x 1/2||Ax - b||_2^2 + alpha * sum_g sqrt(x_g^T W_g x_g)_2 (usually W_g in R^{3 x 3})
    # This is derived from the group lasso, the code for this can be found in version history.
    def _dual_candidate(self, x, A, b, alpha, weights, optimize=True):
        """
        Calculates a dual feasible candidate as y = (Ax - b) * c, where c is chosen to satisfy the dual constraints.

        Parameters:
        -----------
        x : array-like
            Current primal solution.
        A : array-like or LinearOperator
            Forward operator.
        b : array-like
            Measured data.
        alpha : float
            Regularization parameter.
        weights : dict of array-like
            Group weights. 
            (Idea seems to have been that group labels are not necessarily consecutive)
        optimize : bool, optional
            Scale dual candidate to the optimal value, by default True
        """

        Ax = A.dot(x) if isinstance(A, LinearOperator) else A @ x
        residual = Ax - b
        y = residual.copy()

        
        ATy = A.rmatvec(y) if isinstance(A, LinearOperator) else A.T @ y
        theta_max_groupwise = {}
        for g, w in weights.items():
            idx = np.where(self.groups == g)[0]
            v_g = ATy[idx]
            norm_g = np.sqrt(v_g @ np.linalg.solve(w, v_g))       # ||v_g||_{W_g^{-1}}
            
            theta_max_groupwise[g] = alpha / norm_g

        theta_max = min(theta_max_groupwise.values())
        if optimize and np.any(residual):
            c = - residual.dot(b) / (np.linalg.norm(residual)**2)
            c = np.clip(c, -theta_max, theta_max)
        else:
            c = min(1.0, theta_max)
        return c * y
            

    # VSCode just suggested to me the right certificate, did (at least find) the right math!
    def _primal_dual_gap_computation(self, x, A, b, alpha, weights, optimize=True, verbose=True):
        """
        Computes the primal and dual objective values and the gap for convergence checking.
        """
        # Primal 
        Ax = A.dot(x) if isinstance(A, LinearOperator) else A @ x
        primal_fidelity = 0.5 * np.linalg.norm(Ax - b)**2
        primal_reg = 0
        for g, w in weights.items():
            idx = np.where(self.groups == g)[0]
            x_g = x[idx]
            #if np.any(x_g > 0): print(f"Group {idx} has active elements. {x_g} and contributes {np.linalg.norm(w @ x_g)}")
            primal_reg += np.sqrt(x_g.T @ w @ x_g)
        primal_obj = primal_fidelity + alpha * primal_reg

        # Dual
        dual_candidate = self._dual_candidate(x, A, b, alpha, weights, optimize)
        #dual_obj = - 0.5 * np.linalg.norm(dual_candidate)**2 - dual_candidate.dot(b)
        dual_obj = - 0.5 * np.linalg.norm(dual_candidate + b)**2 + 0.5 * np.linalg.norm(b)**2
        
        gap = primal_obj - dual_obj

        if verbose:
            print(f"Primal: {primal_obj:.6e}, Dual: {dual_obj:.6e}, Gap: {gap:.6e}")
            # print(f"Primal Fidelity: {primal_fidelity:.6e}, Primal Reg: {primal_reg:.6e}")
            # print(f"Dual Candidate Norm: {np.linalg.norm(dual_candidate):.6e}, Dual Residual: {np.linalg.norm(b.T @ dual_candidate):.6e}")
        return gap, primal_obj, dual_obj

    def _primal_dual_gap(self, x, A, b, alpha, weights, optimize=True, verbose=True):
        """
        Computes the primal-dual gap for convergence checking.
        """
        return self._primal_dual_gap_computation(x, A, b, alpha, weights, optimize, verbose)[0]

    def _relative_primal_dual_gap(self, x, A, b, alpha, weights, optimize=True, verbose=True):
            """
            Computes the primal-dual gap for convergence checking.
            """
            gap, primal_obj, _ = self._primal_dual_gap_computation(x, A, b, alpha, weights, optimize, verbose)
            if primal_obj == 0:
                print("Warning: Primal objective is zero, relative gap is undefined. Returning 0.")
                return 0.0
            if gap < 0:
                print("Warning: Primal-dual gap is negative, which should not happen. Returning infinity.")
                return np.inf
            return gap / primal_obj

    def _check_optimality(self, x, A, b, alpha, weights, groups_to_check):
        """
        Debug helper: prints the KKT stationarity condition for the penalty sum_g ||S_g x_g||_2
        (S_g = weights[g], same convention as `_primal_dual_gap`) at the given groups:
            A_g^T (A x - b) + alpha * S_g^T S_g x_g / ||S_g x_g|| = 0,   i.e. lhs / rhs = -1.
        """
        res = A @ x - b
        for g in groups_to_check:
            idx = np.where(self.groups == g)[0]
            S_g, x_g = weights[g], x[idx]
            lhs = A[:, idx].T @ res
            rhs = alpha * (S_g @ x_g) / np.sqrt(x_g.T @ S_g @ x_g)
            print(f"Optimality condition at group {g}: lhs: {lhs}, rhs: {rhs}, lhs/rhs = {lhs / rhs}")

    # =========================================================================
    #  HELPERS: Proximal Operators & Math
    # =========================================================================
    def _solve_prox_matrix_weight(self, v_g, W_eig, alpha):
        """
        Solves argmin_u 0.5||u-v||^2 + alpha * sqrt(u' W u)
        Uses Secular Equation via Root Finding.
        """
        Q, Lambda = W_eig['Q'], W_eig['Lambda']
        
        # Rotate to eigenspace
        z = Q.T @ v_g
        
        # Dual Norm Check (Zero Solution?)
        # Condition: || diag(1/sqrt(Lambda)) * z ||_2 <= alpha
        # Handle zero eigenvalues carefully
        mask = Lambda > 1e-12
        if np.sum(mask) == 0: return v_g # Zero weight matrix?
        
        dual_norm_sq = np.sum((z[mask]**2) / Lambda[mask])
        if np.sqrt(dual_norm_sq) <= alpha:
            return np.zeros_like(v_g)
            
        # Secular Equation
        def secular_eq(c):
            denom = (1 + (alpha / c) * Lambda)**2
            return np.sum((z**2 * Lambda) / denom) - c**2
            
        upper = np.sqrt(np.sum(z**2 * Lambda))
        try:
            # Brentq is robust
            res = root_scalar(secular_eq, bracket=[1e-9, upper], method='brentq')
            c_star = res.root
        except:
            c_star = upper 

        coeffs = 1.0 / (1 + (alpha / c_star) * Lambda)
        return Q @ (coeffs * z)

    def _proximal_scalar(self, x, step_size, w_map):
        x_out = x.copy()
        for g, w in w_map.items():
            idx = np.where(self.groups == g)[0]
            norm_g = np.linalg.norm(x[idx])
            thresh = self.alpha * w * step_size
            scale = max(0, 1 - thresh/norm_g) if norm_g > 0 else 0
            x_out[idx] *= scale
        return x_out

    def _estimate_lipschitz(self, A):
        if sparse.issparse(A) or isinstance(A, np.ndarray):
            return svds(A, k=1, return_singular_vectors=False)[0]**2
        rng = np.random.RandomState(42)
        v = rng.randn(A.shape[1])
        v /= np.linalg.norm(v)
        for _ in range(10):
            v = A.rmatvec(A.dot(v))
            v /= np.linalg.norm(v)
        return np.linalg.norm(A.dot(v))**2 * 1.1