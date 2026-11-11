"""Group weights W_g for the penalty sum_g sqrt(x_g^T W_g x_g).

A weighting is a function taking the leadfield L (n_sensors, 3 n_sources) and
returning an array of shape (n_sources, 3, 3) with one SPD block per source.
To try a new weighting, write such a function; nothing else needs to change.
"""
import numpy as np


def groups_of(L):
    M, n = L.shape
    assert n % 3 == 0, "number of columns must be 3m"
    return L.reshape(M, n // 3, 3).transpose(1, 0, 2)           # (m, M, 3)


def weights_identity(L):
    """W_g = I: the unweighted group lasso."""
    m = L.shape[1] // 3
    return np.broadcast_to(np.eye(3), (m, 3, 3)).copy()


def weights_gamma(L):       # Green weighting
    """W_g = L_g^T L_g: the Green's function weighting."""
    Lg = groups_of(L)
    return np.einsum("gai,gaj->gij", Lg, Lg)


def numerical_rank(L, rtol=1e-10):
    s = np.linalg.svd(L, compute_uv=False)
    return int(np.sum(s > rtol * s[0]))


def weights_pinv(L, rank_reduction=1, k=None):      # Projection weighting
    """W_g = (L_k^+ L)_g^T (L_k^+ L)_g: the truncated pseudo-inverse weighting.

    The rank used is k = numerical_rank(L) - rank_reduction, unless k is given.
    The leadfield is average-referenced, so its rank is n_sensors - 1, and
    rank_reduction=0 means no truncation.

    With L = U S V^T, L_k^+ L = V_k V_k^T =: P is an orthogonal projection,
    hence P_g^T P_g = P[g, g] = V_k[g] V_k[g]^T  (cheap, no n x n matrix).
    """
    _, s, Vt = np.linalg.svd(L, full_matrices=False)
    if k is None:
        k = int(np.sum(s > 1e-10 * s[0])) - rank_reduction
    if not 3 <= k <= len(s):
        raise ValueError(f"truncation rank k must be in [3, {len(s)}], got {k}")
    Vk = Vt[:k].T.reshape(-1, 3, k)                             # (m, 3, k)
    return np.einsum("gik,gjk->gij", Vk, Vk)


def combined_weights_exp(L, rank_reduction=1, k=None):      # Rechenlastige Interpolation bei vielen Quellen
    green = weights_gamma(L)
    proj = weights_pinv(L, rank_reduction=1, k=None)
    green_genormt = green / np.linalg.norm(green, ord="fro", axis=(-1,-2),keepdims=True)
    proj_genormt = proj / np.linalg.norm(proj,ord="fro", axis=(-1,-2),keepdims=True)
    delta = 0.75 # delta in intervall [0,1]   # 0,25 / 0,5 / 0,75  
    return (1.0-delta) * green_genormt + delta * proj_genormt

# d = 0,75 ziemlich gut, am konstantesten 
# only green auch gut aber regelmäßig große ausreißer, ist aber am genausten


def combined_weights_cheap():




    
    return


WEIGHTINGS = {
    "identity": weights_identity,
    "greens_function": weights_gamma,
    "truncated_pinv": weights_pinv,
    "combined_expensive": combined_weights_exp,
    "combined_cheap": combined_weights_cheap,
}
