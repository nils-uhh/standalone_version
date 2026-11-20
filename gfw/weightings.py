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


def get_delta():
    d = 0 # delta in intervall [0,1]   
    return d

def combined_weights_exp(L, delta, rank_reduction=1, k=None):      # Rechenlastige Interpolation bei vielen Quellen
    # delta nur gebraucht, wenn sim_löser_rep_schleife genutzt 
    #for i in range (L.shape[1]):
    #    L[:,i] = L[:,i]/np.linalg.norm(L[:,i],2) 

    
    green = weights_gamma(L)
    proj = weights_pinv(L, rank_reduction=1, k=None)
    green_genormt = green / np.linalg.norm(green, ord="fro", axis=(-1,-2),keepdims=True)
    proj_genormt = proj / np.linalg.norm(proj,ord="fro", axis=(-1,-2),keepdims=True)
    # delta = get_delta()   nicht gebraucht, wenn sim_löser_rep_schleife genutzt wird. 
    #print("green",np.average(green[:3]))
    #print("proj",np.average(proj[:3]))
    #print()
    #print("green norm",np.average(green_genormt[:3]))
    #print("proj norm",np.average(proj_genormt[:3])) 
    if delta != 0 and delta != 1:
        return (1.0-delta) * green_genormt + delta * proj_genormt
    elif delta == 0:
        return green
    elif delta == 1:
        return proj

# d = 0,75 ziemlich gut, am konstantesten 
# only green auch gut aber regelmäßig große ausreißer, ist aber am genausten
# ab n = 2 ist mean_loc_error = 30, viel besser wirds nicht, egal mit welchem weighting


def get_alpha_scaled_by_svd(L):     # mit KI gemacht 
    # Get scaled alpha based on SVD

    s = np.linalg.svd(L, compute_uv=False)
    
    # 2. Der größte Singulärwert (sigma_max)
    sigma_max = s[0]
    
    # 3. Referenzskala für die Eigenwerte von LL^T (sigma_max^2)
    lambda_max = sigma_max ** 2
    
    # 4. alpha skalieren (z. B. 1% oder 10% des Maximums als Ausgangspunkt)
    scale_factor = 1        # wie stark lambda_max skalieren, hierbei ausprobieren oder clever wählen
    alpha = scale_factor * lambda_max
    
    return alpha


def combined_weights_cheap(L,alpha):     # mit KI gemacht
    # Push-Through-Identität aus PDF-Datei
    # Lohnt sich rechentechnisch nur wenn Anzahl der Quellen >> Anzahl Sensoren, dann Invertierbarkeit der Matrix schneller
    # alpha = get_alpha_scaled_by_svd(L) # [0,inf) Gewichtung 0: Proj , inf: Green
    N, num_cols = L.shape
    m = num_cols // 3
        
    LLt = L @ L.T

    I_N = np.eye(N)
    inv_term = np.linalg.solve(LLt + alpha * I_N, L)    # zu invertierender Term * L
    
    W = np.zeros((m, 3, 3))

    L_groups = L.reshape(N, m, 3)          # Form: (N, m, 3)
    inv_term_groups = inv_term.reshape(N, m, 3) # Form: (N, m, 3)
    
    for g in range(m):
        L_g = L_groups[:, g, :]          # Spalten für Quelle g: Form (N, 3)
        inv_g = inv_term_groups[:, g, :] # Entsprechender inverser Teil: Form (N, 3)
        
        # Gleichung (9): W_g = (1 + alpha) * L_g^T * (LL^T + alpha * I)^(-1) * L_g[cite: 2]
        W[g] = (1.0 + alpha) * (L_g.T @ inv_g)
        
    return W
    


WEIGHTINGS = {
    "identity": weights_identity,
    "greens_function": weights_gamma,
    "truncated_pinv": weights_pinv,
    "combined_expensive": combined_weights_exp,
    "combined_cheap": combined_weights_cheap,
}
