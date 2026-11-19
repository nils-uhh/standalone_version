"""Noise-free forward simulation on the finer simulation source space."""
import numpy as np
import math as m


def average_reference(potential):
    """Subtract the mean over sensors; the leadfields are average-referenced."""
    return potential - potential.mean(axis=-1, keepdims=True)


def get_snr(snr):
    # snr = 10 # hier gewollten SNR-Wert einsetzen außer wenn sim_löser_rep_schleife läuft
    return snr


def add_noise(potential, sigma):   
    # Noise    
    mittel = 0      # Mittelwert der Normalverteilung
    # noise jetzt basierend auf gewollten SNR-Wert dazuadddieren

    noise = np.random.normal(mittel , sigma, size=potential.shape)
    noised_potential = potential + noise

    return noised_potential




def simulate(sim, snr=None, n_sources=1, orientation="normal", min_separation=0.0, moments=None, indices=None, seed=0):  # snr nur als parameter zulassen, wenn sim_löser_rep_schleife läuft
    """Simulate the potential of `n_sources` dipoles of unit strength.

    sim            : Problem from load_simulation_space
    orientation    : "normal" (along the cortex normal) or "random"
    min_separation : minimal distance between the drawn sources in mm
    indices        : source indices to use instead of drawing them
    moments        : (n_sources, 3) moments to use instead of `orientation`

    Returns (potential, indices, moments).
    """
    rng = np.random.default_rng(seed)
    if indices is None:
        indices = _draw_sources(rng, sim, n_sources, min_separation)
    indices = np.asarray(indices) # n lang 
    if moments is None:
        if orientation == "normal":
            moments = sim.normals[indices].copy()
        elif orientation == "random":
            moments = rng.normal(size=(len(indices), 3))    # (n,3)
            moments /= np.linalg.norm(moments, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown orientation '{orientation}'. Use 'normal' or 'random'.")
    moments = np.asarray(moments)
    potential = sum(sim.leadfield[:, 3 * i:3 * i + 3] @ q for i, q in zip(indices, moments))
    #    liegt in R^79                                 3x1    
    #print(potential)
    #print()

    snr_level = get_snr(snr)
    länge = len(potential)
    sigma = m.sqrt(((sum(potential**2))/länge)/(10**(snr_level/10)))  # Formel für sigma based on SNR-Wert
    noised_potential = add_noise(potential,sigma)
    target = m.sqrt(länge) * sigma          #  Target für Morozov
    return noised_potential, indices, moments, target   


def _draw_sources(rng, sim, n_sources, min_separation):
    chosen = []
    for i in rng.permutation(sim.candidates):
        if all(np.linalg.norm(sim.positions[i] - sim.positions[j]) >= min_separation for j in chosen):
            chosen.append(i)
        if len(chosen) == n_sources:
            return np.array(chosen)
    raise ValueError(f"Could not place {n_sources} sources {min_separation} mm apart.")
