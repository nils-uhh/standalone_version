"""Evaluation of a reconstruction against the simulated sources."""
import numpy as np              
from scipy.optimize import linear_sum_assignment


def _peaks(amplitude, positions, n_peaks, min_distance):
    peaks = []
    for i in np.argsort(amplitude)[::-1]:
        if amplitude[i] <= 0:
            break
        if all(np.linalg.norm(positions[i] - positions[j]) >= min_distance for j in peaks):
            peaks.append(i)
        if len(peaks) == n_peaks:
            break
    return np.array(peaks, dtype=int)


def localization_errors(x, positions, true_positions, min_distance=20.0):
    """Distance in mm from each true source to its matched reconstructed peak.

    Takes one peak per true source, at least `min_distance` mm apart, and matches
    them to the sources by minimal total distance. Unmatched sources give NaN.
    """
    amplitude = np.linalg.norm(np.asarray(x).reshape(-1, 3), axis=1)
    true_positions = np.atleast_2d(true_positions)
    peaks = _peaks(amplitude, positions, len(true_positions), min_distance)
    errors = np.full(len(true_positions), np.nan)
    if peaks.size:
        cost = np.linalg.norm(true_positions[:, None] - positions[peaks][None], axis=2)
        rows, cols = linear_sum_assignment(cost)
        errors[rows] = cost[rows, cols]
    return errors


def source_metrics(x, positions, true_positions, active_threshold=0.1):
    """Summary of a reconstruction: localization error and number of active sources."""
    amplitude = np.linalg.norm(np.asarray(x).reshape(-1, 3), axis=1)
    errors = localization_errors(x, positions, true_positions)
    return {
        "mean_localization_error_mm": float(np.nanmean(errors)) if np.any(~np.isnan(errors)) else np.nan,
        "max_localization_error_mm": float(np.nanmax(errors)) if np.any(~np.isnan(errors)) else np.nan,
        "n_active": int(np.sum(amplitude > active_threshold * amplitude.max())) if amplitude.max() > 0 else 0,
        "n_nonzero": int(np.sum(amplitude > 0)),
    }
