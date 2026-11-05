"""Weighted group lasso for EEG source reconstruction on static data."""
from .data import Problem, load_problem, load_simulation_space, load_surface
from .gap import objective, relative_gap
from .metrics import localization_errors, source_metrics
from .morozov import morozov
from .simulation import average_reference, simulate
from .solvers import alpha_max, solve
from .weightings import WEIGHTINGS, weights_gamma, weights_identity, weights_pinv

__all__ = [
    "Problem", "load_problem", "load_simulation_space", "load_surface",
    "objective", "relative_gap",
    "localization_errors", "source_metrics",
    "morozov",
    "average_reference", "simulate",
    "alpha_max", "solve",
    "WEIGHTINGS", "weights_gamma", "weights_identity", "weights_pinv",
]
