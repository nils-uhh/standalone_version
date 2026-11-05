#!/usr/bin/env python3
"""Solve one noise-free problem with both solvers and compare the results.

    python examples/02_compare_solvers.py --weighting identity

Both solvers minimise the same function, so their objectives must agree.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import gfw


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weighting", default="greens_function", choices=list(gfw.WEIGHTINGS))
    parser.add_argument("--n-sources", type=int, default=1)
    parser.add_argument("--alpha-rel", type=float, default=0.01, help="alpha as a fraction of alpha_max")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = gfw.load_problem()
    simulation = gfw.load_simulation_space()
    L = problem.leadfield
    W = gfw.WEIGHTINGS[args.weighting](L)
    d, indices, _ = gfw.simulate(simulation, n_sources=args.n_sources, min_separation=30.0, seed=args.seed)
    alpha = args.alpha_rel * gfw.alpha_max(L, W, d)

    solutions = {}
    print(f"{'solver':<8}{'time [s]':>10}{'objective':>18}{'rel. gap':>11}{'loc. error [mm]':>17}{'nonzero':>9}")
    for solver in ("fista", "bcd"):
        start = time.perf_counter()
        x = gfw.solve(L, W, d, alpha, solver=solver)
        elapsed = time.perf_counter() - start
        solutions[solver] = x
        metrics = gfw.source_metrics(x, problem.positions, simulation.positions[indices])
        print(f"{solver:<8}{elapsed:>10.2f}{gfw.objective(L, W, d, alpha, x):>18.10e}{gfw.relative_gap(L, W, d, alpha, x):>11.2e}"
              f"{metrics['mean_localization_error_mm']:>17.2f}{metrics['n_nonzero']:>9}")
    difference = np.linalg.norm(solutions["fista"] - solutions["bcd"]) / np.linalg.norm(solutions["bcd"])
    print(f"relative difference between the solutions: {difference:.2e}")


if __name__ == "__main__":
    main()
