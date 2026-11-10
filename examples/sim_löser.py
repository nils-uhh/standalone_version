#!/usr/bin/env python3
"""A single noise-free reconstruction with the Green's function weighting.

    python examples/01_single_reconstruction.py
    python examples/01_single_reconstruction.py --n-sources 3 --weighting truncated_pinv

Writes a figure and a .vtu file for ParaView to the output folder.
"""

# Nachbauen zum solven der Simulationen
# notizen auf zettel

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import gfw
from gfw.data import OUTPUT_DIR
from gfw.plotting import plot_reconstruction
from gfw.vtu_utils import save_points_to_vtu_file, save_surface_to_vtu_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weighting", default="greens_function", choices=list(gfw.WEIGHTINGS))
    parser.add_argument("--solver", default="fista", choices=["bcd", "fista"])
    parser.add_argument("--n-sources", type=int, default=1)
    parser.add_argument("--alpha-rel", type=float, default=0.01, help="alpha as a fraction of alpha_max")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The reconstruction grid and the simulation grid differ, which avoids the same-grid inverse crime.
    problem = gfw.load_problem()
    simulation = gfw.load_simulation_space()
    L = problem.leadfield
    W = gfw.WEIGHTINGS[args.weighting](L)

    d, indices, moments = gfw.simulate(simulation, n_sources=1, min_separation=30.0, seed=args.seed) # n = 1,2,3
    # alpha = args.alpha_rel * gfw.alpha_max(L, W, d)         # dann auch nicht mehr gebraucht, da morozov alpha bestimmt??

    target = gfw.simulation.find_target(d)
    print("Target ist: ",target)
    x,alpha = gfw.morozov(L, W, d, target)       # ersetzen durch morozov, braucht noise - siehe datei
    print()

    true_positions = simulation.positions[indices]
    print(f"weighting {args.weighting}, solver {args.solver}, alpha {alpha:.3e}")
    print(f"relative primal-dual gap {gfw.relative_gap(L, W, d, alpha, x):.2e}")
    print(f"relative residual {np.linalg.norm(L @ x - d) / np.linalg.norm(d):.2e}")
    print(gfw.source_metrics(x, problem.positions, true_positions))

    #OUTPUT_DIR.mkdir(exist_ok=True)
    #stem = f"reconstruction_{args.weighting}_{args.solver}_{args.n_sources}src"
    #moments_x = x.reshape(-1, 3)
   # save_points_to_vtu_file(
   #     problem.positions, OUTPUT_DIR / f"{stem}.vtu",
   #     scalar_data={"amplitude": np.linalg.norm(moments_x, axis=1)},
   #     vector_data={"moment": moments_x},
   # )
   # for surface in ("head", "cortex"):
  #      save_surface_to_vtu_file(*gfw.load_surface(surface), OUTPUT_DIR / f"{surface}_surface.vtu")
   # np.savez(OUTPUT_DIR / f"{stem}.npz", x=x, alpha=alpha, true_positions=true_positions, true_moments=moments)
   # plot_reconstruction(x, problem.positions, true_positions, OUTPUT_DIR / f"{stem}.png", head=gfw.load_surface("head"),
   #                     title=f"{args.weighting}, {args.solver}, alpha = {alpha:.2e}")
   # print(f"Wrote {OUTPUT_DIR / stem}.vtu, .npz and .png, and the head and cortex surfaces as .vtu")


if __name__ == "__main__":
    main()