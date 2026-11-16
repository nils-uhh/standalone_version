#!/usr/bin/env python3
"""A single noise-free reconstruction with the Green's function weighting.

    python examples/01_single_reconstruction.py
    python examples/01_single_reconstruction.py --n-sources 3 --weighting truncated_pinv
s
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

k = 100 # wie viele durchgänge von simulationen 
def main():
    # Startwerte zur Abspeicherung der größten und kleinsten mean/max loc error
    biggest_1 = 0
    biggest_2 = 0
    biggest_3 = 0
    smallest_1 = 500
    smallest_2 = 500
    smallest_3 = 500
    mean_error_sum = 0
    max_error_sum = 0
    safe = np.array([])
    
    for i in range(k):
    
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
        W = gfw.WEIGHTINGS["combined_expensive"](L)     #       "greens_function": weights_gamma,
                                                    #       "truncated_pinv": weights_pinv,
                                                    #       "combined_expensive": combined_weights_exp,
                                                    #       "combined_cheap": combined_weights_cheap,


        d, indices, moments = gfw.simulate(simulation, n_sources=1, min_separation=30.0, seed=args.seed) # n = 1,2,3
        target = gfw.simulation.find_target(d)
        x,alpha = gfw.morozov(L, W, d, target)      
        true_positions = simulation.positions[indices]
        speicher = gfw.metrics.source_metrics_rep(x, problem.positions, true_positions)


        if speicher[0] > biggest_1:
            biggest_1 = speicher[0]
        elif speicher[0] > biggest_2:
            biggest_2 = speicher[0]
        elif speicher[0] > biggest_3:
            biggest_3 = speicher[0]
        else:
            pass

        if speicher[0] < smallest_1:
            smallest_1 = speicher[0]
        elif speicher[0] < smallest_2:
            smallest_2 = speicher[0]
        elif speicher[0] < smallest_3:
            smallest_3 = speicher[0]
        else:   
            pass

        mean_error_sum = mean_error_sum + speicher[0]
        max_error_sum = max_error_sum + speicher[1]


        # speichert den aktuellen durchlauf in datei zum plotten nachher
        # ist es vlt sinnvoller in eine datei alle verschiedenen deltas zu speichern? 

        safe.append(speicher[0])
        print(i+1 , "ter Durchgang")


    # speichert Ergebnisse als npz 
    delta =  gfw.get_delta()
    snr = gfw.get_snr() # to be created in "simulation.py"
    dateiname = f"run_delta:{delta}_SNR:{snr}.npz"

    aktueller_ordner = Path(__file__).resolve().parent
    projekt_hauptordner = aktueller_ordner.parent
    # für jedes delta ncohmal eigenen unterordner, dann ist einfacher
    # darauf zuzugreifen für den plot und übersichtlicher
    dateipfad = projekt_hauptordner / "Ergebnisse_sim_rep" / dateiname 
    np.savez(dateipfad, daten=safe)


    print("Biggest mean_loc_error:",round(biggest_1,5), ",",round(biggest_2,5), ",",round(biggest_3,5))
    print("Smallest mean_loc_error:",round(smallest_1,5),",",round(smallest_2,5),",",round(smallest_3,5))
    print("Durchschnittlicher mean_loc_error ist ", round(mean_error_sum/k,5))
    print("Durchschnittlicher max_loc_error ist ", round(max_error_sum/k,5))


if __name__ == "__main__":
    main()