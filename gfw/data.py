"""Load the input folder. Files keep the pipeline's .npz structure."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


@dataclass
class Problem:
    leadfield: np.ndarray               # (n_sensors, 3 n_sources), columns x, y, z per source, uV/nAm
    positions: np.ndarray               # (n_sources, 3), mm
    normals: np.ndarray                 # (n_sources, 3), unit length
    triangles: np.ndarray               # (n_triangles, 3), source space surface
    electrodes: np.ndarray              # (n_sensors, 3), mm
    candidates: Optional[np.ndarray] = None  # source indices usable for simulation


def _load(leadfield_file, sourcespace_file, input_dir):
    input_dir = Path(input_dir)
    with np.load(input_dir / leadfield_file) as lf, np.load(input_dir / sourcespace_file) as ss:
        return Problem(
            leadfield=lf["eeg_leadfield"],
            positions=lf["source_positions"],
            normals=ss["nn"],
            triangles=ss["tris"],
            electrodes=lf["electrode_positions"],
        )


def load_problem(input_dir=INPUT_DIR):
    """Reconstruction source space (about 5k sources)."""
    return _load("eeg_leadfield_archive.npz", "sourcespace_reconstruction.npz", input_dir)


def load_simulation_space(input_dir=INPUT_DIR, point_equality_tolerance=1e-3):
    """Simulation source space (about 20k sources).

    `candidates` excludes all sources that coincide with a reconstruction source,
    so simulating from it does not commit the same-grid inverse crime.
    """
    sim = _load("eeg_leadfield_archive_simulation.npz", "sourcespace_simulation.npz", input_dir)
    with np.load(Path(input_dir) / "eeg_leadfield_archive.npz") as lf:
        distance, _ = cKDTree(lf["source_positions"]).query(sim.positions)
    sim.candidates = np.flatnonzero(distance > point_equality_tolerance)
    return sim


def load_surface(name="head", input_dir=INPUT_DIR):
    """Return (nodes, triangles) of the "head" or the "cortex" surface."""
    files = {"head": "head_surface.npz", "cortex": "surface_mesh.npz"}
    with np.load(Path(input_dir) / files[name]) as surface:
        return surface["boundary_nodes"], surface["boundary_triangles"]
