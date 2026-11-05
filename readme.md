# Weighted group lasso for EEG source reconstruction

A self-contained Python project for trying out different weightings on one EEG
head model. It reconstructs the sources behind a single time point of EEG data.
There is nothing to compile, and DUNEuro is not needed: the leadfields are
already computed and stored in `input/`.

## Getting started

The required packages are listed in `environment.yml` and installed with
[mamba](https://mamba.readthedocs.io). This only has to be done once.

```bash
mamba env create -f environment.yml
mamba activate gfw
python -m pytest tests                       # should take under a minute
python examples/01_single_reconstruction.py  # noise-free reconstruction, writes to output/
```

The example prints the localization error and writes to `output/`: a `.png`, a `.npz`
with the result, and `.vtu` files to open together in ParaView. Show the reconstruction
with the "Point Gaussian" representation, and lower the opacity of `head_surface.vtu`
to see inside.

Options can be listed with `--help`. For example, three sources with another weighting:

```bash
python examples/01_single_reconstruction.py --n-sources 3 --weighting truncated_pinv
```

## The problem

The sensors measure `d = L x`. `L` is the leadfield with one row per electrode and
three columns per source position (x, y and z moment, in that order). There are far
fewer electrodes (79) than unknowns (14,820), so the data alone do not determine `x`.
We therefore solve

    min_x  1/2 ||L x - d||^2  +  alpha * sum_g sqrt(x_g^T W_g x_g)

where each group `g` is the three columns of one source position. The penalty makes
whole source positions zero, which gives a sparse reconstruction. `alpha` sets how
strongly it acts, and the 3x3 matrices `W_g` are the **weighting**, the subject of
this project.

## What is in the folder

| Path | Content |
|---|---|
| `gfw/weightings.py` | The weightings: `identity`, `greens_function`, `truncated_pinv` |
| `gfw/solvers.py` | `solve(L, W, d, alpha, solver=...)`, `alpha_max(L, W, d)` |
| `gfw/groupSparsity.py` | The BCD solver |
| `gfw/morozov.py` | `morozov(L, W, d, target)`, Morozov's discrepancy principle |
| `gfw/simulation.py` | `simulate(...)`: noise-free potentials of one or more dipoles |
| `gfw/metrics.py` | Localization error and number of active sources |
| `gfw/gap.py` | Objective and primal-dual gap, to check any solution |
| `gfw/vtu_utils.py`, `gfw/plotting.py` | `.vtu` export for ParaView, quick matplotlib figures |
| `examples/` | `01`: one reconstruction; `02`: BCD and FISTA on one problem |
| `input/` | The head model data (see below) |
| `output/` | Your results |

### Input data

All `.npz` files have the same structure as in the main pipeline (DUNEuro column order).
Load them through `gfw.load_problem()`, `gfw.load_simulation_space()` and `gfw.load_surface()`.

| File | Content |
|---|---|
| `eeg_leadfield_archive.npz` | Reconstruction leadfield, 4,940 sources, in uV/nAm |
| `sourcespace_reconstruction.npz` | Its source positions, cortex normals and surface triangles |
| `eeg_leadfield_archive_simulation.npz` | Simulation leadfield, 19,754 sources |
| `sourcespace_simulation.npz` | Its source positions and normals |
| `head_surface.npz` | Outer skin surface, for visualization |
| `surface_mesh.npz` | Inner surfaces around the cortex, for visualization |

This is data from a real subject, so please do not share it outside the group.

### Why there are two leadfields

If you simulate data with the same matrix you reconstruct with, the problem becomes
unrealistically easy. This is called an *inverse crime*. So `simulate` only draws from
the 14,816 simulation sources that do not lie on the reconstruction grid. Both leadfields
still come from the same head model, so errors in the head model itself are not covered.

## Things worth knowing

- **Choosing `alpha`:** its useful size differs by orders of magnitude between weightings,
  so compare weightings relative to `alpha_max(L, W, d)`, the smallest `alpha` giving `x = 0`.
- **Average reference:** every column of the leadfield has zero mean. Any data you
  reconstruct should too; `average_reference` does this.
- **Checking a solution:** `relative_gap` measures distance from the optimum and does not
  depend on the solver. A small gap means the optimisation is done, not that the
  reconstruction is good. For that, use the metrics.
- **Solvers:** both minimise the same function. FISTA is the default everywhere.
  BCD is there for validation and can take minutes on a problem FISTA solves in about a second.
- **Warm starts:** `solve(..., x0=previous_x)` continues from an earlier solution, which helps
  when solving several nearby values of `alpha`.
- **Truncated pseudo-inverse:** `weights_pinv(L, rank_reduction=1)` removes one rank from
  the numerical rank (78 here), so it uses `k = 77`. Pass another `rank_reduction`, or `k`
  directly. `rank_reduction=0` means no truncation.
- **Adding a weighting:** write a function that takes `L` and returns an array of shape
  `(n_sources, 3, 3)` of symmetric positive definite matrices. Then pass the result as `W`,
  or add it to `WEIGHTINGS`.

## Your task

Everything so far is noise-free, which real measurements never are. Your task is to find
out how well the weightings hold up with noise:

1. **Add noise** to the simulated potentials. Think about how to define the noise level
   and how to make it reproducible.
2. **Choose `alpha` with Morozov's discrepancy principle.** `gfw.morozov` needs a `target`
   residual norm. Work out what it should be for your noise.
3. **Run a parameter study** comparing the weightings. Decide what to vary, how many
   repetitions you need and how to report the spread, not just the average.

Results go in `output/`. Please write your own scripts rather than editing `gfw/`, except
to add weightings.
