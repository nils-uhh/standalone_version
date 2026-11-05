"""Run with: python -m pytest tests  (from the project folder)"""
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gfw
from gfw.solvers import solve


def toy_problem(M=30, m=60, seed=0):
    """A small average-referenced leadfield with a planted two-source signal."""
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(M, 3 * m))
    L -= L.mean(axis=0, keepdims=True)
    x_true = np.zeros(3 * m)
    x_true[3:6] = [1.0, -0.5, 0.25]
    x_true[60:63] = [-0.3, 0.8, 0.1]
    return L, L @ x_true


@pytest.fixture(params=list(gfw.WEIGHTINGS))
def weights(request):
    return request.param, gfw.WEIGHTINGS[request.param]


def test_solvers_agree(weights):
    _, weighting = weights
    L, d = toy_problem()
    W = weighting(L)
    alpha = 0.05 * gfw.alpha_max(L, W, d)
    x_bcd = solve(L, W, d, alpha, solver="bcd", rel_primal_dual_gap_tol=1e-9)
    x_fista = solve(L, W, d, alpha, solver="fista", tol=1e-12)
    assert gfw.relative_gap(L, W, d, alpha, x_bcd) < 1e-7
    assert gfw.relative_gap(L, W, d, alpha, x_fista) < 1e-7
    objective_bcd = gfw.objective(L, W, d, alpha, x_bcd)
    assert gfw.objective(L, W, d, alpha, x_fista) == pytest.approx(objective_bcd, rel=1e-6)


def test_fista_is_the_default():
    for function in (solve, gfw.morozov):
        assert inspect.signature(function).parameters["solver"].default == "fista"


def test_alpha_max_gives_zero(weights):
    _, weighting = weights
    L, d = toy_problem()
    W = weighting(L)
    amax = gfw.alpha_max(L, W, d)
    for solver in ("bcd", "fista"):
        assert not np.any(solve(L, W, d, 1.001 * amax, solver=solver))
        assert np.any(solve(L, W, d, 0.9 * amax, solver=solver))


def test_gap_matches_groupsparsity():
    from gfw.groupSparsity import groupSparsity

    L, d = toy_problem()
    W = gfw.weights_gamma(L)
    alpha = 0.1 * gfw.alpha_max(L, W, d)
    x = solve(L, W, d, alpha, solver="fista", tol=1e-5)
    groups = np.arange(L.shape[1]) // 3
    model = groupSparsity(alpha=alpha, groups=groups, group_weights={g: W[g] for g in range(W.shape[0])})
    reference = model._relative_primal_dual_gap(x, L, d, alpha, {g: W[g] for g in range(W.shape[0])}, verbose=False)
    assert gfw.relative_gap(L, W, d, alpha, x) == pytest.approx(reference, rel=1e-8)


@pytest.mark.parametrize("solver", ["bcd", "fista"])
def test_warm_start_from_solution_stays_there(solver):
    L, d = toy_problem()
    W = gfw.weights_gamma(L)
    alpha = 0.05 * gfw.alpha_max(L, W, d)
    x = solve(L, W, d, alpha, solver="fista", tol=1e-12)
    warm = solve(L, W, d, alpha, solver=solver, x0=x)
    assert gfw.objective(L, W, d, alpha, warm) == pytest.approx(gfw.objective(L, W, d, alpha, x), rel=1e-8)


@pytest.mark.parametrize("solver", ["bcd", "fista"])
def test_warm_start_to_larger_alpha_is_solved(solver):
    """With no inactive violators, the working set still has to be re-solved at the new alpha."""
    L, d = toy_problem()
    W = gfw.weights_gamma(L)
    amax = gfw.alpha_max(L, W, d)
    x_small = solve(L, W, d, 0.01 * amax, solver=solver)
    warm = solve(L, W, d, 0.1 * amax, solver=solver, x0=x_small)
    cold = solve(L, W, d, 0.1 * amax, solver=solver)
    assert gfw.relative_gap(L, W, d, 0.1 * amax, warm) < 1e-5
    assert gfw.objective(L, W, d, 0.1 * amax, warm) == pytest.approx(gfw.objective(L, W, d, 0.1 * amax, cold), rel=1e-5)


def test_bcd_warm_start_recomputes_residual():
    """A warm start must use r = A x0 - b: the result may not depend on the start."""
    L, d = toy_problem()
    W = gfw.weights_identity(L)
    alpha = 0.05 * gfw.alpha_max(L, W, d)
    cold = solve(L, W, d, alpha, solver="bcd", rel_change_tol=0.0, max_iter=20000)
    start = np.random.default_rng(1).normal(size=L.shape[1])
    warm = solve(L, W, d, alpha, solver="bcd", x0=start, rel_change_tol=0.0, max_iter=20000)
    assert gfw.objective(L, W, d, alpha, warm) == pytest.approx(gfw.objective(L, W, d, alpha, cold), rel=1e-6)


@pytest.mark.parametrize("solver", ["bcd", "fista"])
def test_morozov_hits_target(solver):
    L, d = toy_problem()
    W = gfw.weights_gamma(L)
    target = 0.05 * np.linalg.norm(d)
    x, alpha = gfw.morozov(L, W, d, target, solver=solver, n_bisect=14)
    residual = np.linalg.norm(L @ x - d)
    assert residual <= target
    assert residual > 0.9 * target
    assert 0 < alpha < gfw.alpha_max(L, W, d)


def test_morozov_warns_if_target_unreachable():
    L, d = toy_problem()
    W = gfw.weights_gamma(L)
    with pytest.warns(UserWarning, match="not reached"):
        gfw.morozov(L, W, d, 1e-14 * np.linalg.norm(d), solver="fista", n_bisect=2, alpha_range=0.5)


def test_pinv_rank_reduction():
    L, _ = toy_problem()
    rank = np.linalg.matrix_rank(L)
    assert rank == L.shape[0] - 1
    np.testing.assert_allclose(gfw.weights_pinv(L, rank_reduction=1), gfw.weights_pinv(L, k=rank - 1))
    assert np.trace(gfw.weights_pinv(L, rank_reduction=1), axis1=1, axis2=2).sum() == pytest.approx(rank - 1)


def test_input_data_structure():
    problem = gfw.load_problem()
    simulation = gfw.load_simulation_space()
    assert problem.leadfield.shape == (79, 3 * len(problem.positions))
    assert simulation.leadfield.shape == (79, 3 * len(simulation.positions))
    np.testing.assert_allclose(problem.electrodes, simulation.electrodes)
    np.testing.assert_allclose(np.linalg.norm(problem.normals, axis=1), 1.0)
    np.testing.assert_allclose(problem.leadfield.mean(axis=0), 0.0, atol=1e-12)


def test_simulation_is_off_grid_and_multi_source():
    problem = gfw.load_problem()
    simulation = gfw.load_simulation_space()
    d, indices, moments = gfw.simulate(simulation, n_sources=3, min_separation=30.0, seed=3)
    assert len(indices) == 3 and moments.shape == (3, 3)
    distance = np.linalg.norm(simulation.positions[indices][:, None] - problem.positions[None], axis=2).min(axis=1)
    assert np.all(distance > 1e-3)
    assert abs(d.mean()) < 1e-12


def test_no_imports_from_the_rest_of_the_repo():
    """The folder has to keep working once it is copied out of the repository."""
    code = "import sys, gfw; assert not [m for m in sys.modules if m.startswith(('src.', 'weighting_matrices', 'duneuropy'))]"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
