"""Quick matplotlib views. For publication-quality views, export a .vtu and use ParaView."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_reconstruction(x, positions, true_positions, filename, head=None, title=None):
    """Three orthogonal projections of the source amplitudes, saved to `filename`.

    true_positions are marked with crosses; head is an optional (nodes, triangles)
    surface, of which the nodes are drawn as a faint outline.
    """
    amplitude = np.linalg.norm(np.asarray(x).reshape(-1, 3), axis=1)
    active = amplitude > 0
    true_positions = np.atleast_2d(true_positions)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    views = [((0, 1), "top", "x [mm]", "y [mm]"), ((0, 2), "front", "x [mm]", "z [mm]"), ((1, 2), "side", "y [mm]", "z [mm]")]
    scatter = None
    for ax, ((i, j), name, xlabel, ylabel) in zip(axes, views):
        if head is not None:
            ax.scatter(head[0][::10, i], head[0][::10, j], s=0.3, color="0.88", rasterized=True)
        ax.scatter(positions[:, i], positions[:, j], s=0.5, color="0.7", rasterized=True)
        order = np.argsort(amplitude[active])
        scatter = ax.scatter(positions[active][order, i], positions[active][order, j], c=amplitude[active][order],
                             s=28, cmap="viridis", vmin=0, vmax=amplitude.max() if amplitude.max() > 0 else 1)
        ax.scatter(true_positions[:, i], true_positions[:, j], marker="x", s=70, color="#d1362f", linewidths=2, label="true source")
        ax.set(title=name, xlabel=xlabel, ylabel=ylabel, aspect="equal")
    axes[0].legend(frameon=False, loc="lower right")
    fig.colorbar(scatter, ax=axes, label="dipole moment [nAm]", shrink=0.85)
    if title:
        fig.suptitle(title)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
