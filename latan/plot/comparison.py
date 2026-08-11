from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage

from latan.statistics.bootstrap import BootstrapArray


def comparison_matrix(
    data: BootstrapArray,
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure | None, Axes, AxesImage]:
    """Plot signed pairwise bootstrap-component differences in sigma.

    Each cell is the significance of the row component minus the column
    component, retaining correlations through aligned bootstrap replicas.
    """
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("data must have shape (n_bootstrap + 1, n_components)")

    n_components = data.shape[1]
    if labels is None:
        labels = tuple(str(index) for index in range(n_components))
    elif len(labels) != n_components:
        raise ValueError(f"labels has {len(labels)} entries, expected {n_components}")

    difference = data.central[:, None] - data.central[None, :]
    sample_difference = data.samples[:, :, None] - data.samples[:, None, :]
    error = sample_difference.std(axis=0)
    sigma = np.divide(
        difference,
        error,
        out=np.zeros_like(difference, dtype=float),
        where=error != 0.0,
    )
    sigma[(error == 0.0) & (difference > 0.0)] = np.inf
    sigma[(error == 0.0) & (difference < 0.0)] = -np.inf
    np.fill_diagonal(sigma, 0.0)

    cmap = LinearSegmentedColormap.from_list(
        "comparison-significance",
        ((0.0, "green"), (2.0 / 3.0, "orange"), (1.0, "red")),
    )
    norm = Normalize(vmin=0.0, vmax=3.0, clip=True)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    assert fig is not None
    image = ax.imshow(np.abs(sigma), cmap=cmap, norm=norm, aspect="equal")
    colorbar_axes = ax.inset_axes((1.03, 0.0, 0.03, 1.0))
    fig.colorbar(image, cax=colorbar_axes, label="difference significance (σ)")
    bounds = ax.get_position()
    cell_size = min(
        fig.bbox.width * bounds.width,
        fig.bbox.height * bounds.height,
    ) * 72.0 / (fig.dpi * n_components)
    if cell_size >= 18.0:
        annotation_size = float(np.clip(0.35 * cell_size, 6.0, 12.0))
        for row, column in np.ndindex(sigma.shape):
            value = sigma[row, column]
            if value == 0.0:
                text = "0"
            elif np.isposinf(value):
                text = "+∞"
            elif np.isneginf(value):
                text = "−∞"
            else:
                text = f"{value:+.1f}".replace("-", "−")
            ax.text(
                column,
                row,
                text,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=annotation_size,
            )

    indices = np.arange(n_components)
    ax.set_yticks(indices, labels=labels)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, labelsize=8, pad=6)
    x_labels = ax.secondary_xaxis("top")
    x_labels.set_xticks(indices, labels=labels)
    x_labels.tick_params(length=0, pad=6, labelsize=8)
    for label in x_labels.get_xticklabels():
        label.set_rotation(270)
    return fig, ax, image
