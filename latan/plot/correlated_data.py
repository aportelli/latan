from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage

from latan.statistics.correlated_data import CorrelatedData
from latan.statistics.correlation import cdr, cov_to_corr


def correlations(
    data: CorrelatedData,
    *,
    indices: Sequence[npt.NDArray] | None = None,
    quantities: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure | None, Axes, AxesImage]:
    """Plot a selected correlation matrix from `CorrelatedData`.

    Args:
        data: Correlated vector data whose covariance is displayed.
        indices: Optional ordered integer indices for each selected quantity.
            Omitting this selects every value in each selected quantity.
        quantities: Optional ordered quantity indices. Omitting this uses all
            quantities in stored order.
        names: Optional names for the selected quantity blocks. Defaults to
            `q0`, `q1`, and so on.
        ax: Optional axis on which to draw the plot.

    Returns:
        The figure, main axis, and correlation-matrix image.
    """
    if quantities is None:
        quantities = tuple(range(data.n_quantities))
    if indices is None:
        indices = [np.arange(data.size(i), dtype=np.intp) for i in quantities]
    if names is None:
        names = [f"q{i}" for i in quantities]
    if len(names) != len(quantities):
        raise ValueError(f"names has {len(names)} entries, expected {len(quantities)}")

    _, covariance = data.total_mean_cov(indices=indices, quantities=quantities)
    correlation, _ = cov_to_corr(covariance)
    block_sizes = [selection.size for selection in indices]
    boundaries = np.cumsum(block_sizes)[:-1] - 0.5
    centers = np.cumsum(block_sizes) - np.asarray(block_sizes) / 2.0 - 0.5

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    assert fig is not None
    image = ax.imshow(correlation, cmap="bwr", vmin=-1.0, vmax=1.0)
    fig.text(
        0.92,
        0.94,
        rf"$\mathrm{{CDR}} = {cdr(correlation):.2f}~\mathrm{{dB}}$",
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.set_xticks([])
    ax.set_yticks(centers, labels=names)
    ax.tick_params(axis="y", length=0, labelsize=8, pad=6)
    for boundary in boundaries:
        ax.axhline(boundary, color="black", linewidth=0.5)
        ax.axvline(boundary, color="black", linewidth=0.5)

    x_names = ax.secondary_xaxis("top")
    x_names.set_xticks(centers, labels=names)
    x_names.tick_params(length=0, pad=6, labelsize=8)
    for label in x_names.get_xticklabels():
        label.set_rotation(270)
    fig.colorbar(image, ax=ax, label="correlation")
    return fig, ax, image
