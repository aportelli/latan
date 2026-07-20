from typing import Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage

from latan.plot.correlated_data import correlations
from latan.statistics.xy_data import XYData


def xy_correlations(
    data: XYData,
    *,
    include: Sequence[Tuple[float | None, float | None]] | None = None,
    exclude: Sequence[Tuple[float | None, float | None]] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure | None, Axes, AxesImage]:
    """Plot the selected correlation matrix of `XYData`.

    Stochastic x coordinates contribute their unique raw values referenced by
    the selected observations. Y coordinates contribute their selected
    observations. Exact x coordinates have no covariance and are therefore
    omitted from the matrix.

    Args:
        data: X/y data whose joint covariance is displayed.
        include: Closed x-coordinate ranges defining retained observations.
        exclude: Closed x-coordinate ranges removed after `include`.
        ax: Optional axis on which to draw the plot.

    Returns:
        The figure, main axis, and correlation-matrix image.
    """
    mask = data.point_mask(include, exclude)
    point_indices = np.flatnonzero(mask)

    quantities: list[int] = []
    selections: list[np.ndarray] = []
    names: list[str] = []
    for i in range(data.x_ndim):
        if data.is_exact_x(i):
            continue
        quantity = data.x_indices[i]
        assert quantity is not None
        quantities.append(quantity)
        selections.append(np.unique(data.x_map(i)[point_indices]))
        names.append(data.x_names[i])
    for i, quantity in enumerate(data.y_indices):
        quantities.append(quantity)
        selections.append(point_indices)
        names.append(data.y_names[i])

    return correlations(
        data.data,
        indices=selections,
        quantities=quantities,
        names=names,
        ax=ax,
    )
