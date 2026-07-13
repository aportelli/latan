import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure, SubFigure


def pvalue_cmap() -> LinearSegmentedColormap:
    red, orange, green = map(to_rgb, ("red", "orange", "green"))
    two_sig = math.erfc(2 / math.sqrt(2))
    three_sig = math.erfc(3 / math.sqrt(2))
    cmap = LinearSegmentedColormap(
        "pvalue",
        {
            "red": [
                (0.0, red[0], red[0]),
                (three_sig, red[0], red[0]),
                (two_sig, red[0], red[0]),
                (two_sig, orange[0], orange[0]),
                (1.0, green[0], green[0]),
            ],
            "green": [
                (0.0, red[1], red[1]),
                (three_sig, red[1], red[1]),
                (two_sig, red[1], red[1]),
                (two_sig, orange[1], orange[1]),
                (1.0, green[1], green[1]),
            ],
            "blue": [
                (0.0, red[2], red[2]),
                (three_sig, red[2], red[2]),
                (two_sig, red[2], red[2]),
                (two_sig, orange[2], orange[2]),
                (1.0, green[2], green[2]),
            ],
            "alpha": [
                (0.0, 0.0, 0.0),
                (three_sig, 0.0, 0.0),
                (three_sig, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ],
        },
        N=8192,
    )
    cmap.set_under((0, 0, 0, 0))
    cmap.set_over("green")
    return cmap


def pvalue_contour(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    Z: npt.ArrayLike,
    *,
    ax: Axes | None = None,
    n_levels: int = 100,
) -> tuple[Figure | SubFigure | None, Axes, QuadContourSet]:
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2")
    two_sig = math.erfc(2 / math.sqrt(2))
    three_sig = math.erfc(3 / math.sqrt(2))
    norm = Normalize(vmin=0.0, vmax=1.0, clip=False)
    levels = np.unique(
        np.concatenate((np.linspace(0.0, 1.0, n_levels), [three_sig, two_sig]))
    )
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    image = ax.contourf(X, Y, Z, levels=levels, cmap=pvalue_cmap(), norm=norm)
    ax.contour(
        X, Y, Z, levels=[three_sig], colors="black", linestyles="--", linewidths=1.0
    )
    ax.contour(X, Y, Z, levels=[two_sig], colors="black", linewidths=1.0)
    return fig, ax, image
