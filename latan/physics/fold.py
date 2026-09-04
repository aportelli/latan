from typing import overload

import numpy as np
import numpy.typing as npt

from latan.statistics.bootstrap import BootstrapArray


@overload
def fold(x: BootstrapArray, axis: int = -1) -> BootstrapArray: ...


@overload
def fold(x: npt.NDArray, axis: int = -1) -> npt.NDArray: ...


def fold(
    x: npt.NDArray | BootstrapArray, axis: int = -1
) -> npt.NDArray | BootstrapArray:
    nt = x.shape[axis]
    if nt % 2 != 0:
        raise ValueError(f"array does not have an even dimension on axis {axis}")
    if isinstance(x, BootstrapArray) and axis % x.ndim == 0:
        raise ValueError("cannot fold the BootstrapArray sample axis")
    indices = np.arange(nt // 2 + 1)
    folded = (np.take(x, indices, axis=axis) + np.take(x, -indices, axis=axis)) / 2
    return BootstrapArray(folded) if isinstance(x, BootstrapArray) else folded
