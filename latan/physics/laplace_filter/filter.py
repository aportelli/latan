from typing import Sequence

import numpy as np
import numpy.typing as npt
from numba import njit

from latan.statistics.correlated_data import CorrelatedData


@njit(cache=True)
def _lfilter_kernel(
    x: npt.NDArray,
    a: float,
    dim: int,
    out: npt.NDArray,
) -> None:
    stride = 1
    for axis in range(dim + 1, x.ndim):
        stride *= x.shape[axis]
    n = x.shape[dim]
    if n == 0:
        return
    outer_count = x.size // (n * stride)
    source = x.reshape(x.size)
    destination = out.reshape(out.size)
    coefficient = 2.0 + a
    for group in range(outer_count * n):
        outer = group // n
        index = group - outer * n
        current = (outer * n + index) * stride
        prev = (outer * n + (index - 1 if index else n - 1)) * stride
        next = (outer * n + (index + 1 if index + 1 < n else 0)) * stride
        for offset in range(stride):
            destination[current + offset] = (
                coefficient * source[current + offset]
                - source[prev + offset]
                - source[next + offset]
            )


def lfilter(
    data: npt.NDArray,
    lamb: float | Sequence[float] | npt.NDArray,
    dim: int | Sequence[int] = 0,
    out: npt.NDArray | None = None,
) -> npt.NDArray:
    """Apply one or more periodic Laplace filters to an array.

    One filter with regulator ``lamb`` on axis ``dim`` maps each element
    ``x[t]`` to ``(2 + lamb**2) * x[t] - x[t - 1] - x[t + 1]``. The two
    neighbours wrap around at the endpoints of the selected axis.

    `lamb` may be one scalar or a sequence of regulators. `dim` may be one
    axis or a sequence of axes. With multiple regulators and axes, filters
    are applied in regulator order and, for each regulator, in axis order.
    For example, ``lfilter(data, [a, b], [0, 1])`` applies ``a`` on axis 0,
    then ``a`` on axis 1, then ``b`` on axis 0, and finally ``b`` on axis 1.
    An empty regulator sequence copies `data` unchanged.

    Args:
        data: Array to filter.
        lamb: One regulator or a sequence of regulators.
        dim: One axis or an ordered sequence of axes.
        out: Optional C-contiguous array with the same shape as `data`. It
            must not overlap `data`.

    Returns:
        The filtered array, or `out` when it is supplied.
    """
    if data.ndim == 0:
        raise ValueError("data must have at least one dimension")
    data = np.ascontiguousarray(data)
    if out is None:
        out = np.empty_like(data, dtype=data.dtype)
    elif out.shape != data.shape or not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous and have data's shape")
    if np.shares_memory(data, out):
        raise ValueError("out must not overlap data")
    if isinstance(lamb, np.ndarray) and lamb.ndim == 0:
        lambs = [lamb.item()]
    elif not isinstance(lamb, (Sequence, np.ndarray)):
        lambs = [lamb]
    else:
        lambs = lamb
    if not isinstance(dim, Sequence):
        dims = [dim]
    else:
        dims = dim
    n_ops = len(lambs) * len(dims)
    if n_ops == 0:
        out[...] = data
    elif n_ops == 1:
        dim = dims[0] % data.ndim
        _lfilter_kernel(data, lambs[0] ** 2, dim, out)
    else:
        buf = np.empty_like(data, dtype=data.dtype)
        src = data
        dest = out if n_ops % 2 else buf
        for la in lambs:
            for d in dims:
                d = d % data.ndim
                _lfilter_kernel(src, la**2, d, dest)
                src = dest
                dest = buf if dest is out else out
    return out


def lfilter_correlated_data(
    data: CorrelatedData,
    lamb: float | Sequence[float] | npt.NDArray,
    out: CorrelatedData | None = None,
) -> CorrelatedData:
    """Apply Laplace filters to correlated means and covariance blocks.

    Means are filtered along their sole axis. Covariance blocks are filtered
    along both axes. `out`, when supplied, must have the same number of
    quantities and compatible mean and covariance-block shapes as `data`.
    """
    if out is None:
        means = [np.empty_like(data.mean(i)) for i in range(data.n_quantities)]
        covs = [
            [
                np.zeros_like(data.cov(i, j))
                if i == j
                else np.empty_like(data.cov(i, j))
                for j in range(i, data.n_quantities)
            ]
            for i in range(data.n_quantities)
        ]
        out = CorrelatedData(means, covs)
    elif out.n_quantities != data.n_quantities:
        raise ValueError("out and data have a different number of quantities")

    for i in range(data.n_quantities):
        lfilter(data.mean(i), lamb, out=out.mean(i))
        for j in range(i, data.n_quantities):
            lfilter(data.cov(i, j), lamb, dim=(0, 1), out=out.cov(i, j))
    return out


def lfilter_tilde(e: npt.ArrayLike) -> np.floating | npt.NDArray:
    return np.sqrt(2.0 * (np.cosh(e) - 1.0))


def lfilter_tilde_inv(lamb: npt.ArrayLike) -> np.floating | npt.NDArray:
    return 2.0 * np.arcsinh(lamb / 2.0)


def lfilter_factor(
    lamb: float | Sequence[float] | npt.NDArray,
    e: float | Sequence[float] | npt.NDArray,
) -> np.floating | npt.NDArray:
    if isinstance(lamb, np.ndarray) and lamb.ndim == 0:
        lambs = [lamb.item()]
    elif not isinstance(lamb, (Sequence, np.ndarray)):
        lambs = [lamb]
    else:
        lambs = lamb
    energies = np.asarray(e)
    lambs = np.asarray(lambs).reshape((-1,) + (1,) * energies.ndim)
    return np.prod(lambs**2 - lfilter_tilde(energies) ** 2, axis=0)
