from collections.abc import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt
from numba import njit

from latan.statistics.bootstrap import BootstrapArray
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


@overload
def lfilter(
    data: BootstrapArray,
    lamb: float | Sequence[float] | npt.NDArray,
    dim: int | Sequence[int] = -1,
    out: BootstrapArray | None = None,
) -> BootstrapArray: ...


@overload
def lfilter(
    data: npt.NDArray,
    lamb: float | Sequence[float] | npt.NDArray,
    dim: int | Sequence[int] = -1,
    out: npt.NDArray | None = None,
) -> npt.NDArray: ...


def lfilter(
    data: npt.NDArray,
    lamb: float | Sequence[float] | npt.NDArray,
    dim: int | Sequence[int] = -1,
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

    The default axis is the last axis. A `BootstrapArray` input remains a
    `BootstrapArray`; filtering its sample axis 0 is not allowed. If `out`
    is supplied for bootstrap data, it must also be a `BootstrapArray`.

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
    lambs = np.atleast_1d(np.asarray(lamb, dtype=float))
    dims = np.atleast_1d(np.asarray(dim, dtype=int))
    n_ops = lambs.size * dims.size
    is_bootstrap = isinstance(data, BootstrapArray)
    if is_bootstrap and any(int(axis) % data.ndim == 0 for axis in dims):
        raise ValueError("cannot filter the BootstrapArray sample axis")
    if is_bootstrap and out is not None and not isinstance(out, BootstrapArray):
        raise TypeError("BootstrapArray input requires a BootstrapArray out")

    data = np.ascontiguousarray(data)
    if out is None:
        out = np.empty_like(data, dtype=data.dtype)
    elif out.shape != data.shape or not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous and have data's shape")
    if np.shares_memory(data, out):
        raise ValueError("out must not overlap data")
    if n_ops == 0:
        out[...] = data
    elif n_ops == 1:
        axis = int(dims[0]) % data.ndim
        _lfilter_kernel(data, float(lambs[0]) ** 2, axis, out)
    else:
        buf = np.empty_like(data, dtype=data.dtype)
        src = data
        dest = out if n_ops % 2 else buf
        for la in lambs:
            for axis in dims:
                axis = int(axis) % data.ndim
                _lfilter_kernel(src, float(la) ** 2, axis, dest)
                src = dest
                dest = buf if dest is out else out
    if is_bootstrap and not isinstance(out, BootstrapArray):
        return BootstrapArray(out)
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


@overload
def lfilter_tilde(e: BootstrapArray) -> BootstrapArray: ...


@overload
def lfilter_tilde(e: npt.NDArray) -> npt.NDArray: ...


@overload
def lfilter_tilde(e: npt.ArrayLike) -> np.floating | npt.NDArray: ...


def lfilter_tilde(e: npt.ArrayLike) -> np.floating | npt.NDArray:
    return np.sqrt(2.0 * (np.cosh(e) - 1.0))


@overload
def lfilter_tilde_inv(lamb: BootstrapArray) -> BootstrapArray: ...


@overload
def lfilter_tilde_inv(lamb: npt.NDArray) -> npt.NDArray: ...


@overload
def lfilter_tilde_inv(lamb: npt.ArrayLike) -> np.floating | npt.NDArray: ...


def lfilter_tilde_inv(lamb: npt.ArrayLike) -> np.floating | npt.NDArray:
    return 2.0 * np.arcsinh(np.multiply(0.5, lamb))


@overload
def lfilter_factor(
    lamb: float | Sequence[float] | npt.NDArray,
    e: BootstrapArray,
) -> BootstrapArray: ...


@overload
def lfilter_factor(
    lamb: float | Sequence[float] | npt.NDArray,
    e: npt.NDArray,
) -> npt.NDArray: ...


@overload
def lfilter_factor(
    lamb: float | Sequence[float] | npt.NDArray,
    e: float | Sequence[float] | npt.NDArray,
) -> np.floating | npt.NDArray: ...


def lfilter_factor(
    lamb: float | Sequence[float] | npt.NDArray,
    e: float | Sequence[float] | npt.NDArray,
) -> np.floating | npt.NDArray:
    energies = np.asarray(e)
    lambs = np.atleast_1d(np.asarray(lamb, dtype=float))
    lambs = lambs.reshape((-1,) + (1,) * energies.ndim)
    factor = np.prod(lambs**2 - lfilter_tilde(energies) ** 2, axis=0)
    if isinstance(e, BootstrapArray):
        return BootstrapArray(factor)
    return factor
