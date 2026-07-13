from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from latan.statistics.correlation import var_to_corr


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


class LaplaceFilteredT2:
    _range: Tuple[int, int]
    _slice: slice
    _data: npt.NDArray
    _var: npt.NDArray
    _data_buf: npt.NDArray
    _var_buf: npt.NDArray
    _n_state: int

    def __init__(self, data: npt.NDArray, range: Tuple[int, int], n_state: int) -> None:
        self._range = range
        self._slice = slice(*range)
        self._data = data.mean(axis=0)
        self._var = np.cov(data, rowvar=False) / data.shape[0]
        self._n_state = n_state
        self._data_buf = np.empty_like(self._data)
        self._var_buf = np.empty_like(self._var)

    @property
    def range(self) -> Tuple[int, int]:
        return self._range

    def __call__(self, lamb: npt.NDArray) -> float:
        lfilter(self._data, lamb, out=self._data_buf)
        lfilter(self._var, lamb, dim=(0, 1), out=self._var_buf)
        data_f = self._data_buf[self._slice]
        var_f = self._var_buf[self._slice, self._slice]
        corr, err = var_to_corr(var_f)
        data_f /= err
        t2 = (data_f @ np.linalg.solve(corr, data_f)).item()
        return t2
