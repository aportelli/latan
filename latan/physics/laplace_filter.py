import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from iminuit import Minuit
from numba import njit
from scipy import stats

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


def lfilter_tilde(e):
    return np.sqrt(2.0 * (np.cosh(e) - 1.0))


def lfilter_tilde_inv(lamb):
    return 2.0 * np.arcsinh(lamb / 2.0)


class LaplaceFilteredT2:
    _range: Tuple[int, int]
    _slice: slice
    _mean: npt.NDArray
    _cov: npt.NDArray
    _mean_buf: npt.NDArray
    _cov_buf: npt.NDArray
    _n_state: int

    def __init__(
        self, mean: npt.NDArray, cov: npt.NDArray, range: Tuple[int, int], n_state: int
    ) -> None:
        self._range = range
        self._slice = slice(*range)
        self._mean = mean
        self._cov = cov
        self._n_state = n_state
        self._mean_buf = np.empty_like(self._mean)
        self._cov_buf = np.empty_like(self._cov)

    @property
    def range(self) -> Tuple[int, int]:
        return self._range

    def __call__(self, lamb: npt.NDArray) -> float:
        lfilter(self._mean, lamb, out=self._mean_buf)
        lfilter(self._cov, lamb, dim=(0, 1), out=self._cov_buf)
        mean_f = self._mean_buf[self._slice]
        cov_f = self._cov_buf[self._slice, self._slice]
        corr, err = var_to_corr(cov_f)
        mean_f /= err
        t2 = (mean_f @ np.linalg.solve(corr, mean_f)).item()
        return t2


@dataclass
class LaplaceFilterSpectrumResult:
    energies: List[npt.NDArray] = field(default_factory=list)
    lambdas: List[npt.NDArray] = field(default_factory=list)
    sig_states: int = -1
    t2: npt.NDArray = field(default_factory=lambda: np.empty(0))
    dt2: npt.NDArray = field(default_factory=lambda: np.empty(0))
    p_val: npt.NDArray = field(default_factory=lambda: np.empty(0))
    pbar_val: npt.NDArray = field(default_factory=lambda: np.empty(0))


def lfilter_spectrum(
    mean: npt.NDArray,
    cov: npt.NDArray,
    n_state: int,
    ti: int,
    tf: int,
    alpha: float = 0.05,
    verbose: bool = False,
    init_lambda: float = 50.0,
) -> LaplaceFilterSpectrumResult:
    msg = ""
    res = LaplaceFilterSpectrumResult()
    t_guess = mean.shape[0] // 4
    m_guess = math.acosh(
        (mean[t_guess - 1] + mean[t_guess + 1]) / (2.0 * mean[t_guess])
    )
    if verbose:
        print(
            f"==== Laplace filter spectrum -- ti = {ti}, tf = {tf} ({tf - ti} points)"
        )
    if verbose:
        print(f"ground state guess: {m_guess:.4f}")
    init = [m_guess]
    p = []
    pb = []
    t2 = []
    dt2 = []
    for r in range(1, n_state + 1):
        t2_func = LaplaceFilteredT2(mean, cov, (ti, tf), r)

        def cost(*lambdas):
            return t2_func(np.asarray(lambdas, dtype=float))

        names = [f"lambda_{i}" for i in range(r)]
        m = Minuit(cost, *init, name=names)
        m.limits = (0, None)
        m.simplex()  # simplex preconditioning
        minimum = m.migrad()
        assert minimum.fmin is not None
        if not minimum.fmin.is_valid:
            print("warning: invalid minimum")
        init = sorted(list(minimum.values))
        lambs = np.array(init)
        res.lambdas.append(lambs)
        res.energies.append(lfilter_tilde_inv(lambs))
        t2.append(t2_func(lambs))
        init.append(init_lambda)
        p.append(1 - stats.chi2.cdf(t2[-1], tf - ti - r))
        if verbose:
            msg = f"{r} states: T^2_{r} = {t2[-1]:.2e} (p_{r} = {p[-1]:.2e})"
        if r > 1:
            dt2.append(math.fabs(t2[-2] - t2[-1]))
            pb.append(1 - stats.chi2.cdf(dt2[-1], 1))
            if verbose:
                msg += f", ΔT^2_{r - 1} = {dt2[-1]:.2e} (pb_{r - 1} = {pb[-1]:.2e})"
        if verbose:
            msg += f", Lambda = {lambs}"
            print(msg)
    order = sorted(range(len(pb)), key=pb.__getitem__)
    max_reject = 0
    reject = ""
    for j in range(0, n_state - 1):
        if pb[order[j]] < alpha / (n_state - 1 - j):
            max_reject = max(max_reject, order[j] + 1)
            reject += f" H{order[j] + 1}"
    if verbose:
        msg = f"Holm-Bonferroni rejections:{reject}"
    if p[max_reject] < alpha:
        if verbose:
            msg += f" -- H{max_reject + 1} rejected, inconclusive"
        res.sig_states = -1
    else:
        if verbose:
            msg += f" -- H{max_reject + 1} not rejected, {max_reject + 1} sigificant states"
        res.sig_states = max_reject + 1
    if verbose:
        print(msg)
    res.p_val = np.array(p)
    res.pbar_val = np.array(pb)
    res.t2 = np.array(t2)
    res.dt2 = np.array(dt2)
    return res
