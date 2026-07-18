from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy import linalg


def cdr(mat: npt.NDArray) -> float:
    s = np.linalg.svd(mat, compute_uv=False)
    return 10.0 * np.log10(s.max() / s.min())


def cov_to_corr(cov: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    err = np.sqrt(cov.diagonal())
    inverr = 1.0 / err
    corr = cov * np.outer(inverr, inverr)
    return corr, err


def corr_to_cov(corr: npt.NDArray, err: npt.NDArray) -> npt.NDArray:
    return corr * np.outer(err, err)


def cov_factor(
    cov: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Return a lower correlation Cholesky factor and standard deviations."""
    corr, err = cov_to_corr(cov)
    factor = linalg.cholesky(corr, lower=True, check_finite=False)
    return factor, err


# Compiled direct forward-substitution algorithm
# (was generally faster than SciPy on O(100^2) matrices)
@njit(cache=True)
def _corr_quadratic_form(
    residual: npt.NDArray,
    err: npt.NDArray,
    factor: npt.NDArray,
    work: npt.NDArray,
) -> float:
    total = 0.0
    for i in range(residual.size):
        value = residual[i] / err[i]
        for j in range(i):
            value -= factor[i, j] * work[j]
        value /= factor[i, i]
        work[i] = value
        total += value * value
    return total


@njit(cache=True)
def _corr_independent_residuals(
    residual: npt.NDArray,
    err: npt.NDArray,
    factor: npt.NDArray,
    out: npt.NDArray,
) -> None:
    for i in range(residual.size):
        value = residual[i] / err[i]
        for j in range(i):
            value -= factor[i, j] * out[j]
        out[i] = value / factor[i, i]


def cov_independent_residuals(
    residual: npt.NDArray,
    cov: npt.NDArray,
    factor: Tuple[npt.NDArray, npt.NDArray] | None = None,
    out: npt.NDArray | None = None,
) -> npt.NDArray:
    """Return statistically independent residuals.

    Args:
        residual: Residual vector.
        cov: Covariance matrix corresponding to `residual`.
        factor: Optional lower Cholesky factor and standard deviations of the
            associated correlation matrix, as returned by `cov_factor`.
            It is computed from `cov` when omitted.
        out: Optional one-dimensional output array with the same shape as
            `residual`. Supplying a reusable array avoids an allocation. Its
            contents are overwritten.

    Returns:
        Independent residuals whose squared Euclidean norm is the
        covariance-normalized quadratic form.
    """
    if factor is None:
        factor = cov_factor(cov)
    lower, err = factor
    if out is None:
        out = np.empty_like(residual)
    _corr_independent_residuals(residual, err, lower, out)
    return out


def cov_quadratic_form(
    residual: npt.NDArray,
    cov: npt.NDArray,
    factor: Tuple[npt.NDArray, npt.NDArray] | None = None,
    work: npt.NDArray | None = None,
) -> float:
    """Evaluate a covariance-normalized quadratic form from a lower factor.

    Args:
        residual: Residual vector.
        cov: Covariance matrix corresponding to `residual`.
        factor: Optional lower Cholesky factor and standard deviations of the
            associated correlation matrix, as returned by `cov_factor`.
            It is computed from `cov` when omitted.
        work: Optional one-dimensional workspace with the same shape as
            `residual`. Supplying a reusable array avoids an allocation for
            each evaluation. Its contents are overwritten.
    """
    if factor is None:
        factor = cov_factor(cov)
    lower, err = factor
    if work is None:
        work = np.empty_like(residual)
    return _corr_quadratic_form(residual, err, lower, work)
