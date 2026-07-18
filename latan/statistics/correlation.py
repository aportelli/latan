from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy import linalg


def cdr(mat: npt.NDArray) -> float:
    s = np.linalg.svd(mat, compute_uv=False)
    return 10.0 * np.log10(s.max() / s.min())


def var_to_corr(var: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    err = np.sqrt(var.diagonal())
    inverr = 1.0 / err
    corr = var * np.outer(inverr, inverr)
    return corr, err


def corr_to_var(corr: npt.NDArray, err: npt.NDArray) -> npt.NDArray:
    return corr * np.outer(err, err)


def corr_factor(
    cov: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Return standard deviations and a lower Cholesky factor of a correlation matrix."""
    corr, err = var_to_corr(cov)
    factor = linalg.cholesky(corr, lower=True, check_finite=False)
    return err, factor


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


def corr_quadratic_form(
    residual: npt.NDArray,
    err: npt.NDArray,
    factor: npt.NDArray,
    work: npt.NDArray | None = None,
) -> float:
    """Evaluate a covariance-normalized quadratic form from a lower factor.

    Args:
        residual: Residual vector.
        err: Standard deviations corresponding to `residual`.
        factor: Lower Cholesky factor of the associated correlation matrix.
        work: Optional one-dimensional workspace with the same shape as
            `residual`. Supplying a reusable array avoids an allocation for
            each evaluation. Its contents are overwritten.
    """
    if work is None:
        work = np.empty_like(residual)
    return _corr_quadratic_form(residual, err, factor, work)
