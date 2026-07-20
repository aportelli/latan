from typing import Tuple

import numpy as np
import numpy.typing as npt
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


# scale vectors by error and invert order for SciPy triangular solve
def _rhs(
    vector: npt.NDArray,
    err: npt.NDArray,
    out: npt.NDArray | None,
) -> Tuple[npt.NDArray, npt.NDArray]:
    if vector.ndim == 0:
        raise ValueError("vector must have at least one dimension")
    n = vector.shape[-1]
    if err.shape != (n,):
        raise ValueError(f"factor errors have shape {err.shape}, expected ({n},)")
    if out is None:
        out = np.empty(vector.shape, dtype=np.result_type(vector.dtype, err.dtype))
    elif out.shape != vector.shape:
        raise ValueError(f"out has shape {out.shape}, expected {vector.shape}")
    elif not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    np.divide(vector, err, out=out)
    return out.reshape(-1, n).T, out


def _factor(
    cov: npt.NDArray,
    factor: Tuple[npt.NDArray, npt.NDArray] | None,
    n: int,
) -> Tuple[npt.NDArray, npt.NDArray]:
    if cov.shape != (n, n):
        raise ValueError(f"cov has shape {cov.shape}, expected ({n}, {n})")
    if factor is None:
        factor = cov_factor(cov)
    lower, err = factor
    if lower.shape != (n, n) or err.shape != (n,):
        raise ValueError("factor has incompatible shapes")
    return lower, err


def cov_independent_residuals(
    residual: npt.NDArray,
    cov: npt.NDArray,
    factor: Tuple[npt.NDArray, npt.NDArray] | None = None,
    out: npt.NDArray | None = None,
) -> npt.NDArray:
    """Return statistically independent residuals.

    Args:
        residual: Residual vectors with shape `(..., n)`.
        cov: Shared covariance matrix with shape `(n, n)`.
        factor: Optional lower Cholesky factor and standard deviations of the
            associated correlation matrix, as returned by `cov_factor`.
            It is computed from `cov` when omitted.
        out: Optional C-contiguous output array with the same shape as
            `residual`. Supplying a reusable array avoids an allocation. Its
            contents are overwritten.

    Returns:
        Independent residuals with shape `(..., n)`. Squaring and summing the
        final axis gives the covariance-normalized quadratic form for each
        batch entry.
    """
    if residual.ndim == 0:
        raise ValueError("residual must have at least one dimension")
    n = residual.shape[-1]
    lower, err = _factor(cov, factor, n)
    rhs, out = _rhs(residual, err, out)
    result = linalg.solve_triangular(
        lower,
        rhs,
        lower=True,
        check_finite=False,
        overwrite_b=True,
    )
    if result is not rhs:
        rhs[...] = result
    return out


def cov_quadratic_form(
    residual: npt.NDArray,
    cov: npt.NDArray,
    factor: Tuple[npt.NDArray, npt.NDArray] | None = None,
    work: npt.NDArray | None = None,
) -> npt.NDArray:
    """Evaluate a covariance-normalized quadratic form from a lower factor.

    Args:
        residual: Residual vectors with shape `(..., n)`.
        cov: Shared covariance matrix with shape `(n, n)`.
        factor: Optional lower Cholesky factor and standard deviations of the
            associated correlation matrix, as returned by `cov_factor`.
            It is computed from `cov` when omitted.
        work: Optional C-contiguous workspace with the same shape as
            `residual`. Supplying a reusable array avoids an allocation for
            each evaluation. Its contents are overwritten.

    Returns:
        Quadratic forms with shape `residual.shape[:-1]`. A one-dimensional
        residual returns a zero-dimensional array.
    """
    res = cov_independent_residuals(residual, cov, factor, work)
    return np.asarray(np.vecdot(res, res))


def cov_inverse_multiply(
    vector: npt.NDArray,
    cov: npt.NDArray,
    factor: Tuple[npt.NDArray, npt.NDArray] | None = None,
    out: npt.NDArray | None = None,
) -> npt.NDArray:
    """Multiply vectors by the inverse of a shared covariance matrix.

    This evaluates `cov^{-1} vector` through the cached correlation factor,
    without explicitly forming an inverse. Vectors have shape `(..., n)` and
    `cov` has shared shape `(n, n)`.

    Args:
        vector: Vectors with shape `(..., n)`.
        cov: Shared covariance matrix with shape `(n, n)`.
        factor: Optional lower correlation Cholesky factor and standard
            deviations, as returned by `cov_factor`.
        out: Optional C-contiguous output array with the same shape as
            `vector`.

    Returns:
        `cov^{-1} vector` with shape `(..., n)`.
    """
    if vector.ndim == 0:
        raise ValueError("vector must have at least one dimension")
    n = vector.shape[-1]
    lower, err = _factor(cov, factor, n)
    rhs, out = _rhs(vector, err, out)
    result = linalg.cho_solve(
        (lower, True),
        rhs,
        check_finite=False,
        overwrite_b=True,
    )
    if result is not rhs:
        rhs[...] = result
    rhs /= err[:, None]
    return out
