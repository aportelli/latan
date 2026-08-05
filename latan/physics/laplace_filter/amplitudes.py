from dataclasses import dataclass
from typing import overload

import numpy as np
import numpy.typing as npt
from scipy import stats

from latan.display.laplace_filter import render_laplace_filter_amplitudes_html
from latan.physics.laplace_filter.filter import (
    lfilter,
    lfilter_correlated_data,
    lfilter_factor,
    lfilter_tilde_inv,
)
from latan.statistics.bootstrap import BootstrapArray
from latan.statistics.correlated_data import CorrelatedData, make_correlated_data
from latan.statistics.correlation import cdr, cov_inverse_multiply, cov_quadratic_form


@dataclass
class LaplaceFilterAmplitudes[T: npt.NDArray]:
    amplitudes: T
    ranges: tuple[tuple[int, int], ...]
    chi2: float
    p_value: float
    dof: int
    cdr: float

    def __repr__(self) -> str:
        msg = ""
        if isinstance(self.amplitudes, BootstrapArray):
            amplitudes = self.amplitudes.central
            errors = self.amplitudes.error()
            for index in np.ndindex(amplitudes.shape):
                label = "_".join(str(i) for i in index)
                msg += f"A_{label} = {amplitudes[index]:.4g} ± {errors[index]:.4g}\n"
        else:
            for index in np.ndindex(self.amplitudes.shape):
                label = "_".join(str(i) for i in index)
                msg += f"A_{label} = {self.amplitudes[index]:.4g}\n"
        ranges = ", ".join(f"[{start}, {stop})" for start, stop in self.ranges)
        msg += f"time range = {ranges}\n"
        msg += f"chi^2/dof = {self.chi2:.4g}/{self.dof} = {self.chi2 / self.dof:.2g}\n"
        msg += f"  p-value = {self.p_value:.2g}\n"
        msg += f"CDR at minimum {self.cdr:.2g} dB"
        return msg

    def _repr_html_(self) -> str:
        return render_laplace_filter_amplitudes_html(self)


@overload
def lfilter_amplitudes(
    data: CorrelatedData,
    ranges: list[tuple[int, int]] | tuple[int, int],
    lambdas: npt.NDArray,
    *,
    amplitude_lambda: float | None = None,
    time_period: int | None = None,
) -> LaplaceFilterAmplitudes[npt.NDArray]: ...


@overload
def lfilter_amplitudes(
    data: list[BootstrapArray] | BootstrapArray,
    ranges: list[tuple[int, int]] | tuple[int, int],
    lambdas: npt.NDArray | BootstrapArray,
    *,
    amplitude_lambda: float | None = None,
    time_period: int | None = None,
) -> LaplaceFilterAmplitudes[BootstrapArray]: ...


def lfilter_amplitudes(
    data: CorrelatedData | list[BootstrapArray] | BootstrapArray,
    ranges: list[tuple[int, int]] | tuple[int, int],
    lambdas: npt.NDArray | BootstrapArray,
    *,
    amplitude_lambda: float | None = None,
    time_period: int | None = None,
) -> LaplaceFilterAmplitudes[npt.NDArray] | LaplaceFilterAmplitudes[BootstrapArray]:
    """Determine amplitudes from spectrum through a linear regression.

    Each quantity has one amplitude per supplied regulator. Bootstrap data
    are fitted in one batched linear solve with the central covariance fixed.
    An optional Laplace filter regulator can be provided to improve the conditioning
    of the correlation matrix. If provided, the resulting amplitudes are corrected to
    fit the unfiltered data.

    When `time_period` is provided, the regression basis includes the backward
    propagator appropriate for periodic time boundaries.
    """
    if isinstance(ranges, tuple):
        ranges = [ranges]
    if isinstance(data, CorrelatedData):
        if isinstance(lambdas, BootstrapArray):
            raise TypeError("CorrelatedData requires non-bootstrap lambdas")
        if amplitude_lambda is not None:
            cdata = lfilter_correlated_data(data, amplitude_lambda)
        else:
            cdata = data
        y = None
        n_bootstrap = None
    else:
        if isinstance(data, BootstrapArray):
            data = [data]
        if not isinstance(data, list) or not data:
            raise TypeError("data must be CorrelatedData or a list of BootstrapArray")
        if not all(isinstance(item, BootstrapArray) for item in data):
            raise TypeError("data must be CorrelatedData or a list of BootstrapArray")
        if len(ranges) != len(data):
            raise ValueError("number of ranges and bootstrap quantities mismatch")
        if amplitude_lambda is not None:
            data_f = [lfilter(b, amplitude_lambda) for b in data]
        else:
            data_f = data
        cdata = make_correlated_data(data_f)
        n_bootstrap = data_f[0].shape[0]
        if any(item.shape[0] != n_bootstrap for item in data_f):
            raise ValueError("bootstrap quantities have different sample counts")
        y = np.concatenate(
            [
                np.asarray(item)[..., start:stop]
                for item, (start, stop) in zip(data_f, ranges)
            ],
            axis=-1,
        )

    mean, cov = cdata.total_mean_cov(ranges)
    if n_bootstrap is None:
        if lambdas.ndim != 1:
            raise ValueError("lambdas must have shape (n_states,)")
    elif isinstance(lambdas, BootstrapArray):
        if lambdas.ndim != 2 or lambdas.shape[0] != n_bootstrap:
            raise ValueError("bootstrap lambdas must align with bootstrap data")
    elif lambdas.ndim != 1:
        raise ValueError("fixed lambdas must have shape (n_states,)")

    energies = np.asarray(lfilter_tilde_inv(lambdas))
    n_states = energies.shape[-1]
    n_quantities = cdata.n_quantities
    n_amplitudes = n_quantities * n_states
    n_points = mean.size
    dof = n_points - n_amplitudes
    if dof <= 0:
        raise ValueError(f"non-positive degrees of freedom ({dof})")

    # build exponential design matrix
    #
    # L_ti = exp(-E_i*t) + exp(-E_i*(T-t)) for periodic time boundaries
    #
    # if several quantities are present, L is extended as a block-diagonal matrix for
    # every time range. If amplitude_lambda is provided, L gets filtered to generate
    # the amplitude correction factor.
    lmat = np.zeros((*energies.shape[:-1], n_points, n_amplitudes))
    offset = 0
    for quantity, (start, stop) in enumerate(ranges):
        time = np.arange(cdata.mean(quantity).size)
        block = np.exp(-energies[..., :, None] * time)
        if time_period is not None:
            block += np.exp(-energies[..., :, None] * (time_period - time))
        if amplitude_lambda is not None:
            block = lfilter(block, amplitude_lambda, dim=-1)
        block = block[..., start:stop]
        size = stop - start
        lmat[
            ...,
            offset : offset + size,
            quantity * n_states : (quantity + 1) * n_states,
        ] = np.swapaxes(block, -1, -2)
        offset += size

    # linear regression for the unfiltered amplitudes
    #
    # B = (L^T * V^-1 * L)^-1 * (L^T * V^-1 * y)
    #
    # where V is the covariance matrix and y the data
    if y is None:
        y = mean
    lmat_t = np.swapaxes(lmat, -1, -2)
    vinv_lmat = cov_inverse_multiply(lmat_t, cov)  # transposed for columns as batch
    lmat_vinv_lmat = lmat_t @ np.swapaxes(vinv_lmat, -1, -2)
    vinv_y = cov_inverse_multiply(y, cov)
    rhs = lmat_t @ vinv_y[..., None]
    amplitudes = np.linalg.solve(lmat_vinv_lmat, rhs).squeeze(-1)

    central_lmat = lmat[0] if lmat.ndim == 3 else lmat
    central_y = y[0] if y.ndim == 2 else y
    central_amplitudes = amplitudes[0] if amplitudes.ndim == 2 else amplitudes
    residual = central_y - central_lmat @ central_amplitudes
    chi2 = float(cov_quadratic_form(residual, cov))

    shape = (*amplitudes.shape[:-1], n_quantities, n_states)
    amplitudes = amplitudes.reshape(shape)
    if n_quantities == 1:
        amplitudes = amplitudes[..., 0, :]
    if n_bootstrap is not None:
        amplitudes = BootstrapArray(amplitudes)
    return LaplaceFilterAmplitudes(
        amplitudes=amplitudes,
        ranges=tuple(ranges),
        chi2=chi2,
        p_value=stats.chi2.sf(chi2, dof).item(),
        dof=dof,
        cdr=cdr(cov),
    )
