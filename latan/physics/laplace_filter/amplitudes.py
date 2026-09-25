from dataclasses import dataclass
from typing import overload

import numpy as np
import numpy.typing as npt
from scipy import stats

from latan.display._common import (
    asymmetric_error_text,
    bootstrap_error_text,
    bootstrap_normality,
    non_gaussian_text,
    normality_significance,
)
from latan.display.laplace_filter import render_laplace_filter_amplitudes_html
from latan.physics.laplace_filter.filter import (
    lfilter,
    lfilter_correlated_data,
)
from latan.statistics.bootstrap import BootstrapArray
from latan.statistics.correlated_data import CorrelatedBootstrapData, CorrelatedData
from latan.statistics.correlation import cdr, cov_inverse_multiply, cov_quadratic_form
from latan.statistics.fit import fit
from latan.statistics.model import Model
from latan.statistics.xy_data import XYBootstrapData


@dataclass
class LaplaceFilterAmplitudes[T: npt.NDArray]:
    amplitudes: T
    ranges: tuple[tuple[int, int], ...]
    chi2: float
    p_value: float
    dof: int
    cdr: float
    linear_corrections: bool = False

    def _label(self, index: tuple[int, ...]) -> str:
        if self.linear_corrections:
            name = "A" if index[-1] == 0 else "B"
            index = index[:-1]
            if len(index) == 1 and self.amplitudes.shape[-2] == 1:
                index = ()
        else:
            name = "A"
        return f"{name}_{'_'.join(str(i) for i in index)}" if index else name

    def __str__(self) -> str:
        msg = "Laplace-filter amplitudes\n"
        if isinstance(self.amplitudes, BootstrapArray):
            amplitudes = self.amplitudes.central
            errors = self.amplitudes.error()
            lower, upper, non_gaussian, normality_p = bootstrap_normality(
                self.amplitudes
            )
            for index in np.ndindex(amplitudes.shape):
                label = self._label(index)
                msg += (
                    f"{label} = {bootstrap_error_text(amplitudes[index], errors[index])}"
                    f"{non_gaussian_text(normality_p[index], errors=(asymmetric_error_text(amplitudes[index], lower[index], upper[index]),) if non_gaussian[index] else ())}\n"
                )
        else:
            for index in np.ndindex(self.amplitudes.shape):
                label = self._label(index)
                msg += f"{label} = {self.amplitudes[index]:.4g}\n"
        ranges = ", ".join(f"[{start}, {stop})" for start, stop in self.ranges)
        msg += f"time range = {ranges}\n"
        msg += f"chi^2/dof = {self.chi2:.4g}/{self.dof} = {self.chi2 / self.dof:.2g}\n"
        msg += f"p = {self.p_value:.2g} ({normality_significance(self.p_value):.2g}σ)\n"
        msg += f"CDR at minimum = {self.cdr:.2g} dB"
        return msg

    def _repr_html_(self) -> str:
        return render_laplace_filter_amplitudes_html(self)


def _design_matrix(
    energies: npt.NDArray,
    ranges: list[tuple[int, int]],
    lengths: list[int],
    time_period: int | None,
    amplitude_lambda: float | None,
    linear_corrections: bool,
) -> npt.NDArray:
    """Build the spectral design matrix for one or more quantities.

    For state `i`, the amplitude column is `L_ti = exp(-E_i*t)`. If
    `time_period = T`, it is instead `L_ti = exp(-E_i*t) + exp(-E_i*(T-t))`.
    With `linear_corrections`, each state has consecutive columns
    `L_t,2i` is given the `L_ti` previously defined, and the first-order corrections
    `L_t,2i+1 = -t*exp(-E_i*t) - (T-t)*exp(-E_i*(T-t))`,
    omitting the second term for non-periodic data. Leading axes of `energies`
    produce a batch of design matrices.

    Quantities occupy separate column blocks over their half-open fit ranges.
    If `amplitude_lambda` is set, each full-time basis column is Laplace-filtered
    before selecting its fit range, leaving the coefficients in the unfiltered
    correlator convention.
    """
    n_states = energies.shape[-1]
    n_columns = 2 if linear_corrections else 1
    n_points = sum(stop - start for start, stop in ranges)
    lmat = np.zeros(
        (*energies.shape[:-1], n_points, len(ranges) * n_states * n_columns)
    )
    offset = 0
    for quantity, ((start, stop), length) in enumerate(zip(ranges, lengths)):
        time = np.arange(length)
        forward = np.exp(-energies[..., :, None] * time)
        block = forward.copy()
        if linear_corrections:
            derivative = -time * forward
        if time_period is not None:
            backward_time = time_period - time
            backward = np.exp(-energies[..., :, None] * backward_time)
            block += backward
            if linear_corrections:
                derivative -= backward_time * backward
        if linear_corrections:
            block = np.stack((block, derivative), axis=-2)
            block = block.reshape(*energies.shape[:-1], n_states * 2, length)
        if amplitude_lambda is not None:
            block = lfilter(block, amplitude_lambda, dim=-1)
        size = stop - start
        lmat[
            ...,
            offset : offset + size,
            quantity * n_states * n_columns : (quantity + 1) * n_states * n_columns,
        ] = np.swapaxes(block[..., start:stop], -1, -2)
        offset += size
    return lmat


@dataclass
class _LatentAmplitudeModel:
    ranges: list[tuple[int, int]]
    lengths: list[int]
    time_period: int | None
    amplitude_lambda: float | None
    linear_corrections: bool

    def __call__(self, x: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
        # Every energy coordinate is shared by all observations in x.
        lmat = _design_matrix(
            x[0],
            self.ranges,
            self.lengths,
            self.time_period,
            self.amplitude_lambda,
            self.linear_corrections,
        )
        return lmat @ p


@overload
def lfilter_amplitudes(
    data: CorrelatedData,
    ranges: list[tuple[int, int]] | tuple[int, int],
    energies: npt.NDArray,
    *,
    amplitude_lambda: float | None = None,
    time_period: int | None = None,
    external_energies: bool = False,
    linear_corrections: bool = False,
    workers: int = 1,
) -> LaplaceFilterAmplitudes[npt.NDArray]: ...


@overload
def lfilter_amplitudes(
    data: list[BootstrapArray] | BootstrapArray,
    ranges: list[tuple[int, int]] | tuple[int, int],
    energies: npt.NDArray | BootstrapArray,
    *,
    amplitude_lambda: float | None = None,
    time_period: int | None = None,
    external_energies: bool = False,
    linear_corrections: bool = False,
    workers: int = 1,
) -> LaplaceFilterAmplitudes[BootstrapArray]: ...


def lfilter_amplitudes(
    data: CorrelatedData | list[BootstrapArray] | BootstrapArray,
    ranges: list[tuple[int, int]] | tuple[int, int],
    energies: npt.NDArray | BootstrapArray,
    *,
    amplitude_lambda: float | None = None,
    time_period: int | None = None,
    external_energies: bool = False,
    linear_corrections: bool = False,
    workers: int = 1,
) -> LaplaceFilterAmplitudes[npt.NDArray] | LaplaceFilterAmplitudes[BootstrapArray]:
    """Determine amplitudes from spectrum through a linear regression.

    Each quantity has one amplitude per supplied energy. By default, bootstrap
    data are fitted in one batched linear solve with the central covariance fixed.
    An optional Laplace filter regulator can be provided to improve the conditioning
    of the correlation matrix. If provided, the spectral basis is filtered too, so
    the fitted amplitudes still refer to the unfiltered data.

    When `time_period` is provided, the regression basis includes the backward
    propagator appropriate for periodic time boundaries. `linear_corrections`
    adds a derivative-in-energy column for each state, returning raw `(A, B)`
    coefficients on a trailing axis. With `external_energies`, bootstrap
    energies enter the fit as latent variables with their cross-covariance.
    """
    if isinstance(ranges, tuple):
        ranges = [ranges]
    if isinstance(data, CorrelatedData):
        if isinstance(energies, BootstrapArray):
            raise TypeError("CorrelatedData requires non-bootstrap energies")
        if external_energies:
            raise TypeError("external_energies requires bootstrap data and energies")
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
        cdata = CorrelatedBootstrapData(data_f)
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
        if energies.ndim != 1:
            raise ValueError("energies must have shape (n_states,)")
    elif isinstance(energies, BootstrapArray):
        if energies.ndim != 2 or energies.shape[0] != n_bootstrap:
            raise ValueError("bootstrap energies must align with bootstrap data")
    elif external_energies:
        raise TypeError("external_energies requires bootstrap energies")
    elif energies.ndim != 1:
        raise ValueError("fixed energies must have shape (n_states,)")

    energies_array = np.asarray(energies)
    n_states = energies_array.shape[-1]
    n_quantities = cdata.n_quantities
    n_columns = 2 if linear_corrections else 1
    n_amplitudes = n_quantities * n_states * n_columns
    n_points = mean.size
    dof = n_points - n_amplitudes
    if dof <= 0:
        raise ValueError(f"non-positive degrees of freedom ({dof})")

    lengths = [cdata.mean(quantity).size for quantity in range(n_quantities)]
    design_energies = energies_array[0] if external_energies else energies_array

    # make exponential design matrix L
    lmat = _design_matrix(
        design_energies,
        ranges,
        lengths,
        time_period,
        amplitude_lambda,
        linear_corrections,
    )

    if external_energies:
        # linear regression to the model L*B,
        # with energies refitted as latent parameters
        assert isinstance(energies, BootstrapArray) and y is not None
        central_lmat = lmat
        vinv_lmat = cov_inverse_multiply(central_lmat.T, cov)
        normal = central_lmat.T @ vinv_lmat.T
        rhs = central_lmat.T @ cov_inverse_multiply(y[0], cov)
        initial = np.linalg.solve(normal, rhs)
        energy_data = [
            BootstrapArray(energies_array[:, i : i + 1]) for i in range(n_states)
        ]
        joint = CorrelatedBootstrapData([*energy_data, BootstrapArray(y)])
        model = Model(
            _LatentAmplitudeModel(
                ranges, lengths, time_period, amplitude_lambda, linear_corrections
            ),
            n_var=n_states,
            n_par=n_amplitudes,
        )
        xy_data = XYBootstrapData(
            joint,
            x=list(range(n_states)),
            y_indices=[n_states],
            x_map=[0] * n_states,
        )
        result = fit(xy_data, model, p0=initial, workers=workers)
        amplitudes = np.asarray(result.model_parameters).reshape(
            -1, n_quantities, n_states, *([2] if linear_corrections else [])
        )
        if n_quantities == 1:
            amplitudes = amplitudes[:, 0]
        amplitudes = LaplaceFilterAmplitudes(
            amplitudes=BootstrapArray(amplitudes),
            ranges=tuple(ranges),
            chi2=result.chi2,
            p_value=result.p_value,
            dof=result.dof,
            cdr=result.cdr,
            linear_corrections=linear_corrections,
        )

    else:
        # analytic linear regression for the unfiltered amplitudes
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

        shape = (
            *amplitudes.shape[:-1],
            n_quantities,
            n_states,
            *([2] if linear_corrections else []),
        )
        amplitudes = amplitudes.reshape(shape)
        if n_quantities == 1:
            amplitudes = (
                amplitudes[..., 0, :, :]
                if linear_corrections
                else amplitudes[..., 0, :]
            )
        if n_bootstrap is not None:
            amplitudes = BootstrapArray(amplitudes)
        amplitudes = LaplaceFilterAmplitudes(
            amplitudes=amplitudes,
            ranges=tuple(ranges),
            chi2=chi2,
            p_value=stats.chi2.sf(chi2, dof).item(),
            dof=dof,
            cdr=cdr(cov),
            linear_corrections=linear_corrections,
        )

    return amplitudes
