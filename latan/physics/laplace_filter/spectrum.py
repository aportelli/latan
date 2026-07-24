import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt
from iminuit import Minuit
from scipy import stats

from latan.display.laplace_filter import (
    render_laplace_filter_amplitudes_html,
    render_laplace_filter_energies_html,
)
from latan.physics.laplace_filter.filter import (
    lfilter,
    lfilter_correlated_data,
    lfilter_factor,
    lfilter_tilde_inv,
)
from latan.physics.laplace_filter.t2 import LaplaceFilteredT2
from latan.statistics.bootstrap import BootstrapArray
from latan.statistics.correlated_data import CorrelatedData, make_correlated_data
from latan.statistics.correlation import cdr, cov_inverse_multiply, cov_quadratic_form


@dataclass
class LaplaceFilterEnergies[T: npt.NDArray]:
    energies: T
    lambdas: T
    t2: float
    p_value: float
    dof: int
    cdr: float

    def __repr__(self) -> str:
        msg = ""
        if isinstance(self.energies, BootstrapArray):
            energies = self.energies.central
            energy_err = self.energies.std()
            assert isinstance(self.lambdas, BootstrapArray)
            lambdas = self.lambdas.central
            lambda_err = self.lambdas.std()
            for i, (energy, error, lamb, lamb_error) in enumerate(
                zip(energies, energy_err, lambdas, lambda_err)
            ):
                msg += (
                    f"E_{i} = {energy:.4g} ± {error:.4g}, "
                    f"lambda_{i} = {lamb:.4g} ± {lamb_error:.4g}\n"
                )
        else:
            for i, (energy, lamb) in enumerate(zip(self.energies, self.lambdas)):
                msg += f"E_{i} = {energy:.4g}, lambda_{i} = {lamb:.4g}\n"
        msg += f"T^2/dof = {self.t2:.4g}/{self.dof} = {self.t2 / self.dof:.2g}\n"
        msg += f"  p-value = {self.p_value:.2g}\n"
        msg += f"CDR at minimum {self.cdr:.2g} dB"
        return msg

    def _repr_html_(self) -> str:
        return render_laplace_filter_energies_html(self)

@dataclass
class LaplaceFilterAmplitudes[T: npt.NDArray]:
    amplitudes: T
    chi2: float
    p_value: float
    dof: int
    cdr: float

    def __repr__(self) -> str:
        msg = ""
        if isinstance(self.amplitudes, BootstrapArray):
            amplitudes = self.amplitudes.central
            errors = self.amplitudes.std()
            for index in np.ndindex(amplitudes.shape):
                label = "_".join(str(i) for i in index)
                msg += f"A_{label} = {amplitudes[index]:.4g} ± {errors[index]:.4g}\n"
        else:
            for index in np.ndindex(self.amplitudes.shape):
                label = "_".join(str(i) for i in index)
                msg += f"A_{label} = {self.amplitudes[index]:.4g}\n"
        msg += f"chi^2/dof = {self.chi2:.4g}/{self.dof} = {self.chi2 / self.dof:.2g}\n"
        msg += f"  p-value = {self.p_value:.2g}\n"
        msg += f"CDR at minimum {self.cdr:.2g} dB"
        return msg

    def _repr_html_(self) -> str:
        return render_laplace_filter_amplitudes_html(self)

@dataclass
class LaplaceFilterSpectrumTest:
    spectra: List[LaplaceFilterEnergies[npt.NDArray]] = field(default_factory=list)
    dt2: npt.NDArray = field(default_factory=lambda: np.empty(0))
    pbar_val: npt.NDArray = field(default_factory=lambda: np.empty(0))
    sig_states: int = -1


def _lfilter_spectrum(
    data: CorrelatedData,
    ranges: List[Tuple[int, int]],
    n_state: int,
    *,
    m_guess: Optional[float] = None,
    initial_lambdas: Optional[npt.NDArray] = None,
    init_lambda: float = 100.0,
    ncall: int = 5000,
) -> LaplaceFilterEnergies[npt.NDArray]:
    if n_state < 1:
        raise ValueError("n_state must be positive")
    if len(ranges) != data.n_quantities:
        raise ValueError(
            f"number of ranges and quantities mismatch "
            f"(got {len(ranges)}, expected {data.n_quantities})"
        )
    dof = sum(stop - start for start, stop in ranges) - n_state
    if dof <= 0:
        raise ValueError(f"non-positive degrees of freedom ({dof})")

    t2_func = LaplaceFilteredT2(data, ranges)

    # cost function using iMinuit call convention
    def cost(*lambdas):
        return t2_func(np.asarray(lambdas, dtype=float))

    names = [f"lambda_{i}" for i in range(n_state)]

    # if no initial lambda is provided, make an uncorrelated T2 minimisation
    # to determine guess (only on central value for bootstrap)
    if initial_lambdas is None:
        t2_uncorr_func = LaplaceFilteredT2(data.uncorrelated(), ranges)

        def cost_uncorr(*lambdas):
            return t2_uncorr_func(np.asarray(lambdas, dtype=float))

        # if no ground state guess is provided, use log effective mass at nt/4
        if m_guess is None:
            mean = data.mean()
            t_guess = mean.shape[0] // 4
            m_guess = math.log(mean[t_guess - 1] / mean[t_guess])
        start = np.array([m_guess, *([init_lambda] * (n_state - 1))])
        m_uncorr = Minuit(cost_uncorr, *start, name=names)
        m_uncorr.limits = (0, None)
        m_uncorr.simplex()
        m_uncorr.migrad()
        start = np.asarray(m_uncorr.values)
    else:
        start = np.asarray(initial_lambdas, dtype=float)
        if start.shape != (n_state,):
            raise ValueError(
                f"initial_lambdas has shape {start.shape}, expected ({n_state},)"
            )

    minimum = Minuit(cost, *start, name=names)
    minimum.limits = (0, None)
    result = minimum.migrad(ncall=ncall)
    assert result.fmin is not None
    if not result.fmin.is_valid:
        print("warning: invalid minimum")
    lambdas = np.asarray(sorted(minimum.values))
    t2 = t2_func(lambdas)
    filtered_data = lfilter_correlated_data(data, lambdas)
    _, cov = filtered_data.total_mean_cov(ranges)
    spectrum = LaplaceFilterEnergies(
        energies=lfilter_tilde_inv(lambdas),
        lambdas=lambdas,
        t2=t2,
        p_value=stats.chi2.sf(t2, dof).item(),
        dof=dof,
        cdr=cdr(cov),
    )
    return spectrum


def _spectrum_batch(
    indices: npt.NDArray,
    samples: List[npt.NDArray],
    covs: List[List[npt.NDArray]],
    ranges: List[Tuple[int, int]],
    n_state: int,
    initial_lambdas: npt.NDArray,
    init_lambda: float,
    ncall: int,
) -> Tuple[npt.NDArray, npt.NDArray]:
    lambdas = np.empty((len(indices), n_state))
    energies = np.empty_like(lambdas)
    for row, index in enumerate(indices):
        sample_data = CorrelatedData([sample[index] for sample in samples], covs)
        spectrum = _lfilter_spectrum(
            sample_data,
            ranges,
            n_state,
            initial_lambdas=initial_lambdas,
            init_lambda=init_lambda,
            ncall=ncall,
        )
        lambdas[row] = spectrum.lambdas
        energies[row] = spectrum.energies
    return lambdas, energies


@overload
def lfilter_spectrum(
    data: CorrelatedData,
    ranges: List[Tuple[int, int]] | Tuple[int, int],
    n_state: int,
    *,
    m_guess: Optional[float] = None,
    initial_lambdas: Optional[npt.NDArray] = None,
    init_lambda: float = 100.0,
    ncall: int = 5000,
    workers: int = 1,
) -> LaplaceFilterEnergies[npt.NDArray]: ...


@overload
def lfilter_spectrum(
    data: List[BootstrapArray] | BootstrapArray,
    ranges: List[Tuple[int, int]] | Tuple[int, int],
    n_state: int,
    *,
    m_guess: Optional[float] = None,
    initial_lambdas: Optional[npt.NDArray] = None,
    init_lambda: float = 100.0,
    ncall: int = 5000,
    workers: int = 1,
) -> LaplaceFilterEnergies[BootstrapArray]: ...


@overload
def lfilter_amplitudes(
    data: CorrelatedData,
    ranges: List[Tuple[int, int]] | Tuple[int, int],
    lambdas: npt.NDArray,
    *,
    amplitude_lambda: Optional[float] = None,
    time_period: Optional[int] = None,
) -> LaplaceFilterAmplitudes[npt.NDArray]: ...


@overload
def lfilter_amplitudes(
    data: List[BootstrapArray] | BootstrapArray,
    ranges: List[Tuple[int, int]] | Tuple[int, int],
    lambdas: npt.NDArray | BootstrapArray,
    *,
    amplitude_lambda: Optional[float] = None,
    time_period: Optional[int] = None,
) -> LaplaceFilterAmplitudes[BootstrapArray]: ...


def lfilter_amplitudes(
    data: CorrelatedData | List[BootstrapArray] | BootstrapArray,
    ranges: List[Tuple[int, int]] | Tuple[int, int],
    lambdas: npt.NDArray | BootstrapArray,
    *,
    amplitude_lambda: Optional[float] = None,
    time_period: Optional[int] = None,
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

    # build Laplace transform matrix
    #
    # L_ti = exp(-E_i*t) + exp(-E_i*(T-t)) for periodic time boundaries
    #
    # if several quantities are present, L is extended as a block-diagonal matrix for
    # every time range.
    lmat = np.zeros((*energies.shape[:-1], n_points, n_amplitudes))
    offset = 0
    for quantity, (start, stop) in enumerate(ranges):
        time = np.arange(start, stop)
        block = np.exp(-energies[..., :, None] * time)
        if time_period is not None:
            block += np.exp(-energies[..., :, None] * (time_period - time))
        size = time.size
        lmat[
            ...,
            offset : offset + size,
            quantity * n_states : (quantity + 1) * n_states,
        ] = np.swapaxes(block, -1, -2)
        offset += size

    # linear regression for the (potentially filtered) amplitudes
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
    filtered_amplitudes = np.linalg.solve(lmat_vinv_lmat, rhs).squeeze(-1)

    central_lmat = lmat[0] if lmat.ndim == 3 else lmat
    central_y = y[0] if y.ndim == 2 else y
    central_amplitudes = (
        filtered_amplitudes[0] if filtered_amplitudes.ndim == 2 else filtered_amplitudes
    )
    residual = central_y - central_lmat @ central_amplitudes
    chi2 = float(cov_quadratic_form(residual, cov))

    # package final result, if the optional amplitude_lambda regulator was provided,
    # amplitudes are corrected by a factor 1 / (amplitude_lambda^2 - Etilde_i^2)
    # with Etilde_i^2 = 2.0 * [cosh(E_i) - 1.0] (cf. lfilter_tilde)
    shape = (*filtered_amplitudes.shape[:-1], n_quantities, n_states)
    amplitudes = filtered_amplitudes.reshape(shape)
    if amplitude_lambda is not None:
        factor = np.asarray(lfilter_factor(amplitude_lambda, energies))
        amplitudes /= factor[..., None, :]
    if n_quantities == 1:
        amplitudes = amplitudes[..., 0, :]
    if n_bootstrap is not None:
        amplitudes = BootstrapArray(amplitudes)
    return LaplaceFilterAmplitudes(
        amplitudes=amplitudes,
        chi2=chi2,
        p_value=stats.chi2.sf(chi2, dof).item(),
        dof=dof,
        cdr=cdr(cov),
    )


def lfilter_spectrum(
    data: CorrelatedData | List[BootstrapArray] | BootstrapArray,
    ranges: List[Tuple[int, int]] | Tuple[int, int],
    n_state: int,
    *,
    m_guess: Optional[float] = None,
    initial_lambdas: Optional[npt.NDArray] = None,
    init_lambda: float = 100.0,
    ncall: int = 5000,
    workers: int = 1,
) -> LaplaceFilterEnergies[npt.NDArray] | LaplaceFilterEnergies[BootstrapArray]:
    """Fit a fixed number of Laplace-filter states.

    Args:
        data: Correlated data or aligned bootstrap samples.
        ranges: One half-open index interval `(start, stop)` per quantity.
            For example, `(6, 22)` selects indices 6 through 21.
        n_state: Number of exponential states to fit.
    """
    if isinstance(ranges, tuple):
        ranges = [ranges]
    if isinstance(data, CorrelatedData):
        return _lfilter_spectrum(
            data,
            ranges,
            n_state,
            m_guess=m_guess,
            initial_lambdas=initial_lambdas,
            init_lambda=init_lambda,
            ncall=ncall,
        )
    if isinstance(data, BootstrapArray):
        data = [data]
    if not isinstance(data, list):
        raise TypeError("data must be CorrelatedData or a list of BootstrapArray")
    if workers < 1:
        raise ValueError("workers must be positive")

    corr_data = make_correlated_data(data)
    n_bootstrap = data[0].samples.shape[0]
    lambdas = np.empty((n_bootstrap + 1, n_state))
    energies = np.empty_like(lambdas)
    central = _lfilter_spectrum(
        corr_data,
        ranges,
        n_state,
        m_guess=m_guess,
        initial_lambdas=initial_lambdas,
        init_lambda=init_lambda,
        ncall=ncall,
    )
    lambdas[0] = central.lambdas
    energies[0] = central.energies
    samples = [bootstrap.samples for bootstrap in data]
    if workers == 1:
        for sample in range(n_bootstrap):
            corr_data.set_means([values[sample] for values in samples])
            fit = _lfilter_spectrum(
                corr_data,
                ranges,
                n_state,
                initial_lambdas=central.lambdas,
                init_lambda=init_lambda,
                ncall=ncall,
            )
            lambdas[sample + 1] = fit.lambdas
            energies[sample + 1] = fit.energies
    else:
        workers = min(workers, n_bootstrap)
        covs = corr_data.covs
        batches = [
            batch
            for batch in np.array_split(np.arange(n_bootstrap), workers)
            if len(batch)
        ]
        try:
            context = mp.get_context("fork")
        except ValueError:
            context = None
        fit_batch = partial(
            _spectrum_batch,
            samples=samples,
            covs=covs,
            ranges=ranges,
            n_state=n_state,
            initial_lambdas=central.lambdas,
            init_lambda=init_lambda,
            ncall=ncall,
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            for indices, (batch_lambdas, batch_energies) in zip(
                batches, executor.map(fit_batch, batches)
            ):
                lambdas[indices + 1] = batch_lambdas
                energies[indices + 1] = batch_energies
    return LaplaceFilterEnergies(
        energies=BootstrapArray(energies),
        lambdas=BootstrapArray(lambdas),
        t2=central.t2,
        p_value=central.p_value,
        dof=central.dof,
        cdr=central.cdr,
    )


def lfilter_spectrum_test(
    data: CorrelatedData,
    ranges: List[Tuple[int, int]],
    n_state: int,
    alpha: float = 0.05,
    verbose: bool = False,
    init_lambda: float = 100.0,
    ncall: int = 5000,
) -> LaplaceFilterSpectrumTest:
    """Test successive Laplace-filter state counts.

    Args:
        data: Correlated time-series data.
        ranges: One half-open index interval `(start, stop)` per quantity.
            For example, `(6, 22)` selects indices 6 through 21.
        n_state: Largest state count tested.
        alpha: Family-wise significance threshold for the state-count test.
        verbose: Print the fitted spectra and test summary.
        init_lambda: Initial regulator for an added state.
        ncall: Maximum Minuit function evaluations per fit.
    """
    msg = ""
    res = LaplaceFilterSpectrumTest()
    if verbose:
        ranges_str = ", ".join(f"[{start}, {stop})" for start, stop in ranges)
        print(f"==== Laplace filter spectrum -- ranges = {ranges_str}")
    p = []
    pb = []
    t2 = []
    dt2 = []
    previous_lambs: Optional[npt.NDArray] = None
    for r in range(1, n_state + 1):
        initial_lambdas = (
            None if previous_lambs is None else np.append(previous_lambs, init_lambda)
        )
        spectrum = lfilter_spectrum(
            data,
            ranges,
            r,
            initial_lambdas=initial_lambdas,
            init_lambda=init_lambda,
            ncall=ncall,
        )
        res.spectra.append(spectrum)
        t2.append(spectrum.t2)
        p.append(spectrum.p_value)
        previous_lambs = spectrum.lambdas
        if verbose:
            msg = (
                f"{r} states: T^2_{r} = {spectrum.t2:.2e} "
                f"(p_{r} = {spectrum.p_value:.2e})"
            )
        if r > 1:
            dt2.append(math.fabs(t2[-2] - t2[-1]))
            pb.append(1 - stats.chi2.cdf(dt2[-1], 1))
            if verbose:
                msg += f", ΔT^2_{r - 1} = {dt2[-1]:.2e} (pb_{r - 1} = {pb[-1]:.2e})"
        if verbose:
            print(f"{msg}, Lambda = {spectrum.lambdas}")
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
    res.pbar_val = np.array(pb)
    res.dt2 = np.array(dt2)
    return res
