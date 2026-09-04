import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import overload

import numpy as np
import numpy.typing as npt
from iminuit import Minuit
from scipy import stats

from latan.display._common import (
    asymmetric_error_text,
    bootstrap_error_text,
    bootstrap_normality,
    non_gaussian_text,
    normality_significance,
)
from latan.display.laplace_filter import (
    render_laplace_filter_energies_html,
)
from latan.physics.laplace_filter.filter import (
    lfilter_correlated_data,
    lfilter_tilde_inv,
)
from latan.physics.laplace_filter.t2 import LaplaceFilteredT2
from latan.statistics.bootstrap import BootstrapArray
from latan.statistics.correlated_data import CorrelatedData
from latan.statistics.correlation import cdr


def _spectrum_normality_text(
    energy: float,
    energy_lower: float,
    energy_upper: float,
    energy_non_gaussian: bool,
    energy_p_value: float,
    lamb: float,
    lambda_lower: float,
    lambda_upper: float,
    lambda_non_gaussian: bool,
    lambda_p_value: float,
) -> str:
    errors = ()
    if energy_non_gaussian:
        errors += (asymmetric_error_text(energy, energy_lower, energy_upper, "E"),)
    if lambda_non_gaussian:
        errors += (asymmetric_error_text(lamb, lambda_lower, lambda_upper, "λ"),)
    return non_gaussian_text((energy_p_value, lambda_p_value), errors=errors)


@dataclass
class LaplaceFilterEnergies[T: npt.NDArray]:
    energies: T
    lambdas: T
    ranges: tuple[tuple[int, int], ...]
    t2: float
    p_value: float
    dof: int
    cdr: float

    def __str__(self) -> str:
        msg = "Laplace-filter spectrum\n"
        if isinstance(self.energies, BootstrapArray):
            energies = self.energies.central
            energy_err = self.energies.error()
            assert isinstance(self.lambdas, BootstrapArray)
            lambdas = self.lambdas.central
            lambda_err = self.lambdas.error()
            energy_lower, energy_upper, energy_ng, energy_normality_p = (
                bootstrap_normality(self.energies)
            )
            lambda_lower, lambda_upper, lambda_ng, lambda_normality_p = (
                bootstrap_normality(self.lambdas)
            )
            for i, (
                energy,
                error,
                lamb,
                lamb_error,
                energy_lo,
                energy_hi,
                e_ng,
                energy_p,
                lambda_lo,
                lambda_hi,
                l_ng,
                lambda_p,
            ) in enumerate(
                zip(
                    energies,
                    energy_err,
                    lambdas,
                    lambda_err,
                    energy_lower,
                    energy_upper,
                    energy_ng,
                    energy_normality_p,
                    lambda_lower,
                    lambda_upper,
                    lambda_ng,
                    lambda_normality_p,
                )
            ):
                msg += (
                    f"E_{i} = {bootstrap_error_text(energy, error)}, "
                    f"lambda_{i} = {bootstrap_error_text(lamb, lamb_error)}"
                    f"{_spectrum_normality_text(energy, energy_lo, energy_hi, e_ng, energy_p, lamb, lambda_lo, lambda_hi, l_ng, lambda_p)}\n"
                )
        else:
            for i, (energy, lamb) in enumerate(zip(self.energies, self.lambdas)):
                msg += f"E_{i} = {energy:.4g}, lambda_{i} = {lamb:.4g}\n"
        ranges = ", ".join(f"[{start}, {stop})" for start, stop in self.ranges)
        msg += f"time range = {ranges}\n"
        msg += f"T^2/dof = {self.t2:.4g}/{self.dof} = {self.t2 / self.dof:.2g}\n"
        msg += f"p = {self.p_value:.2g} ({normality_significance(self.p_value):.2g}σ)\n"
        msg += f"CDR at minimum = {self.cdr:.2g} dB"
        return msg

    def _repr_html_(self) -> str:
        return render_laplace_filter_energies_html(self)


@dataclass
class LaplaceFilterSpectrumTest:
    spectra: list[LaplaceFilterEnergies[npt.NDArray]] = field(default_factory=list)
    dt2: npt.NDArray = field(default_factory=lambda: np.empty(0))
    pbar_val: npt.NDArray = field(default_factory=lambda: np.empty(0))
    sig_states: int = -1


def _lfilter_spectrum(
    data: CorrelatedData,
    ranges: list[tuple[int, int]],
    n_state: int,
    *,
    m_guess: float | None = None,
    initial_lambdas: npt.NDArray | None = None,
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
        ranges=tuple(ranges),
        t2=t2,
        p_value=stats.chi2.sf(t2, dof).item(),
        dof=dof,
        cdr=cdr(cov),
    )
    return spectrum


def _spectrum_batch(
    indices: npt.NDArray,
    samples: list[npt.NDArray],
    covs: list[list[npt.NDArray]],
    ranges: list[tuple[int, int]],
    n_state: int,
    initial_lambdas: npt.NDArray,
    init_lambda: float,
    ncall: int,
) -> tuple[npt.NDArray, npt.NDArray]:
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
    ranges: list[tuple[int, int]] | tuple[int, int],
    n_state: int,
    *,
    m_guess: float | None = None,
    initial_lambdas: npt.NDArray | None = None,
    init_lambda: float = 100.0,
    ncall: int = 5000,
    workers: int = 1,
) -> LaplaceFilterEnergies[npt.NDArray]: ...


@overload
def lfilter_spectrum(
    data: list[BootstrapArray] | BootstrapArray,
    ranges: list[tuple[int, int]] | tuple[int, int],
    n_state: int,
    *,
    m_guess: float | None = None,
    initial_lambdas: npt.NDArray | None = None,
    init_lambda: float = 100.0,
    ncall: int = 5000,
    workers: int = 1,
) -> LaplaceFilterEnergies[BootstrapArray]: ...


def lfilter_spectrum(
    data: CorrelatedData | list[BootstrapArray] | BootstrapArray,
    ranges: list[tuple[int, int]] | tuple[int, int],
    n_state: int,
    *,
    m_guess: float | None = None,
    initial_lambdas: npt.NDArray | None = None,
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

    corr_data = CorrelatedData.from_bootstrap(data)
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
        ranges=central.ranges,
        t2=central.t2,
        p_value=central.p_value,
        dof=central.dof,
        cdr=central.cdr,
    )


def lfilter_spectrum_test(
    data: CorrelatedData,
    ranges: list[tuple[int, int]],
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
    previous_lambs: npt.NDArray | None = None
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
    for j in range(n_state - 1):
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
