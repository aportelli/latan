import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from iminuit import Minuit
from scipy import stats

from latan.physics.laplace_filter.filter import lfilter_tilde_inv
from latan.physics.laplace_filter.t2 import LaplaceFilteredT2
from latan.statistics.bootstrap import BootstrapArray
from latan.statistics.correlated_data import CorrelatedData, make_correlated_data


@dataclass
class LaplaceFilterSpectrum:
    energies: npt.NDArray | BootstrapArray
    lambdas: npt.NDArray | BootstrapArray
    t2: float
    p_value: float
    dof: int


@dataclass
class LaplaceFilterSpectrumTest:
    spectra: List[LaplaceFilterSpectrum] = field(default_factory=list)
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
) -> LaplaceFilterSpectrum:
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

    if m_guess is None:
        mean = data.mean()
        t_guess = mean.shape[0] // 4
        m_guess = math.log(mean[t_guess - 1] / mean[t_guess])

    t2_func = LaplaceFilteredT2(data, ranges)

    def cost(*lambdas):
        return t2_func(np.asarray(lambdas, dtype=float))

    names = [f"lambda_{i}" for i in range(n_state)]
    if initial_lambdas is None:
        t2_uncorr_func = LaplaceFilteredT2(data.uncorrelated(), ranges)

        def cost_uncorr(*lambdas):
            return t2_uncorr_func(np.asarray(lambdas, dtype=float))

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
    spectrum = LaplaceFilterSpectrum(
        energies=lfilter_tilde_inv(lambdas),
        lambdas=lambdas,
        t2=t2,
        p_value=stats.chi2.sf(t2, dof).item(),
        dof=dof,
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
) -> LaplaceFilterSpectrum:
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
    if not data:
        raise ValueError("bootstrap data list is empty")
    if not all(isinstance(datum, BootstrapArray) for datum in data):
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
    return LaplaceFilterSpectrum(
        energies=BootstrapArray(energies),
        lambdas=BootstrapArray(lambdas),
        t2=central.t2,
        p_value=central.p_value,
        dof=central.dof,
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
