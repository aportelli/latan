from typing import List, Optional, Tuple, overload

import numpy.typing as npt

from latan.physics.laplace_filter.amplitudes import (
    LaplaceFilterAmplitudes,
    lfilter_amplitudes,
)
from latan.physics.laplace_filter.correlations import lfilter_optimize_cdr
from latan.physics.laplace_filter.spectrum import (
    LaplaceFilterEnergies,
    lfilter_spectrum,
    lfilter_spectrum_test,
)
from latan.statistics.bootstrap import BootstrapArray
from latan.statistics.correlated_data import CorrelatedData, make_correlated_data


@overload
def lfilter_full_spectrum(
    data: CorrelatedData,
    tf: int,
    n_state: int,
    *,
    tested_states: Optional[int] = None,
    alpha: float = 0.05,
    time_period: Optional[int] = None,
    ncall: int = 5000,
    workers: int = 1,
    verbose: bool = False,
) -> Tuple[
    LaplaceFilterEnergies[npt.NDArray], LaplaceFilterAmplitudes[npt.NDArray]
]: ...


@overload
def lfilter_full_spectrum(
    data: BootstrapArray,
    tf: int,
    n_state: int,
    *,
    tested_states: Optional[int] = None,
    alpha: float = 0.05,
    time_period: Optional[int] = None,
    ncall: int = 5000,
    workers: int = 1,
    verbose: bool = False,
) -> Tuple[
    LaplaceFilterEnergies[BootstrapArray], LaplaceFilterAmplitudes[BootstrapArray]
]: ...


def lfilter_full_spectrum(
    data: CorrelatedData | BootstrapArray,
    tf: int,
    n_state: int,
    *,
    tested_states: Optional[int] = None,
    alpha: float = 0.05,
    time_period: Optional[int] = None,
    ncall: int = 5000,
    workers: int = 1,
    verbose: bool = False,
) -> Tuple[LaplaceFilterEnergies, LaplaceFilterAmplitudes]:
    if tested_states is None:
        tested_states = n_state + 1
    if isinstance(data, CorrelatedData):
        if data.n_quantities > 1:
            raise ValueError(
                "lfilter_full_spectrum is a convenience function for single-quantity data, you might want to consider a custom procedure for combined analysis"
            )
        test_data = data
    else:
        test_data = make_correlated_data(data)
    fit_ti = -1
    for ti in range(tested_states + 1, tf - tested_states):
        test = lfilter_spectrum_test(
            test_data,
            [(ti, tf)],
            tested_states,
            verbose=verbose,
            alpha=alpha,
            ncall=ncall,
        )
        if test.sig_states == n_state:
            fit_ti = ti
            if verbose:
                print(f"earliest {n_state}-state t_i = {fit_ti}")
            break
    if fit_ti < 0:
        raise RuntimeError(f"no initial time with {n_state} significant states found")
    energies = lfilter_spectrum(
        data, (fit_ti, tf), n_state, workers=workers, ncall=ncall
    )
    _, cov = test_data.total_mean_cov()
    la = lfilter_optimize_cdr(cov, (fit_ti, tf))
    if verbose:
        print(f"optimal CDR lambda = {la}")
    amplitudes = lfilter_amplitudes(
        data,
        (fit_ti, tf),
        energies.lambdas,
        amplitude_lambda=la,
        time_period=time_period,
    )
    return energies, amplitudes
