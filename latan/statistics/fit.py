import multiprocessing as mp
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast, overload

import numpy as np
import numpy.typing as npt
from iminuit import Minuit
from scipy import stats
from scipy.optimize import least_squares

from latan.display._common import (
    asymmetric_error_text,
    bootstrap_error_text,
    bootstrap_normality,
    non_gaussian_text,
    normality_significance,
)
from latan.display.fit_result import render_fit_result_html
from latan.statistics.bootstrap import BootstrapArray
from latan.statistics.chi2 import Chi2, PointRanges
from latan.statistics.correlation import cdr
from latan.statistics.model import Model
from latan.statistics.xy_data import XYData


@dataclass
class FitResult[T: npt.NDArray]:
    """Result of a correlated fit.

    `parameters` contains physical model parameters followed by latent-x
    parameters. When bootstrap samples are fitted, it is a `BootstrapArray`
    with the central result at index 0.
    """

    parameters: T
    chi2: float
    dof: int
    p_value: float
    cdr: float
    n_model_parameters: int
    parameter_names: tuple[str, ...] = ()

    @property
    def model_parameters(self) -> T:
        """Fitted physical model parameters."""
        return cast(T, self.parameters[..., : self.n_model_parameters])

    @property
    def latent_parameters(self) -> T:
        """Fitted latent-x parameters."""
        return cast(T, self.parameters[..., self.n_model_parameters :])

    @property
    def _display_parameter_names(self) -> tuple[str, ...]:
        if len(self.parameter_names) == self.n_model_parameters:
            return self.parameter_names
        return tuple(f"p_{i}" for i in range(self.n_model_parameters))

    def __repr__(self) -> str:
        msg = "Fit summary\n"
        if isinstance(self.parameters, BootstrapArray):
            mean = self.parameters.central
            err = self.parameters.error()
            lower, upper, non_gaussian, normality_p = bootstrap_normality(
                self.parameters
            )
            for name, value, error, lo, hi, ng, p_value in zip(
                self._display_parameter_names,
                mean,
                err,
                lower,
                upper,
                non_gaussian,
                normality_p,
            ):
                msg += (
                    f"{name} = {bootstrap_error_text(value, error)}"
                    f"{non_gaussian_text(p_value, errors=(asymmetric_error_text(value, lo, hi),) if ng else ())}\n"
                )
        else:
            for name, value in zip(self._display_parameter_names, self.parameters):
                msg += f"{name} = {value:.4g}\n"
        if self.parameters.shape[-1] > self.n_model_parameters:
            n_latent = self.parameters.shape[-1] - self.n_model_parameters
            msg += f"({n_latent} latent parameters hidden)\n"
        msg += f"chi^2/dof = {self.chi2:.4g}/{self.dof} = {self.chi2 / self.dof:.2g}\n"
        msg += f"p = {self.p_value:.2g} ({normality_significance(self.p_value):.2g}σ)\n"
        msg += f"CDR at minimum = {self.cdr:.2g} dB"
        return msg

    def _repr_html_(self) -> str:
        return render_fit_result_html(self)

# chi^2 minimization helper
# least-squares efficiently finds the minimum before Minuit validates and refines it
# SciPy least_squares is fast but tends to miss fit that have not converged,
# Minuit is slower but more conservative with checks.
def _minimize(chi2: Chi2, initial: npt.NDArray, ncall: int) -> npt.NDArray:
    # first LM least-squares pass
    result = least_squares(
        chi2.residual,
        initial,
        method="lm",
        max_nfev=ncall,
    )
    chi2_value = chi2(result.x)
    if not np.isfinite(chi2_value):
        raise RuntimeError("fit produced a non-finite chi2")

    # cost function using iMinuit call convention
    def cost(*parameters: float) -> float:
        return chi2(np.asarray(parameters))

    # final Minuit pass
    minimum = Minuit(cost, *result.x)
    minuit_result = minimum.migrad(ncall=ncall)
    assert minuit_result.fmin is not None
    if not minuit_result.fmin.is_valid:
        print("warning: invalid minimum")

    parameters = np.asarray(minimum.values)
    chi2_value = chi2(parameters)
    if not np.isfinite(chi2_value):
        raise RuntimeError("fit produced a non-finite chi2")
    return parameters


# helper fitting a batch of bootstrap samples
def _fit_batch(
    indices: npt.NDArray,
    chi2: Chi2,
    samples: Sequence[npt.NDArray],
    initial: npt.NDArray,
    ncall: int,
) -> npt.NDArray:
    parameters = np.empty((len(indices), chi2.n_parameters))
    for row, index in enumerate(indices):
        chi2.set_means([sample[index] for sample in samples])
        parameters[row] = _minimize(chi2, initial, ncall)
    return parameters


# helper validating and extracting bootstrap samples
def _bootstrap_samples(
    data: XYData,
    bootstrap: BootstrapArray | Sequence[BootstrapArray],
) -> list[npt.NDArray]:
    if isinstance(bootstrap, BootstrapArray):
        bootstrap = [bootstrap]
    else:
        bootstrap = list(bootstrap)
    if len(bootstrap) != data.data.n_quantities:
        raise ValueError(
            "number of bootstrap quantities and correlated-data quantities mismatch "
            f"(got {len(bootstrap)}, expected {data.data.n_quantities})"
        )
    n_samples = bootstrap[0].samples.shape[0]
    samples: list[npt.NDArray] = []
    for i, item in enumerate(bootstrap):
        mean = data.data.mean(i)
        if item.samples.shape != (n_samples, mean.size):
            raise ValueError(
                f"bootstrap quantity {i} has shape {item.samples.shape}, "
                f"expected ({n_samples}, {mean.size})"
            )
        if not np.allclose(item.central, mean):
            raise ValueError(
                f"bootstrap quantity {i} central value does not match data"
            )
        samples.append(item.samples)
    return samples


@overload
def fit(
    data: XYData,
    model: Model,
    p0: npt.NDArray,
    *,
    include: PointRanges | None = None,
    exclude: PointRanges | None = None,
    bootstrap: None = None,
    ncall: int = 5000,
    workers: int = 1,
    covariance: Literal["full", "diagonal"] = "full",
) -> FitResult[npt.NDArray]: ...


@overload
def fit(
    data: XYData,
    model: Model,
    p0: npt.NDArray,
    *,
    include: PointRanges | None = None,
    exclude: PointRanges | None = None,
    bootstrap: BootstrapArray | Sequence[BootstrapArray],
    ncall: int = 5000,
    workers: int = 1,
    covariance: Literal["full", "diagonal"] = "full",
) -> FitResult[BootstrapArray]: ...


def fit(
    data: XYData,
    model: Model,
    p0: npt.NDArray,
    *,
    include: PointRanges | None = None,
    exclude: PointRanges | None = None,
    bootstrap: BootstrapArray | Sequence[BootstrapArray] | None = None,
    ncall: int = 5000,
    workers: int = 1,
    covariance: Literal["full", "diagonal"] = "full",
) -> FitResult[npt.NDArray] | FitResult[BootstrapArray]:
    """Fit a model to correlated x/y data.

    A cheap uncorrelated fit with observed x values fixed preconditions the
    requested fit. When bootstrap samples are supplied, the central covariance
    remains fixed while every aligned sample is fitted.

    Args:
        data: Pointwise x/y data and the fixed central covariance.
        model: Model relating x coordinates to y coordinates.
        p0: Initial physical model parameters.
        include: One closed `(low, high)` value interval per x coordinate.
            Finite endpoints are included; `None` is unbounded.
        exclude: Closed coordinate intervals removed after `include`.
        bootstrap: One bootstrap array per stochastic quantity in `data`.
        ncall: Maximum number of least-squares function evaluations.
        workers: Number of bootstrap worker processes.
        covariance: Use the full covariance matrix or only its diagonal.
    """

    # argument validation
    if ncall < 1:
        raise ValueError("ncall must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    p0 = np.asarray(p0, dtype=float)
    if p0.ndim != 1 or p0.size != model.n_par:
        raise ValueError(f"p0 must have shape ({model.n_par},), got {p0.shape}")

    # chi^2 function and validation
    chi2 = Chi2(
        data,
        model,
        include=include,
        exclude=exclude,
        covariance=covariance,
    )
    if chi2.dof <= 0:
        raise ValueError(f"chi^2 has non-positive degrees of freedom ({chi2.dof})")

    # uncorrelated fit as preconditioner
    preconditioner = chi2.uncorrelated(exact_x=True)
    preconditioned = _minimize(preconditioner, p0, ncall)

    # central value fit
    central_parameters = _minimize(chi2, chi2.full_parameters(preconditioned), ncall)
    central_chi2 = chi2(central_parameters)
    if covariance == "full":
        assert chi2._cov is not None
        fit_cdr = cdr(chi2._cov)
    else:
        fit_cdr = 0.0
    central_result = FitResult(
        parameters=central_parameters,
        chi2=central_chi2,
        dof=chi2.dof,
        p_value=stats.chi2.sf(central_chi2, chi2.dof).item(),
        cdr=fit_cdr,
        n_model_parameters=model.n_par,
        parameter_names=model.parameter_names,
    )

    # if fit is not bootstrapped we are done
    if bootstrap is None:
        return central_result

    # if bootstrapped do potentially parallel loop on bootstrap samples
    samples = _bootstrap_samples(data, bootstrap)
    n_bootstrap = samples[0].shape[0]
    parameters = np.empty((n_bootstrap + 1, chi2.n_parameters))
    parameters[0] = central_parameters
    means = [data.data.mean(i).copy() for i in range(data.data.n_quantities)]
    if workers == 1:
        try:
            for sample in range(n_bootstrap):
                chi2.set_means([values[sample] for values in samples])
                parameters[sample + 1] = _minimize(chi2, central_parameters, ncall)
        finally:
            chi2.set_means(means)
    else:
        workers = min(workers, n_bootstrap)
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
            _fit_batch,
            chi2=chi2,
            samples=samples,
            initial=central_parameters,
            ncall=ncall,
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            for indices, batch_parameters in zip(
                batches, executor.map(fit_batch, batches)
            ):
                parameters[indices + 1] = batch_parameters
    return FitResult(
        parameters=BootstrapArray(parameters),
        chi2=central_result.chi2,
        dof=central_result.dof,
        p_value=central_result.p_value,
        cdr=central_result.cdr,
        n_model_parameters=central_result.n_model_parameters,
        parameter_names=central_result.parameter_names,
    )
