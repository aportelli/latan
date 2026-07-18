import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import least_squares

from latan.statistics.bootstrap import BootstrapArray
from latan.statistics.chi2 import Chi2, PointRanges
from latan.statistics.model import Model
from latan.statistics.xy_data import XYData


@dataclass
class FitResult:
    """Result of a correlated fit.

    `parameters` contains physical model parameters followed by latent-x
    parameters. When bootstrap samples are fitted, it is a `BootstrapArray`
    with the central result at index 0.
    """

    parameters: npt.NDArray | BootstrapArray
    chi2: float
    dof: int
    p_value: float
    n_model_parameters: int

    @property
    def model_parameters(self) -> npt.NDArray | BootstrapArray:
        """Fitted physical model parameters."""
        return self.parameters[..., : self.n_model_parameters]

    @property
    def latent_parameters(self) -> npt.NDArray | BootstrapArray:
        """Fitted latent-x parameters."""
        return self.parameters[..., self.n_model_parameters :]


# chi2 minimization helper
# least-squares on residuals was tested to be generally faster than Minuit
def _minimize(chi2: Chi2, initial: npt.NDArray, ncall: int) -> npt.NDArray:
    result = least_squares(
        chi2.residual,
        initial,
        method="lm",
        max_nfev=ncall,
    )
    if not result.success:
        raise RuntimeError(f"fit failed: {result.message}")
    return result.x


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
) -> FitResult:
    """Fit a model to correlated x/y data.

    A cheap uncorrelated fit with observed x values fixed preconditions the
    full correlated fit. When bootstrap samples are supplied, the central
    covariance remains fixed while every aligned sample is fitted.

    Args:
        data: Pointwise x/y data and the fixed central covariance.
        model: Model relating x coordinates to y coordinates.
        p0: Initial physical model parameters.
        include: Closed x-coordinate ranges defining retained points.
        exclude: Closed x-coordinate ranges removed after `include`.
        bootstrap: One bootstrap array per stochastic quantity in `data`.
        ncall: Maximum number of least-squares function evaluations.
        workers: Number of bootstrap worker processes.
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
    chi2 = Chi2(data, model, include=include, exclude=exclude)
    if chi2.dof <= 0:
        raise ValueError(f"chi^2 has non-positive degrees of freedom ({chi2.dof})")

    # uncorrelated fit as preconditioner
    preconditioner = chi2.uncorrelated(exact_x=True)
    preconditioned = _minimize(preconditioner, p0, ncall)

    # central value fit
    central_parameters = _minimize(chi2, chi2.full_parameters(preconditioned), ncall)
    central_chi2 = chi2(central_parameters)
    result = FitResult(
        parameters=central_parameters,
        chi2=central_chi2,
        dof=chi2.dof,
        p_value=stats.chi2.sf(central_chi2, chi2.dof).item(),
        n_model_parameters=model.n_par,
    )

    # if fit is not bootstrapped we are done
    if bootstrap is None:
        return result

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
    result.parameters = BootstrapArray(parameters)
    return result
