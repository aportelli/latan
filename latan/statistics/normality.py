from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from latan.display.normality import render_normality_html
from latan.statistics.bootstrap import BootstrapArray


@dataclass
class NormalityTest:
    """Componentwise asymptotic normality diagnostics."""

    n_samples: int
    observable_shape: tuple[int, ...]
    skewness: npt.NDArray
    kurtosis_excess: npt.NDArray
    reduced_statistic: npt.NDArray
    p_value: npt.NDArray
    qq_theoretical: npt.NDArray
    qq_observed: npt.NDArray

    @property
    def n_components(self) -> int:
        """Number of scalar observable components."""
        return int(np.prod(self.observable_shape, dtype=int))

    @property
    def is_scalar(self) -> bool:
        """Whether the tested observable is scalar."""
        return not self.observable_shape

    def __repr__(self) -> str:
        return (
            f"Componentwise normality diagnostic: {self.n_samples} samples\n"
            f"skewness = {np.array2string(self.skewness, precision=4)}\n"
            f"excess kurtosis = {np.array2string(self.kurtosis_excess, precision=4)}\n"
            f"D'Agostino-Pearson K^2 / 2 = {np.array2string(self.reduced_statistic, precision=4)}\n"
            f"p-value = {np.array2string(self.p_value, precision=4)}"
        )

    def _repr_html_(self) -> str:
        return render_normality_html(self)


def normality_test(data: npt.ArrayLike | BootstrapArray) -> NormalityTest:
    """Return componentwise normality diagnostics for samples along axis 0.

    `BootstrapArray` inputs exclude their central value automatically. For any
    other array, every entry along axis 0 is treated as a sample.
    """
    if isinstance(data, BootstrapArray):
        samples = data.samples
    else:
        samples = np.asarray(data)
    if samples.ndim < 1:
        raise ValueError("data must have a sample axis")
    if samples.shape[0] < 8:
        raise ValueError("normality test requires at least 8 samples")

    n_samples = samples.shape[0]
    observable_shape = samples.shape[1:]
    values = samples.reshape(n_samples, -1)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    normaltest = stats.normaltest(values, axis=0)
    qq_observed = np.sort((values - mean) / scale, axis=0).reshape(
        n_samples, *observable_shape
    )
    qq_theoretical = stats.norm.ppf((np.arange(n_samples) + 0.5) / n_samples)

    return NormalityTest(
        n_samples=n_samples,
        observable_shape=observable_shape,
        skewness=np.asarray(stats.skew(values, axis=0, bias=False)).reshape(
            observable_shape
        ),
        kurtosis_excess=np.asarray(
            stats.kurtosis(values, axis=0, bias=False)
        ).reshape(observable_shape),
        reduced_statistic=(np.asarray(normaltest.statistic) / 2).reshape(
            observable_shape
        ),
        p_value=np.asarray(normaltest.pvalue).reshape(observable_shape),
        qq_theoretical=qq_theoretical,
        qq_observed=qq_observed,
    )
