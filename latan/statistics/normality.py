from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from latan.display._common import non_gaussian_text
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

    def __str__(self) -> str:
        labels = (
            ["Value"]
            if self.is_scalar
            else [
                "[" + ", ".join(str(i) for i in index) + "]"
                for index in np.ndindex(self.observable_shape)
            ]
        )
        rows = [
            (
                label,
                f"{skewness:.4g}",
                f"{kurtosis:.4g}",
                f"{statistic:.4g}",
                non_gaussian_text(p_value, sigma=True),
            )
            for label, skewness, kurtosis, statistic, p_value in zip(
                labels,
                self.skewness.ravel(),
                self.kurtosis_excess.ravel(),
                self.reduced_statistic.ravel(),
                self.p_value.ravel(),
            )
        ]
        headers = ("Component", "Skewness", "Excess kurtosis", "K^2 / 2")
        widths = [
            max(len(header), *(len(row[i]) for row in rows))
            for i, header in enumerate(headers)
        ]
        title = "Normality diagnostic" if self.is_scalar else "Componentwise normality diagnostic"
        header = "  ".join(
            [
                f"{headers[0]:<{widths[0]}}",
                *(f"{item:>{width}}" for item, width in zip(headers[1:], widths[1:])),
            ]
        )
        body = "\n".join(
            "  ".join(
                [
                    f"{row[0]:<{widths[0]}}",
                    *(f"{item:>{width}}" for item, width in zip(row[1:4], widths[1:])),
                ]
            )
            + row[4]
            for row in rows
        )
        return f"{title}: {self.n_samples} samples\n{header}\n{body}"

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
