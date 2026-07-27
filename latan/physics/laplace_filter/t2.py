
import numpy as np
import numpy.typing as npt

from latan.physics.laplace_filter.filter import lfilter_correlated_data
from latan.statistics.correlated_data import CorrelatedData
from latan.statistics.correlation import cov_quadratic_form


class LaplaceFilteredT2:
    """Laplace-filtered T-squared function for correlated time series.

    `ranges` selects a half-open index interval `(start, stop)` for each
    quantity, following Python slicing: `start` is included and `stop` is
    excluded.
    """

    _data: CorrelatedData
    _ranges: list[tuple[int, int]]
    _filtered_data: CorrelatedData

    def __init__(self, data: CorrelatedData, ranges: list[tuple[int, int]]) -> None:
        """Create a Laplace-filtered T-squared function.

        Args:
            data: Correlated time-series data.
            ranges: One half-open index interval `(start, stop)` per
                quantity. For example, `(6, 22)` selects indices 6 through
                21.
        """
        if len(ranges) != data.n_quantities:
            raise ValueError(
                f"number of ranges and quantities mismatch "
                f"(got {len(ranges)}, expected {data.n_quantities})"
            )
        self._data = data
        self._ranges = list(ranges)
        mean_buf = [np.empty_like(data.mean(i)) for i in range(data.n_quantities)]
        cov_buf = [
            [np.zeros_like(data.cov(i, j)) for j in range(i, data.n_quantities)]
            for i in range(data.n_quantities)
        ]
        self._filtered_data = CorrelatedData(mean_buf, cov_buf)

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        """Half-open index intervals used for each data quantity."""
        return tuple(self._ranges)

    def __call__(self, lamb: npt.NDArray) -> float:
        lfilter_correlated_data(self._data, lamb, out=self._filtered_data)
        mean_f, cov_f = self._filtered_data.total_mean_cov(self._ranges)
        return float(cov_quadratic_form(mean_f, cov_f))
