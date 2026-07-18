from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from latan.physics.laplace_filter.filter import lfilter
from latan.statistics.correlated_data import CorrelatedData
from latan.statistics.correlation import cov_quadratic_form


class LaplaceFilteredT2:
    _data: CorrelatedData
    _ranges: List[Tuple[int, int]]
    _filtered_data: CorrelatedData

    def __init__(self, data: CorrelatedData, ranges: List[Tuple[int, int]]) -> None:
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
    def ranges(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self._ranges)

    def __call__(self, lamb: npt.NDArray) -> float:
        for i in range(self._data.n_quantities):
            lfilter(self._data.mean(i), lamb, out=self._filtered_data.mean(i))
            for j in range(i, self._data.n_quantities):
                lfilter(
                    self._data.cov(i, j),
                    lamb,
                    dim=(0, 1),
                    out=self._filtered_data.cov(i, j),
                )
        mean_f, cov_f = self._filtered_data.total_mean_cov(self._ranges)
        return cov_quadratic_form(mean_f, cov_f)
