from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from latan.statistics.bootstrap import BootstrapArray


class CorrelatedData:
    _mean: List[npt.NDArray]
    _cov: List[List[npt.NDArray]]

    def __init__(
        self,
        mean: List[npt.NDArray],
        cov: List[List[npt.NDArray]],
    ) -> None:
        n = len(mean)
        if n == 0:
            raise ValueError("mean list is empty")
        for m in mean:
            if m.ndim != 1:
                raise ValueError(
                    f"all means are expected to have 1 dimension (got {m.ndim})"
                )
        if len(cov) != n:
            raise ValueError(
                f"number of covariance rows and quantities mismatch "
                f"(got {len(cov)}, expected {n})"
            )
        for i in range(n):
            expected_blocks = n - i
            if len(cov[i]) != expected_blocks:
                raise ValueError(
                    f"covariance row {i} has {len(cov[i])} blocks "
                    f"(expected {expected_blocks})"
                )
            for j in range(i, n):
                block = cov[i][j - i]
                if block.ndim != 2:
                    raise ValueError(
                        f"a covariance matrix is not dimension 2 "
                        f"((i,j) = ({i},{j}), ndim = {block.ndim})"
                    )
                cov_nx, cov_ny = block.shape
                nx = mean[i].shape[0]
                ny = mean[j].shape[0]
                if cov_nx != nx or cov_ny != ny:
                    raise ValueError(
                        f"(i,j) = ({i},{j}) covariance matrix does not have shape ({nx}, {ny}) (got ({cov_nx}, {cov_ny}))"
                    )
            if not np.allclose(cov[i][0], cov[i][0].T):
                raise ValueError(
                    f"diagonal covariance block ({i},{i}) is not symmetric"
                )
        self._mean = mean
        self._cov = cov

    def _cov_block(self, i: int, j: int) -> npt.NDArray:
        if i <= j:
            return self._cov[i][j - i]
        return self._cov[j][i - j].T

    @property
    def n_quantities(self) -> int:
        return len(self._mean)

    def _validate_index(self, index: int) -> None:
        if not 0 <= index < self.n_quantities:
            raise IndexError(f"quantity index {index} out of range")

    def mean(self, index: int) -> npt.NDArray:
        self._validate_index(index)
        return self._mean[index]

    def covariance(self, i: int, j: int) -> npt.NDArray:
        self._validate_index(i)
        self._validate_index(j)
        return self._cov_block(i, j)

    def size(self, index: int) -> int:
        self._validate_index(index)
        return len(self._mean[index])

    def total_mean_cov(
        self, ranges: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        slices = []
        n = self.n_quantities
        if ranges is not None:
            if len(ranges) != n:
                raise ValueError(
                    f"number of ranges and quantities mismatch (got {len(ranges)}, expected {n})"
                )
            for r in ranges:
                slices.append(slice(r[0], r[1]))
        else:
            slices += [slice(None)] * n
        mean = np.concatenate([self._mean[i][slices[i]] for i in range(n)])
        cov = np.block([
            [self._cov_block(i, j)[slices[i], slices[j]] for j in range(n)]
            for i in range(n)
        ])
        return mean, cov


def make_correlated_data(
    data: List[BootstrapArray] | List[npt.NDArray],
) -> CorrelatedData:
    if not data:
        raise ValueError("data list is empty")

    if all(isinstance(datum, BootstrapArray) for datum in data):
        samples = [bootstrap.samples for bootstrap in data]
        means = [bootstrap.central for bootstrap in data]
        kind = "bootstrap"
        covariance_scale = 1.0
    elif all(type(datum) is np.ndarray for datum in data):
        samples = data
        means = [sample.mean(axis=0) for sample in samples]
        kind = "primary"
        covariance_scale = 1.0 / samples[0].shape[0]
    else:
        raise TypeError(
            "data must contain only BootstrapArray or only plain numpy.ndarray"
        )

    n_samples = samples[0].shape[0]
    if n_samples < 2:
        raise ValueError(f"at least two {kind} samples are required")
    for i, (m, s) in enumerate(zip(means, samples)):
        if m.ndim != 1:
            raise ValueError(
                f"{kind} datum {i} has a non-vector mean (ndim = {m.ndim})"
            )
        if s.ndim != 2:
            raise ValueError(
                f"{kind} datum {i} has non-matrix samples (ndim = {s.ndim})"
            )
        if s.shape != (n_samples, m.size):
            raise ValueError(
                f"{kind} datum {i} has sample shape {s.shape}, "
                f"expected ({n_samples}, {m.size})"
            )
    joint_samples = np.concatenate(samples, axis=1)
    joint_cov = np.atleast_2d(np.cov(joint_samples, rowvar=False)) * covariance_scale
    bounds = np.cumsum([0, *(mean.size for mean in means)])
    cov = [
        [
            joint_cov[bounds[i] : bounds[i + 1], bounds[j] : bounds[j + 1]]
            for j in range(i, len(means))
        ]
        for i in range(len(means))
    ]
    return CorrelatedData(means, cov)
