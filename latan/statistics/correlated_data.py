from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt

from latan.statistics.bootstrap import BootstrapArray


class CorrelatedData:
    """Means and covariance blocks for correlated vector quantities.

    Each quantity is a one-dimensional mean vector. `covs[i][j - i]` stores
    the covariance block between quantities `i` and `j` for `i <= j`.
    Lower-triangular covariance blocks are obtained by transposition.

    Example:
        ```python
        # A single quantity with two correlated components.
        data = CorrelatedData(
            means=np.array([1.0, 2.0]),
            covs=np.array([[0.04, 0.01], [0.01, 0.09]]),
        )
        ```

        ```python
        # Two quantities: covs stores the upper covariance-block triangle.
        data = CorrelatedData(
            means=[np.array([1.0, 2.0]), np.array([3.0])],
            covs=[
                [np.eye(2), np.array([[0.01], [0.02]])],
                [np.array([[0.09]])],
            ],
        )
        # data.cov(1, 0) is the transpose of data.cov(0, 1).
        ```
    """

    _means: List[npt.NDArray]
    _covs: List[List[npt.NDArray]]

    def __init__(
        self,
        means: List[npt.NDArray] | npt.NDArray,
        covs: List[List[npt.NDArray]] | npt.NDArray,
    ) -> None:
        """Create correlated vector data.

        Args:
            means: Either one NumPy array with shape `(n,)` or a list of
                NumPy arrays with shapes `(n_0,)`, `(n_1,)`, and so on.
            covs: For one quantity, a NumPy array with shape `(n, n)`. For
                multiple quantities, upper-triangular covariance blocks:
                `covs[i][j - i]` must have shape `(n_i, n_j)` for
                `i <= j`. Each diagonal block must be symmetric.
        """
        if isinstance(means, np.ndarray):
            if not isinstance(covs, np.ndarray):
                raise TypeError("single mean requires a single covariance matrix")
            means_list = [means]
            covs_list = [[covs]]
        else:
            if isinstance(covs, np.ndarray):
                raise TypeError("mean list requires covariance block rows")
            means_list = means
            covs_list = covs

        n = len(means_list)
        if n == 0:
            raise ValueError("mean list is empty")
        for m in means_list:
            if m.ndim != 1:
                raise ValueError(
                    f"all means are expected to have 1 dimension (got {m.ndim})"
                )
        if len(covs_list) != n:
            raise ValueError(
                f"number of covariance rows and quantities mismatch "
                f"(got {len(covs_list)}, expected {n})"
            )
        for i in range(n):
            expected_blocks = n - i
            if len(covs_list[i]) != expected_blocks:
                raise ValueError(
                    f"covariance row {i} has {len(covs_list[i])} blocks "
                    f"(expected {expected_blocks})"
                )
            for j in range(i, n):
                block = covs_list[i][j - i]
                if block.ndim != 2:
                    raise ValueError(
                        f"a covariance matrix is not dimension 2 "
                        f"((i,j) = ({i},{j}), ndim = {block.ndim})"
                    )
                cov_nx, cov_ny = block.shape
                nx = means_list[i].shape[0]
                ny = means_list[j].shape[0]
                if cov_nx != nx or cov_ny != ny:
                    raise ValueError(
                        f"(i,j) = ({i},{j}) covariance matrix does not have shape ({nx}, {ny}) (got ({cov_nx}, {cov_ny}))"
                    )
            if not np.allclose(covs_list[i][0], covs_list[i][0].T):
                raise ValueError(
                    f"diagonal covariance block ({i},{i}) is not symmetric"
                )
        self._means = means_list
        self._covs = covs_list

    def _cov_block(self, i: int, j: int) -> npt.NDArray:
        if i <= j:
            return self._covs[i][j - i]
        return self._covs[j][i - j].T

    @property
    def n_quantities(self) -> int:
        return len(self._means)

    def _validate_index(self, index: int) -> None:
        if not 0 <= index < self.n_quantities:
            raise IndexError(f"quantity index {index} out of range")

    def mean(self, index: int = 0) -> npt.NDArray:
        self._validate_index(index)
        return self._means[index]

    def set_means(self, means: List[npt.NDArray]) -> None:
        if len(means) != self.n_quantities:
            raise ValueError(
                f"number of means and quantities mismatch "
                f"(got {len(means)}, expected {self.n_quantities})"
            )
        for i, value in enumerate(means):
            if value.ndim != 1 or value.shape != self._means[i].shape:
                raise ValueError(
                    f"mean {i} has shape {value.shape}, expected {self._means[i].shape}"
                )
        self._means = means

    def cov(self, i: int = 0, j: int = 0) -> npt.NDArray:
        self._validate_index(i)
        self._validate_index(j)
        return self._cov_block(i, j)

    @property
    def covs(self) -> List[List[npt.NDArray]]:
        return self._covs

    def uncorrelated(self) -> "CorrelatedData":
        cov = [
            [
                np.diag(np.diag(self.cov(i, j)))
                if i == j
                else np.zeros_like(self.cov(i, j))
                for j in range(i, self.n_quantities)
            ]
            for i in range(self.n_quantities)
        ]
        return CorrelatedData(self._means, cov)

    def size(self, index: int) -> int:
        self._validate_index(index)
        return len(self._means[index])

    def total_mean_cov(
        self,
        ranges: Optional[Sequence[Tuple[int, int]]] = None,
        *,
        indices: Optional[Sequence[npt.NDArray]] = None,
        quantities: Optional[Sequence[int]] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Assemble selected means and covariance blocks into one vector and matrix.

        Args:
            ranges: One half-open `(start, stop)` range per selected quantity.
                This is mutually exclusive with `indices`.
            indices: One ordered one-dimensional integer array per selected
                quantity. This is mutually exclusive with `ranges`.
            quantities: Ordered quantity indices to include. By default, every
                quantity is included in its stored order.
        """
        if ranges is not None and indices is not None:
            raise ValueError("ranges and indices are mutually exclusive")

        if quantities is None:
            quantities = tuple(range(self.n_quantities))
        else:
            quantities = tuple(quantities)
            if len(set(quantities)) != len(quantities):
                raise ValueError("quantities must be unique")
            if not all(isinstance(index, (int, np.integer)) for index in quantities):
                raise TypeError("quantities must contain integer indices")
            quantities = tuple(int(index) for index in quantities)
        for index in quantities:
            self._validate_index(index)

        n = len(quantities)
        if ranges is not None:
            if len(ranges) != n:
                raise ValueError(
                    f"number of ranges and quantities mismatch (got {len(ranges)}, expected {n})"
                )
            selections = [slice(start, stop) for start, stop in ranges]
        elif indices is not None:
            if len(indices) != n:
                raise ValueError(
                    f"number of index arrays and quantities mismatch (got {len(indices)}, expected {n})"
                )
            selections = []
            for quantity, selection in zip(quantities, indices):
                selection = np.asarray(selection)
                if selection.ndim != 1 or not np.issubdtype(
                    selection.dtype, np.integer
                ):
                    raise TypeError("indices must be one-dimensional integer arrays")
                if np.any(selection < 0) or np.any(selection >= self.size(quantity)):
                    raise IndexError(
                        f"indices for quantity {quantity} are out of range"
                    )
                selections.append(selection.astype(np.intp, copy=False))
        else:
            selections = [slice(None)] * n

        mean = np.concatenate([
            self._means[quantity][selection]
            for quantity, selection in zip(quantities, selections)
        ])
        if indices is None:
            cov = np.block([
                [
                    self._cov_block(i, j)[selections[row], selections[column]]
                    for column, j in enumerate(quantities)
                ]
                for row, i in enumerate(quantities)
            ])
        else:
            index_selections: List[npt.NDArray] = []
            for selection in selections:
                assert isinstance(selection, np.ndarray)
                index_selections.append(selection)
            cov = np.block([
                [
                    self._cov_block(i, j)[
                        np.ix_(
                            index_selections[row],
                            index_selections[column],
                        )
                    ]
                    for column, j in enumerate(quantities)
                ]
                for row, i in enumerate(quantities)
            ])
        return mean, cov


def make_correlated_data(
    data: List[BootstrapArray] | List[npt.NDArray] | BootstrapArray | npt.NDArray,
) -> CorrelatedData:
    """Build correlated data from aligned bootstrap or primary samples.

    The covariance retains correlations between every supplied quantity. For
    primary samples, the covariance of the mean is used. For bootstrap data,
    the stored central value is used as the mean and the bootstrap samples
    define the covariance.

    Args:
        data: One `BootstrapArray` or NumPy array, or a non-empty list of
            arrays of the same kind. A plain NumPy array must have shape
            `(n_samples, n_components)`, with primary samples on axis 0.
            Every input must have the same number of samples. A
            `BootstrapArray` supplies its central value and aligned bootstrap
            samples.

    Returns:
        A `CorrelatedData` instance with one quantity per input array.
    """
    if isinstance(data, np.ndarray):
        data = [data]
    if not data:
        raise ValueError("data list is empty")

    if all(isinstance(datum, BootstrapArray) for datum in data):
        bootstrap_data = cast(List[BootstrapArray], data)
        samples = [bootstrap.samples for bootstrap in bootstrap_data]
        means = [bootstrap.central for bootstrap in bootstrap_data]
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
