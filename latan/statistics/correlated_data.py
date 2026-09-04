from collections.abc import Sequence

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

    _means: list[npt.NDArray]
    _covs: list[list[npt.NDArray]]
    _bootstrap: tuple[BootstrapArray, ...] | None

    def __init__(
        self,
        means: list[npt.NDArray] | npt.NDArray,
        covs: list[list[npt.NDArray]] | npt.NDArray,
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
        self._bootstrap = None

    def _set_bootstrap(self, bootstrap: Sequence[BootstrapArray] | None) -> None:
        """Attach already-validated bootstrap replicas."""
        self._bootstrap = None if bootstrap is None else tuple(bootstrap)

    @classmethod
    def _from_sample_arrays(
        cls,
        samples: list[npt.NDArray],
        means: list[npt.NDArray],
        *,
        kind: str,
        covariance_scale: float,
    ) -> "CorrelatedData":
        n_samples = samples[0].shape[0]
        if n_samples < 2:
            raise ValueError(f"at least two {kind} samples are required")
        for i, (mean, sample) in enumerate(zip(means, samples)):
            if mean.ndim != 1:
                raise ValueError(
                    f"{kind} datum {i} has a non-vector mean (ndim = {mean.ndim})"
                )
            if sample.ndim != 2:
                raise ValueError(
                    f"{kind} datum {i} has non-matrix samples (ndim = {sample.ndim})"
                )
            if sample.shape != (n_samples, mean.size):
                raise ValueError(
                    f"{kind} datum {i} has sample shape {sample.shape}, "
                    f"expected ({n_samples}, {mean.size})"
                )
        joint_samples = np.concatenate(samples, axis=1)
        joint_cov = (
            np.atleast_2d(np.cov(joint_samples, rowvar=False)) * covariance_scale
        )
        bounds = np.cumsum([0, *(mean.size for mean in means)])
        cov = [
            [
                joint_cov[bounds[i] : bounds[i + 1], bounds[j] : bounds[j + 1]]
                for j in range(i, len(means))
            ]
            for i in range(len(means))
        ]
        return cls(means, cov)

    @classmethod
    def from_samples(
        cls, data: list[npt.NDArray] | npt.NDArray
    ) -> "CorrelatedData":
        """Build correlated data from aligned primary samples.

        A plain NumPy array must have shape `(n_samples, n_components)`, with
        primary samples on axis 0. Each input must have the same number of
        samples. The resulting covariance is the covariance of the mean.
        """
        if isinstance(data, np.ndarray):
            data = [data]
        if not data:
            raise ValueError("data list is empty")
        if not all(type(datum) is np.ndarray for datum in data):
            raise TypeError("data must contain only plain numpy.ndarray")
        samples = data
        return cls._from_sample_arrays(
            samples,
            [sample.mean(axis=0) for sample in samples],
            kind="primary",
            covariance_scale=1.0 / samples[0].shape[0],
        )

    @classmethod
    def from_bootstrap(
        cls, data: list[BootstrapArray] | BootstrapArray
    ) -> "CorrelatedData":
        """Build correlated data from aligned bootstrap quantities.

        A `BootstrapArray` supplies the central mean and aligned replicas.
        The resulting covariance is the covariance of the replicas, which are
        retained for automatic bootstrap fits.
        """
        if isinstance(data, BootstrapArray):
            data = [data]
        elif isinstance(data, np.ndarray):
            raise TypeError("data must contain only BootstrapArray")
        if not data:
            raise ValueError("data list is empty")
        if not all(isinstance(datum, BootstrapArray) for datum in data):
            raise TypeError("data must contain only BootstrapArray")
        bootstrap = tuple(data)
        correlated = cls._from_sample_arrays(
            [item.samples for item in bootstrap],
            [item.central for item in bootstrap],
            kind="bootstrap",
            covariance_scale=1.0,
        )
        correlated._set_bootstrap(bootstrap)
        return correlated

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

    def set_means(self, means: list[npt.NDArray]) -> None:
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
    def covs(self) -> list[list[npt.NDArray]]:
        return self._covs

    @property
    def bootstrap(self) -> tuple[BootstrapArray, ...] | None:
        """Aligned bootstrap replicas, if this data was built from them."""
        return self._bootstrap

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
        data = CorrelatedData(self._means, cov)
        if self._bootstrap is not None:
            data._set_bootstrap(self._bootstrap)
        return data

    def size(self, index: int) -> int:
        self._validate_index(index)
        return len(self._means[index])

    def total_mean_cov(
        self,
        ranges: Sequence[tuple[int, int]] | None = None,
        *,
        indices: Sequence[npt.NDArray] | None = None,
        quantities: Sequence[int] | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
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
            index_selections: list[npt.NDArray] = []
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
