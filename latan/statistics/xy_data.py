from numbers import Real
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from latan.statistics.correlated_data import CorrelatedData


class XYData:
    """Joint data for a pointwise relationship between `x` and `y`.

    All coordinates have the same number of points. Point `i` is one input
    vector `x[i]` paired with one output vector `y[i]`.

    The wrapped `CorrelatedData` contains every stochastic x coordinate and
    every y coordinate, including their cross-covariance blocks. An array in
    `x` is an exact coordinate and is not stored in that covariance data.
    Integer entries in `x` and all entries in `y_indices` identify
    stochastic quantities in the wrapped data.

    Example:
        ```python
        # joint holds one stochastic x coordinate (0) and one y coordinate (1).
        joint = CorrelatedData(
            means=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            covs=[[np.eye(2), np.zeros((2, 2))], [np.eye(2)]],
        )
        # The first x coordinate is exact; the second is joint quantity 0.
        data = XYData(joint, x=[np.array([0.0, 1.0]), 0], y_indices=[1])
        ```
    """

    _data: CorrelatedData
    _x: Tuple[int | npt.NDArray, ...]
    _y_indices: Tuple[int, ...]

    def __init__(
        self,
        data: CorrelatedData,
        x: Sequence[int | npt.NDArray],
        y_indices: Sequence[int],
    ) -> None:
        """Create data for a pointwise relationship between x and y.

        Args:
            data: Joint correlated data containing every stochastic x and y
                coordinate. Each quantity must have shape
                `(n_points,)`. Its covariance blocks retain all correlations
                between stochastic coordinates.
            x: One entry per x coordinate. An integer is the index of a
                stochastic quantity in `data`. A NumPy array with shape
                `(n_points,)` is an exact coordinate and is excluded from
                `data` and its covariance blocks.
            y_indices: Integer indices of y quantities in `data`. Together
                with the stochastic x indices, these must be a disjoint
                partition of every quantity in `data`.
        """
        x = tuple(x)
        y_indices = tuple(y_indices)
        if not x or not y_indices:
            raise ValueError("x and y_indices must not be empty")

        x_indices: list[int] = []
        for i, value in enumerate(x):
            if isinstance(value, np.ndarray):
                if value.ndim != 1:
                    raise ValueError(f"exact x coordinate {i} is not one-dimensional")
            else:
                x_indices.append(int(value))
        if not all(isinstance(index, (int, np.integer)) for index in y_indices):
            raise TypeError("y_indices must contain integer quantity indices")
        y_indices = tuple(int(index) for index in y_indices)

        indices = (*x_indices, *y_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("stochastic x and y indices must be disjoint and unique")
        if set(indices) != set(range(data.n_quantities)):
            raise ValueError("stochastic x and y indices must partition all quantities")

        sizes = [
            value.size if isinstance(value, np.ndarray) else data.mean(value).size
            for value in x
        ]
        sizes += [data.mean(index).size for index in y_indices]
        if len(set(sizes)) != 1:
            raise ValueError("all x and y coordinates must have the same size")

        self._data = data
        self._x = x
        self._y_indices = y_indices

    @property
    def data(self) -> CorrelatedData:
        return self._data

    @property
    def x_ndim(self) -> int:
        """Number of x coordinates."""
        return len(self._x)

    @property
    def x_inexact_ndim(self) -> int:
        """Number of stochastic x coordinates."""
        return self.x_ndim - self.x_exact_ndim

    @property
    def x_exact_ndim(self) -> int:
        """Number of exact x coordinates."""
        return len(self.exact_x)

    @property
    def y_ndim(self) -> int:
        """Number of y coordinates."""
        return len(self._y_indices)

    @property
    def x_indices(self) -> Tuple[Optional[int], ...]:
        return tuple(
            None if isinstance(value, np.ndarray) else value for value in self._x
        )

    @property
    def y_indices(self) -> Tuple[int, ...]:
        return self._y_indices

    @property
    def exact_x(self) -> frozenset[int]:
        return frozenset(
            index
            for index, value in enumerate(self._x)
            if isinstance(value, np.ndarray)
        )

    def x(self, index: int = 0) -> npt.NDArray:
        value = self._x[index]
        return value if isinstance(value, np.ndarray) else self._data.mean(value)

    def y(self, index: int = 0) -> npt.NDArray:
        return self._data.mean(self._y_indices[index])

    def is_exact_x(self, index: int = 0) -> bool:
        return isinstance(self._x[index], np.ndarray)

    def point_mask(
        self,
        include: Sequence[Tuple[float | None, float | None]] | None = None,
        exclude: Sequence[Tuple[float | None, float | None]] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return points selected by closed ranges of the x coordinates.

        A point must satisfy every `include` range and must not satisfy every
        `exclude` range. Use `(None, None)` for an unconstrained coordinate.

        Args:
            include: One `(low, high)` range per x coordinate. `None` leaves
                an endpoint unbounded. Points outside any range are removed.
            exclude: One `(low, high)` range per x coordinate. Matching points
                are removed after applying `include`.

        Returns:
            A Boolean array with shape `(n_points,)`.
        """

        def range_mask(
            ranges: Sequence[Tuple[float | None, float | None]], name: str
        ) -> npt.NDArray[np.bool_]:
            if len(ranges) != len(self._x):
                raise ValueError(
                    f"{name} has {len(ranges)} ranges, expected {len(self._x)}"
                )
            mask = np.ones(self.x().shape, dtype=bool)
            for i, bounds in enumerate(ranges):
                if len(bounds) != 2:
                    raise ValueError(f"{name} range {i} must contain two bounds")
                low, high = bounds
                if low is not None and not isinstance(low, Real):
                    raise TypeError(f"{name} lower bound {i} must be a number or None")
                if high is not None and not isinstance(high, Real):
                    raise TypeError(f"{name} upper bound {i} must be a number or None")
                if low is not None and high is not None and low > high:
                    raise ValueError(
                        f"{name} range {i} has lower bound above upper bound"
                    )
                x = self.x(i)
                if low is not None:
                    mask &= x >= low
                if high is not None:
                    mask &= x <= high
            return mask

        mask = np.ones(self.x().shape, dtype=bool)
        if include is not None:
            mask &= range_mask(include, "include")
        if exclude is not None:
            mask &= ~range_mask(exclude, "exclude")
        if not np.any(mask):
            raise ValueError("point selection is empty")
        return mask
