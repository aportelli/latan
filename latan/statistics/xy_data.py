from numbers import Integral, Real
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from latan.statistics.correlated_data import CorrelatedData


class XYData:
    """Joint data for a relationship between x coordinates and y points.

    Every y coordinate has one value per observation point. Each x coordinate
    may instead have fewer raw values, which `x_map` assigns to observations.
    This supports one shared stochastic x value used by many y points without
    duplicating it in the covariance matrix.

    Example:
        ```python
        # Three stochastic x values are shared by five y values.
        x_values = np.array([0.0, 0.5, 1.0])
        y_values = np.array([1.1, 0.9, 1.8, 2.7, 2.5])
        joint = CorrelatedData(
            means=[x_values, y_values],
            covs=[[cov_xx, cov_xy], [cov_yy]],
        )
        data = XYData(
            joint,
            x=[0],
            y_indices=[1],
            x_map=[np.array([0, 0, 1, 2, 2])],
            x_names=["x"],
            y_names=["y"],
        )
        # The five points are:
        # (0.0, 1.1), (0.0, 0.9), (0.5, 1.8), (1.0, 2.7), (1.0, 2.5).
        ```
    """

    _data: CorrelatedData
    _x: Tuple[int | npt.NDArray, ...]
    _x_map: Tuple[npt.NDArray, ...]
    _y_indices: Tuple[int, ...]
    _x_names: Tuple[str, ...]
    _y_names: Tuple[str, ...]
    _n_points: int
    _exact_x: frozenset[int]

    def __init__(
        self,
        data: CorrelatedData,
        x: Sequence[int | npt.NDArray],
        y_indices: Sequence[int],
        *,
        x_map: Sequence[int | npt.NDArray | None] | None = None,
        x_names: Sequence[str] | None = None,
        y_names: Sequence[str] | None = None,
    ) -> None:
        """Create x/y data with optional x-to-observation mappings.

        Args:
            data: Joint correlated data containing every stochastic x and y
                coordinate. Y quantities must have shape `(n_points,)`.
            x: One entry per x coordinate. An integer identifies a stochastic
                quantity in `data`; an array supplies exact raw x values.
            y_indices: Indices of y quantities in `data`. Together with
                stochastic x indices, they must partition `data`.
            x_map: One map per x coordinate. `None` uses the identity map,
                an integer uses one raw x value for every observation, and an
                integer array maps every observation to a raw x value.
            x_names: Optional names for x coordinates. Defaults to `x0`,
                `x1`, and so on.
            y_names: Optional names for y coordinates. Defaults to `y0`,
                `y1`, and so on.
        """
        # validate no empty data was provided
        x = tuple(x)
        y_indices = tuple(y_indices)
        if not x or not y_indices:
            raise ValueError("x and y_indices must not be empty")

        # separate exact x values and inexact x indices
        x_indices: list[int] = []
        exact_x: list[int] = []
        for i, val in enumerate(x):
            if isinstance(val, np.ndarray):
                if val.ndim != 1:
                    raise ValueError(f"exact x coordinate {i} is not one-dimensional")
                exact_x.append(i)
            else:
                x_indices.append(val)

        # validate inexact x & y indices
        indices = (*x_indices, *y_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("stochastic x and y indices must be disjoint and unique")
        if set(indices) != set(range(data.n_quantities)):
            raise ValueError("stochastic x and y indices must partition all quantities")

        # validate y data size
        y_sizes = [data.mean(index).size for index in y_indices]
        if len(set(y_sizes)) != 1:
            raise ValueError("all y coordinates must have the same size")
        n_points = y_sizes[0]

        if x_map is None:
            x_map = (None,) * len(x)
        else:
            x_map = tuple(x_map)
            if len(x_map) != len(x):
                raise ValueError(f"x_map has {len(x_map)} entries, expected {len(x)}")

        # normalised list of x maps, each of shape (n_points,)
        maps: list[npt.NDArray] = []
        for i, (val, m) in enumerate(zip(x, x_map)):
            raw_size = val.size if isinstance(val, np.ndarray) else data.mean(val).size
            if m is None:
                if raw_size != n_points:
                    raise ValueError(
                        f"x coordinate {i} has {raw_size} raw values and None was provided as a map; "
                        f"data with less than {n_points} values require a map"
                    )
                map = np.arange(n_points, dtype=np.intp)
            elif isinstance(m, np.ndarray):
                map = m
                if map.ndim != 1 or map.size != n_points:
                    raise ValueError(f"x_map entry {i} must have shape ({n_points},)")
                if not np.issubdtype(map.dtype, np.integer):
                    raise TypeError(f"x_map entry {i} must contain integer indices")
            else:
                assert isinstance(m, Integral)
                map = np.full(n_points, m, dtype=np.intp)
            if np.any(map < 0) or np.any(map >= raw_size):
                raise IndexError(f"x_map entry {i} contains indices out of range")
            maps.append(map)

        def normalize_names(
            names: Sequence[str] | None, count: int, prefix: str
        ) -> Tuple[str, ...]:
            if names is None:
                return tuple(f"{prefix}{i}" for i in range(count))
            names = tuple(names)
            if len(names) != count:
                raise ValueError(
                    f"{prefix}_names has {len(names)} entries, expected {count}"
                )
            if len(set(names)) != len(names):
                raise ValueError(f"{prefix}_names must be unique")
            return names

        self._data = data
        self._x = x
        self._x_map = tuple(maps)
        self._y_indices = y_indices
        self._x_names = normalize_names(x_names, len(x), "x")
        self._y_names = normalize_names(y_names, len(y_indices), "y")
        self._n_points = n_points
        self._exact_x = frozenset(exact_x)

    @property
    def data(self) -> CorrelatedData:
        return self._data

    @property
    def n_points(self) -> int:
        """Number of y observation points."""
        return self._n_points

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
        return len(self._exact_x)

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
    def x_names(self) -> Tuple[str, ...]:
        """Names of x coordinates."""
        return self._x_names

    @property
    def y_names(self) -> Tuple[str, ...]:
        """Names of y coordinates."""
        return self._y_names

    @property
    def exact_x(self) -> frozenset[int]:
        return self._exact_x

    def x_values(self, index: int = 0) -> npt.NDArray:
        """Return raw x values before applying the observation map."""
        value = self._x[index]
        return value if isinstance(value, np.ndarray) else self._data.mean(value)

    def x_map(self, index: int = 0) -> npt.NDArray:
        """Return the raw-x index used by each observation point."""
        return self._x_map[index].copy()

    def x(self, index: int = 0) -> npt.NDArray:
        """Return x values expanded to one entry per observation point."""
        return self.x_values(index)[self._x_map[index]]

    def y(self, index: int = 0) -> npt.NDArray:
        return self._data.mean(self._y_indices[index])

    def is_exact_x(self, index: int = 0) -> bool:
        return isinstance(self._x[index], np.ndarray)

    def point_mask(
        self,
        include: Sequence[Tuple[float | None, float | None]] | None = None,
        exclude: Sequence[Tuple[float | None, float | None]] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return points selected by closed coordinate intervals.

        Every `include` or `exclude` entry is a `(low, high)` value interval
        for one x coordinate. Both finite endpoints are included; use `None`
        for an unbounded endpoint. When both are given, `exclude` is applied
        after `include` and therefore removes overlapping points.
        """

        def range_mask(
            ranges: Sequence[Tuple[float | None, float | None]], name: str
        ) -> npt.NDArray[np.bool_]:
            if len(ranges) != len(self._x):
                raise ValueError(
                    f"{name} has {len(ranges)} ranges, expected {len(self._x)}"
                )
            mask = np.ones(self.n_points, dtype=bool)
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
                values = self.x(i)
                if low is not None:
                    mask &= values >= low
                if high is not None:
                    mask &= values <= high
            return mask

        mask = np.ones(self.n_points, dtype=bool)
        if include is not None:
            mask &= range_mask(include, "include")
        if exclude is not None:
            mask &= ~range_mask(exclude, "exclude")
        if not np.any(mask):
            raise ValueError("point selection is empty")
        return mask
