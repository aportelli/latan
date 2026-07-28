from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from latan.statistics.correlated_data import CorrelatedData
from latan.statistics.correlation import (
    cov_factor,
    cov_independent_residuals,
    cov_quadratic_form,
)
from latan.statistics.model import Model
from latan.statistics.xy_data import XYData

type PointRanges = Sequence[tuple[float | None, float | None]]


class Chi2:
    """Chi-squared function for correlated x/y data and a model.

    Inexact x coordinates are fitted as latent values. A mapped x coordinate
    contributes one latent value per selected unique raw x entry, even when
    several y observations refer to that entry.

    This strategy is described in Boggs, Byrd, and Schnabel 1987,
    https://doi.org/10.1137/0908085.
    """

    _xydata: XYData
    _model: Model
    _n_par: int
    _covariance_mode: Literal["full", "diagonal"]
    _include: PointRanges | None
    _exclude: PointRanges | None
    _active_pts: npt.NDArray[np.intp]
    _inex_x_dim: tuple[int, ...]
    _inex_x_ind: tuple[int, ...]
    _inex_x_val_ind: tuple[npt.NDArray[np.intp], ...]
    _inex_x_obs_ind: tuple[npt.NDArray[np.intp], ...]
    _x_buf: npt.NDArray
    _y_buf: npt.NDArray
    _var_buffer: npt.NDArray
    _cov: npt.NDArray | None
    _factor: tuple[npt.NDArray, npt.NDArray] | None
    _err: npt.NDArray
    _residual: npt.NDArray
    _x_residual: npt.NDArray
    _y_residual: npt.NDArray
    _work: npt.NDArray | None

    def __init__(
        self,
        data: XYData,
        model: Model,
        *,
        include: PointRanges | None = None,
        exclude: PointRanges | None = None,
        covariance: Literal["full", "diagonal"] = "full",
    ) -> None:
        """Create a chi-squared function for correlated x/y data.

        Args:
            data: Mapped x/y data with a joint covariance for stochastic
                coordinates.
            model: Model receiving x with shape `(n_selected_points, n_var)`.
            include: One closed `(low, high)` value interval per x coordinate.
                Finite endpoints are included; `None` is unbounded.
            exclude: Closed coordinate intervals removed after `include`.
            covariance: Use `"full"` for the full covariance matrix or
                `"diagonal"` to ignore every covariance correlation.
        """
        if model.n_var != data.x_ndim:
            raise ValueError(
                f"model has {model.n_var} variables, expected {data.x_ndim}"
            )
        if covariance not in ("full", "diagonal"):
            raise ValueError('covariance must be "full" or "diagonal"')

        self._xydata = data
        self._model = model
        self._n_par = model.n_par
        self._covariance_mode = covariance
        self._include = tuple(include) if include is not None else None
        self._exclude = tuple(exclude) if exclude is not None else None
        point_mask = data.point_mask(self._include, self._exclude)
        self._active_pts = np.flatnonzero(point_mask)
        n_points = self._active_pts.size

        # model variable buffer with pre-filled exact x values
        self._var_buffer = np.empty((n_points, data.x_ndim))
        for i in range(data.x_ndim):
            if data.is_exact_x(i):
                self._var_buffer[:, i] = data.x(i)[self._active_pts]

        # inexact x dimension lookup tables
        inex_x_dim: list[int] = []
        inex_x_ind: list[int] = []
        inex_x_val_ind: list[npt.NDArray[np.intp]] = []
        inex_x_obs_ind: list[npt.NDArray[np.intp]] = []
        for i in range(data.x_ndim):
            if data.is_exact_x(i):
                continue
            quantity = data.x_indices[i]
            assert quantity is not None
            active_map = data.x_map(i)[self._active_pts]
            val, obs_ind = np.unique(active_map, return_inverse=True)
            inex_x_dim.append(i)
            inex_x_ind.append(quantity)
            inex_x_val_ind.append(val)
            inex_x_obs_ind.append(obs_ind)
        self._inex_x_dim = tuple(inex_x_dim)  # dimension indices
        self._inex_x_ind = tuple(inex_x_ind)  # dimension indices in data
        self._inex_x_val_ind = tuple(inex_x_val_ind)  # unique active x value indices
        self._inex_x_obs_ind = tuple(inex_x_obs_ind)  # pt index -> index in x_val_ind

        # assemble selected stochastic x and y means and covariance blocks
        quantities = [*self._inex_x_ind, *data.y_indices]
        active = [  # full data selection indices
            *self._inex_x_val_ind,
            *([self._active_pts] * data.y_ndim),
        ]
        active_cov: npt.NDArray | None = None
        if covariance == "full":
            mean, active_cov = data.data.total_mean_cov(
                indices=active,
                quantities=quantities,
            )
            assert active_cov is not None
            self._err = np.sqrt(active_cov.diagonal())
        else:
            mean = np.concatenate([
                data.data.mean(quantity)[selection]
                for quantity, selection in zip(quantities, active)
            ])
            self._err = np.concatenate([
                np.sqrt(data.data.cov(quantity, quantity).diagonal())[selection]
                for quantity, selection in zip(quantities, active)
            ])

        # cache data and reusable buffers for chi-squared evaluations
        n_x = sum(indices.size for indices in self._inex_x_val_ind)
        self._x_buf = mean[:n_x]
        self._y_buf = mean[n_x:].reshape(data.y_ndim, n_points).T
        if covariance == "full":
            assert active_cov is not None
            self._cov = active_cov
            self._factor = cov_factor(active_cov)
            self._work = np.empty(active_cov.shape[0])
        else:
            if np.any(self._err <= 0.0):
                raise ValueError("diagonal covariance entries must be positive")
            self._cov = None
            self._factor = None
            self._work = None
        self._residual = np.empty(mean.size)
        self._reset_residual_views()

    # recreate views into the residual buffer after process serialization
    def _reset_residual_views(self) -> None:
        n_x = self._x_buf.size
        self._x_residual = self._residual[:n_x]
        self._y_residual = (
            self._residual[n_x:].reshape(self._xydata.y_ndim, self.n_points).T
        )

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._reset_residual_views()

    def set_means(self, means: Sequence[npt.NDArray]) -> None:
        """Update stochastic means while retaining the selected covariance."""
        self._xydata.data.set_means(list(means))

        # refresh selected means while retaining the covariance factor
        offset = 0
        for quantity, indices in zip(self._inex_x_ind, self._inex_x_val_ind):
            size = indices.size
            self._x_buf[offset : offset + size] = self._xydata.data.mean(quantity)[
                indices
            ]
            offset += size
        for i, yi in enumerate(self._xydata.y_indices):
            self._y_buf[:, i] = self._xydata.data.mean(yi)[self._active_pts]

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of points included in this chi-squared function."""
        mask = np.zeros(self._xydata.n_points, dtype=bool)
        mask[self._active_pts] = True
        return mask

    @property
    def n_points(self) -> int:
        """Number of selected y observation points."""
        return self._active_pts.size

    @property
    def n_parameters(self) -> int:
        """Total number of physical and latent-x fit parameters."""
        return self._n_par + self._x_buf.size

    @property
    def dof(self) -> int:
        """Degrees of freedom including the latent x coordinates."""
        return self._y_buf.size - self._n_par

    def full_parameters(self, model_parameters: npt.NDArray) -> npt.NDArray:
        """Append observed selected stochastic x values to model parameters."""
        model_parameters = np.asarray(model_parameters, dtype=float)
        if model_parameters.ndim != 1:
            raise ValueError("model_parameters must be one-dimensional")
        if model_parameters.size != self._n_par:
            raise ValueError(
                f"model_parameters has {model_parameters.size} entries, "
                f"expected {self._n_par}"
            )
        return np.concatenate((model_parameters, self._x_buf))

    def uncorrelated(self, exact_x: bool = False) -> "Chi2":
        """Return an uncorrelated chi-squared function for selected points."""
        if not exact_x:
            return Chi2(
                self._xydata,
                self._model,
                include=self._include,
                exclude=self._exclude,
                covariance="diagonal",
            )

        x = [
            self._xydata.x(i)[self._active_pts].copy()
            for i in range(self._xydata.x_ndim)
        ]
        means = [self._y_buf[:, i].copy() for i in range(self._xydata.y_ndim)]
        covs = [
            [
                self._xydata.data.cov(i, j)[np.ix_(self._active_pts, self._active_pts)]
                for j in self._xydata.y_indices[row:]
            ]
            for row, i in enumerate(self._xydata.y_indices)
        ]
        return Chi2(
            XYData(
                CorrelatedData(means, covs),
                x=x,
                y_indices=tuple(range(self._xydata.y_ndim)),
                x_names=self._xydata.x_names,
                y_names=self._xydata.y_names,
            ),
            self._model,
            covariance="diagonal",
        )

    def _set_residual(self, parameters: npt.NDArray) -> None:
        if parameters.ndim != 1:
            raise ValueError("parameters must be one-dimensional")
        if parameters.size != self.n_parameters:
            raise ValueError(
                f"parameters has {parameters.size} entries, "
                f"expected {self.n_parameters}"
            )

        # split physical model parameters from compact latent x values
        model_parameters = parameters[: self._n_par]
        latent_x = parameters[self._n_par :]

        # expand each compact latent coordinate to the selected observations
        offset = 0
        for x_index, (dimension, indices) in enumerate(
            zip(self._inex_x_dim, self._inex_x_obs_ind)
        ):
            size = self._inex_x_val_ind[x_index].size
            self._var_buffer[:, dimension] = latent_x[offset : offset + size][indices]
            offset += size

        # evaluate the model at exact and latent x values
        prediction = self._model._function(self._var_buffer, model_parameters)

        # accept a one-dimensional result for one y coordinate
        if self._y_buf.shape[1] == 1 and prediction.shape == (self.n_points,):
            prediction = prediction[:, np.newaxis]

        # validate the model output before writing the residual buffers
        if prediction.shape != self._y_buf.shape:
            raise ValueError(
                f"model returned shape {prediction.shape}, expected {self._y_buf.shape}"
            )

        # assemble the x-minus-latent and y-minus-model residuals
        np.subtract(self._x_buf, latent_x, out=self._x_residual)
        np.subtract(self._y_buf, prediction, out=self._y_residual)

    def residual(self, parameters: npt.NDArray) -> npt.NDArray:
        """Return statistically independent residuals for `parameters`."""
        self._set_residual(parameters)
        if self._covariance_mode == "diagonal":
            return self._residual / self._err
        assert self._cov is not None
        assert self._factor is not None
        assert self._work is not None
        cov_independent_residuals(self._residual, self._cov, self._factor, self._work)
        return self._work.copy()

    def __call__(self, parameters: npt.NDArray) -> float:
        """Evaluate chi-squared for physical and latent-x parameters."""
        self._set_residual(parameters)
        if self._covariance_mode == "diagonal":
            standardized = self._residual / self._err
            return float(standardized @ standardized)
        assert self._cov is not None
        assert self._factor is not None
        assert self._work is not None
        return float(
            cov_quadratic_form(self._residual, self._cov, self._factor, self._work)
        )
