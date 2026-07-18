from typing import Sequence, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

from latan.statistics.correlation import (
    corr_factor,
    corr_quadratic_form,
)
from latan.statistics.model import Model
from latan.statistics.xy_data import XYData

PointRanges: TypeAlias = Sequence[Tuple[float | None, float | None]]


class Chi2:
    """Chi-squared function for given x/y data and model.

    Inexact x coordinates are fitted as latent values. The parameter vector
    starts with the model parameters, followed by one latent value for every
    selected point of each inexact x coordinate.

    Example:
        ```python
        model = Model(lambda x, p: p[..., 0] + p[..., 1] * x[..., 0], 1, 2)

        chi2 = Chi2(data, model)
        parameters = chi2.full_parameters(np.array([0.0, 1.0]))
        value = chi2(parameters)
        ```
    """

    _xydata: XYData
    _model: Model
    _n_par: int
    _mask: npt.NDArray[np.bool_]
    _point_indices: npt.NDArray[np.intp]
    _inexact_x_dim: Tuple[int, ...]
    _x_indices: Tuple[int, ...]
    _x_buf: npt.NDArray
    _y_buf: npt.NDArray
    _var_buffer: npt.NDArray
    _err: npt.NDArray
    _factor: npt.NDArray
    _residual: npt.NDArray
    _x_residual: npt.NDArray
    _y_residual: npt.NDArray
    _work: npt.NDArray

    def __init__(
        self,
        data: XYData,
        model: Model,
        *,
        include: PointRanges | None = None,
        exclude: PointRanges | None = None,
    ) -> None:
        """Create a chi-squared function for correlated x/y data.

        Args:
            data: Pointwise x/y data with a joint covariance for all
                stochastic coordinates.
            model: A `Model` whose variable dimension matches the x data.
                It receives x with shape `(n_selected_points, n_var)` and
                physical model parameters. It must return shape
                `(n_selected_points, n_y)`, or `(n_selected_points,)` when
                there is one y coordinate.
            include: Closed x-coordinate ranges defining initially retained
                points. `None` selects every point.
            exclude: Closed x-coordinate ranges removed after `include`.
        """
        if model.n_var != data.x_ndim:
            raise ValueError(
                f"model has {model.n_var} variables, expected {data.x_ndim}"
            )

        self._xydata = data
        self._model = model
        self._n_par = model.n_par
        self._mask = data.point_mask(include, exclude)
        self._point_indices = np.flatnonzero(self._mask)
        n_points = self._point_indices.size

        # cache inexact x dimensions
        self._inexact_x_dim = tuple(
            i for i in range(data.x_ndim) if not data.is_exact_x(i)
        )

        # buffer for model x variables
        self._var_buffer = np.empty((n_points, data.x_ndim))
        for i in range(data.x_ndim):
            if data.is_exact_x(i):
                self._var_buffer[:, i] = data.x(i)[self._mask]

        # cache indices in X/Y data of inexact x dimensions
        self._x_indices = tuple(index for index in data.x_indices if index is not None)

        # cache mean and covariance from data for selected fit points
        quantities = list(self._x_indices) + list(data.y_indices)
        mean, covariance = data.data.total_mean_cov(
            indices=[self._point_indices] * (data.x_inexact_ndim + data.y_ndim),
            quantities=quantities,
        )

        n_x = data.x_inexact_ndim * n_points
        self._x_buf = mean[:n_x].reshape(data.x_inexact_ndim, n_points).T
        self._y_buf = mean[n_x:].reshape(data.y_ndim, n_points).T
        self._err, self._factor = corr_factor(covariance)
        self._residual = np.empty(covariance.shape[0])
        self._x_residual = self._residual[:n_x].reshape(data.x_inexact_ndim, n_points).T
        self._y_residual = self._residual[n_x:].reshape(data.y_ndim, n_points).T
        self._work = np.empty_like(self._residual)

    def set_means(self, means: Sequence[npt.NDArray]) -> None:
        """Update stochastic means while retaining the selected covariance.

        Args:
            means: One mean vector per stochastic quantity in the wrapped
                `CorrelatedData`, in its stored order.
        """
        self._xydata.data.set_means(list(means))
        for i, xi in enumerate(self._x_indices):
            self._x_buf[:, i] = self._xydata.data.mean(xi)[self._point_indices]
        for i, yi in enumerate(self._xydata.y_indices):
            self._y_buf[:, i] = self._xydata.data.mean(yi)[self._point_indices]

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of the points included in this chi-squared function."""
        return self._mask.copy()

    @property
    def n_points(self) -> int:
        """Number of selected points."""
        return self._point_indices.size

    @property
    def n_parameters(self) -> int:
        """Total number of physical and latent-x fit parameters."""
        return self._n_par + self._x_buf.size

    @property
    def dof(self) -> int:
        """Degrees of freedom including the latent x coordinates."""
        return self._y_buf.size - self._n_par

    def full_parameters(self, model_parameters: npt.NDArray) -> npt.NDArray:
        """Append the observed inexact x values to model parameters.

        Args:
            model_parameters: One-dimensional array with
                `model.n_par` entries.
        """
        model_parameters = np.asarray(model_parameters, dtype=float)
        if model_parameters.ndim != 1:
            raise ValueError("model_parameters must be one-dimensional")
        if model_parameters.size != self._n_par:
            raise ValueError(
                f"model_parameters has {model_parameters.size} entries, "
                f"expected {self._n_par}"
            )
        return np.concatenate((model_parameters, self._x_buf.ravel()))

    def __call__(self, parameters: npt.NDArray) -> float:
        """Evaluate chi-squared for physical and latent-x parameters."""
        if parameters.ndim != 1:
            raise ValueError("parameters must be one-dimensional")
        if parameters.size != self.n_parameters:
            raise ValueError(
                f"parameters has {parameters.size} entries, "
                f"expected {self.n_parameters}"
            )

        # separate parameters in model & latent parts
        model_parameters = parameters[: self._n_par]
        latent_x = parameters[self._n_par :].reshape(self._x_buf.shape)

        # evaluate model
        for source, target in enumerate(self._inexact_x_dim):
            self._var_buffer[:, target] = latent_x[:, source]
        prediction = self._model._function(self._var_buffer, model_parameters)

        # scalar model support
        if self._y_buf.shape[1] == 1 and prediction.shape == (self.n_points,):
            prediction = prediction[:, np.newaxis]

        # validate model output shape
        if prediction.shape != self._y_buf.shape:
            raise ValueError(
                f"model returned shape {prediction.shape}, expected {self._y_buf.shape}"
            )

        # compute chi^2
        np.subtract(self._x_buf, latent_x, out=self._x_residual)
        np.subtract(self._y_buf, prediction, out=self._y_residual)
        return corr_quadratic_form(self._residual, self._err, self._factor, self._work)
