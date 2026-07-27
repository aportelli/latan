from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from latan.statistics.bootstrap import BootstrapArray

type ModelFunction = Callable[[npt.NDArray, npt.NDArray], npt.NDArray]


class Model:
    """A dimension-checked model function with broadcast batch axes.

    The wrapped function receives arrays whose final axes contain variables
    and parameters. All preceding axes are evaluation or sample dimensions.

    Example:
        ```python
        model = Model(
            lambda x, p: p[..., 0] + p[..., 1] * x[..., 0],
            1,
            2,
            parameter_names=("intercept", "slope"),
        )
        y = model(np.array([[0.0], [1.0]]), np.array([1.0, 2.0]))
        ```
    """

    _function: ModelFunction
    _n_var: int
    _n_par: int
    _parameter_names: tuple[str, ...]

    def __init__(
        self,
        function: ModelFunction,
        n_var: int,
        n_par: int,
        *,
        parameter_names: Sequence[str] | None = None,
    ) -> None:
        """Create a model with fixed variable and parameter dimensions.

        Args:
            function: A callable `function(x, p)` returning a NumPy array.
            n_var: Number of variables stored on the final x axis.
            n_par: Number of parameters stored on the final p axis.
            parameter_names: Optional display names for model parameters.
        """
        if n_var < 1:
            raise ValueError("n_var must be positive")
        if n_par < 0:
            raise ValueError("n_par must not be negative")
        self._function = function
        self._n_var = n_var
        self._n_par = n_par
        if parameter_names is None:
            self._parameter_names = tuple(f"p_{i}" for i in range(n_par))
        else:
            self._parameter_names = tuple(parameter_names)
            if len(self._parameter_names) != n_par:
                raise ValueError(
                    f"parameter_names must have length {n_par}, "
                    f"got {len(self._parameter_names)}"
                )
            if not all(isinstance(name, str) for name in self._parameter_names):
                raise TypeError("parameter_names must contain only strings")

    @property
    def n_var(self) -> int:
        """Number of variables expected on the final x axis."""
        return self._n_var

    @property
    def n_par(self) -> int:
        """Number of parameters expected on the final p axis."""
        return self._n_par

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Display names for model parameters."""
        return self._parameter_names

    def __call__(self, x: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
        """Validate dimensions and forward x and p to the wrapped function.

        The x and parameter arrays must have the same batch shape. x may have
        one additional point axis immediately before its final variable axis;
        parameters are then broadcast over that point axis.
        """
        if x.ndim < 1 or x.shape[-1] != self._n_var:
            raise ValueError(f"x must have shape (..., {self._n_var})")
        if p.ndim < 1 or p.shape[-1] != self._n_par:
            raise ValueError(f"p must have shape (..., {self._n_par})")
        x_batch = x.shape[:-1]
        p_batch = p.shape[:-1]
        if x_batch == p_batch:
            pass
        elif x_batch[:-1] == p_batch:
            p = p.reshape(
                *p_batch,
                1,
                p.shape[-1],
            )
        else:
            raise ValueError("x and p must have the same batch shape")
        return self._function(x, p)

    def bootstrap(
        self,
        x: npt.NDArray | BootstrapArray,
        p: npt.NDArray | BootstrapArray,
    ) -> BootstrapArray:
        """Evaluate the model for vectorized bootstrap samples.

        At least one argument must be a `BootstrapArray`. An ordinary array is
        broadcast over bootstrap samples. If both arguments are bootstrap
        arrays, they must have the same number of samples. The wrapped model
        must support vectorized batch dimensions.
        """
        if not isinstance(x, BootstrapArray) and not isinstance(p, BootstrapArray):
            raise TypeError("x or p must be a BootstrapArray")

        n_bootstrap = x.shape[0] if isinstance(x, BootstrapArray) else p.shape[0]
        if (
            isinstance(x, BootstrapArray)
            and isinstance(p, BootstrapArray)
            and x.shape[0] != p.shape[0]
        ):
            raise ValueError("bootstrap x and p must have the same sample count")

        if isinstance(x, BootstrapArray) and not isinstance(p, BootstrapArray):
            p = np.broadcast_to(p, (n_bootstrap, *p.shape))
        elif not isinstance(x, BootstrapArray) and isinstance(p, BootstrapArray):
            x = np.broadcast_to(x, (n_bootstrap, *x.shape))
        result = np.asarray(self(x, p))
        if result.ndim == 0 or result.shape[0] != n_bootstrap:
            raise ValueError(
                "vectorized model output must have the bootstrap axis first"
            )
        if result.ndim == 1:
            result = result[:, None]
        return BootstrapArray(result)
