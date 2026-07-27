from typing import Any

import numpy as np
import numpy.typing as npt
from iminuit import Minuit

from latan.physics.laplace_filter.filter import lfilter
from latan.statistics.correlation import cdr


class _CdrCost:
    _var: npt.NDArray
    _varf: npt.NDArray
    _range: tuple[int, int] | None

    def __init__(self, var: npt.NDArray, range: tuple[int, int] | None = None) -> None:
        self._var = var
        self._varf = np.zeros(var.shape)
        self._range = range

    def __call__(self, *args: Any) -> float:
        (lamb,) = args
        lfilter(self._var, lamb, dim=(0, 1), out=self._varf)
        if self._range is None:
            return cdr(self._varf)
        else:
            r = self._range
            return cdr(self._varf[r[0] : r[1], r[0] : r[1]])


def lfilter_optimize_cdr(
    cov: npt.NDArray, range: tuple[int, int] | None = None, *, init_lambda: float = 5.0
) -> float:
    cost = _CdrCost(cov, range)
    m = Minuit(cost, init_lambda, name=("lambda",))
    m.limits["lambda"] = (0.0, None)
    m.simplex()
    result = m.migrad()
    assert result.fmin is not None
    if not result.fmin.is_valid:
        print("warning: invalid minimum")
    return result.values["lambda"]
