from typing import overload

import numpy as np
import numpy.typing as npt

from latan.statistics.bootstrap import BootstrapArray


@overload
def eff_mass_cosh(c: BootstrapArray) -> BootstrapArray: ...


@overload
def eff_mass_cosh(c: npt.NDArray) -> npt.NDArray: ...


def eff_mass_cosh(c: npt.NDArray | BootstrapArray) -> npt.NDArray | BootstrapArray:
    return np.acosh((c[..., 0:-2] + c[..., 2:]) / (2.0 * c[..., 1:-1]))


@overload
def eff_mass_log(c: BootstrapArray) -> BootstrapArray: ...


@overload
def eff_mass_log(c: npt.NDArray) -> npt.NDArray: ...


def eff_mass_log(c: npt.NDArray | BootstrapArray) -> npt.NDArray | BootstrapArray:
    return np.log(c[..., 0:-1] / c[..., 1:])
