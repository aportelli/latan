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
def eff_mass_cosh_correction(
    c: BootstrapArray, dc: BootstrapArray
) -> BootstrapArray: ...


@overload
def eff_mass_cosh_correction(c: npt.NDArray, dc: npt.NDArray) -> npt.NDArray: ...


def eff_mass_cosh_correction(
    c: npt.NDArray | BootstrapArray, dc: npt.NDArray | BootstrapArray
) -> npt.NDArray | BootstrapArray:
    c_ratio = (c[..., 0:-2] + c[..., 2:]) / (2.0 * c[..., 1:-1])
    dc_ratio = (dc[..., 0:-2] + dc[..., 2:]) / (2.0 * c[..., 1:-1])
    return (dc_ratio - dc[..., 1:-1] * c_ratio / c[..., 1:-1]) / np.sqrt(
        c_ratio**2 - 1.0
    )


@overload
def eff_mass_log(c: BootstrapArray) -> BootstrapArray: ...


@overload
def eff_mass_log(c: npt.NDArray) -> npt.NDArray: ...


def eff_mass_log(c: npt.NDArray | BootstrapArray) -> npt.NDArray | BootstrapArray:
    return np.log(c[..., 0:-1] / c[..., 1:])
