from typing import Tuple

import numpy as np
import numpy.typing as npt


def cdr(mat: npt.NDArray) -> float:
    s = np.linalg.svd(mat, compute_uv=False)
    return 10.0 * np.log10(s.max() / s.min())


def var_to_corr(var: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    err = np.sqrt(var.diagonal())
    inverr = 1.0 / err
    corr = var * np.outer(inverr, inverr)
    return corr, err


def corr_to_var(corr: npt.NDArray, err: npt.NDArray) -> npt.NDArray:
    return corr * np.outer(err, err)
