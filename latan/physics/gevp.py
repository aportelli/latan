from typing import overload

import numpy as np
import numpy.typing as npt
from scipy import linalg

from latan.statistics.bootstrap import BootstrapArray


@overload
def gevp(matrix: BootstrapArray, t0: int = 0) -> BootstrapArray: ...


@overload
def gevp(matrix: npt.NDArray, t0: int = 0) -> npt.NDArray: ...


def gevp(
    matrix: npt.NDArray | BootstrapArray, t0: int = 0
) -> npt.NDArray | BootstrapArray:
    """Solve a Hermitian generalized eigenvalue problem at every time.

    `matrix` has shape `(..., n_op, n_op, nt)`. The result has shape
    `(..., n_op, nt)`. All leading axes are passed to SciPy's batched
    Hermitian generalized eigensolver. A `BootstrapArray` is preserved.
    Eigenvalues are in descending order, so the first level has the slowest
    decay for Euclidean correlators. NB: the matrix will be forced to be Hermitian, and
    there is no hermiticity check.
    """
    matrices = np.moveaxis(matrix, -1, -3)
    matrices = 0.5 * (matrices + np.swapaxes(matrices, -1, -2).conj())
    reference = np.broadcast_to(
        matrices[..., t0, :, :][..., None, :, :], matrices.shape
    )
    values = linalg.eigvalsh(matrices, reference, check_finite=False)[..., ::-1]
    values = np.moveaxis(values, -2, -1)
    return BootstrapArray(values) if isinstance(matrix, BootstrapArray) else values
