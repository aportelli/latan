import numpy as np
import numpy.typing as npt

from latan.statistics.correlation import var_to_corr


# stabler version of the NumPy multivariate Gaussian RNG, using the correlation matrix
# for the sampling instead of the covariance matrix
def gaussian_sample(
    rng: np.random.Generator, size: int, mean: npt.NDArray, var: npt.NDArray
):
    corr, err = var_to_corr(var)
    sample = rng.multivariate_normal(mean / err, corr, size)
    sample *= err
    return sample
