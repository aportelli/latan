from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import numpy as np
import numpy.typing as npt

from latan.statistics.gaussian_rng import gaussian_sample


class Bootstrap(ABC):
    _bitgen: np.random.BitGenerator
    _gen: np.random.Generator

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self._bitgen = np.random.PCG64(seed)
        self._gen = np.random.Generator(self._bitgen)

    @property
    def state(self) -> Mapping[str, Any]:
        return self._bitgen.state

    @state.setter
    def state(self, value: Mapping[str, Any]) -> None:
        self._bitgen.state = value

    @abstractmethod
    def sample(self, data: npt.NDArray, size: int) -> npt.NDArray:
        pass


class ParametricGaussianBootstrap(Bootstrap):
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(seed)

    def sample(self, data: npt.NDArray, size: int) -> npt.NDArray:
        s = np.zeros([size + 1, *data.shape[1:]])
        n = data.shape[0]
        mean = data.mean(axis=0)
        var = np.cov(data, rowvar=False) / n
        s[0] = mean
        s[1:] = gaussian_sample(self._gen, size, mean, var)
        return s


class NonparametricBootstrap(Bootstrap):
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(seed)

    def sample(self, data: npt.NDArray, size: int) -> npt.NDArray:
        s = np.zeros([size + 1, *data.shape[1:]])
        n = data.shape[0]
        mean = data.mean(axis=0)
        ind = self._gen.integers(n, size=(n, size))
        data_boot = data[ind, ...]
        s[0] = mean
        s[1:] = data_boot.mean(axis=0)
        return s
