from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt

from latan.statistics.gaussian_rng import gaussian_sample


class BootstrapArray(np.ndarray):
    def __new__(cls, array: npt.ArrayLike):
        array = np.asarray(array)
        if array.ndim < 2 or array.shape[0] < 2:
            raise ValueError("expected shape (n_bootstrap + 1, ...)")
        return array.view(cls)

    def __array_finalize__(self, obj) -> None:
        pass

    @property
    def central(self) -> npt.NDArray:
        return np.asarray(self[0])

    @property
    def samples(self) -> npt.NDArray:
        return np.asarray(self[1:])

    def cov(self) -> npt.NDArray:
        return np.cov(self.samples, rowvar=False)

    def error(self) -> npt.NDArray:
        return np.std(self.samples, axis=0)


class Bootstrap(ABC):
    _bitgen: np.random.BitGenerator
    _gen: np.random.Generator
    _initial_state: Mapping[str, Any]

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._bitgen = np.random.PCG64(seed)
        self._gen = np.random.Generator(self._bitgen)
        self._initial_state = deepcopy(self.state)

    def _reset_state(self) -> None:
        self._bitgen.state = deepcopy(self._initial_state)

    @property
    def state(self) -> Mapping[str, Any]:
        return self._bitgen.state

    @state.setter
    def state(self, value: Mapping[str, Any]) -> None:
        self._bitgen.state = value

    @abstractmethod
    def sample(self, data: npt.NDArray, size: int) -> BootstrapArray:
        pass


class ParametricGaussianBootstrap(Bootstrap):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)

    def sample(self, data: npt.NDArray, size: int) -> BootstrapArray:
        self._reset_state()
        s = np.zeros([size + 1, *data.shape[1:]])
        n = data.shape[0]
        mean = data.mean(axis=0)
        var = np.cov(data, rowvar=False) / n
        s[0] = mean
        s[1:] = gaussian_sample(self._gen, size, mean, var)
        return BootstrapArray(s)


class NonparametricBootstrap(Bootstrap):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)

    def sample(self, data: npt.NDArray, size: int) -> BootstrapArray:
        self._reset_state()
        s = np.zeros([size + 1, *data.shape[1:]])
        n = data.shape[0]
        mean = data.mean(axis=0)
        ind = self._gen.integers(n, size=(n, size))
        data_boot = data[ind, ...]
        s[0] = mean
        s[1:] = data_boot.mean(axis=0)
        return BootstrapArray(s)
