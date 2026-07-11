import unittest

import numpy as np

import latan


class TestLaplaceFilter(unittest.TestCase):
    def test_laplace_filter_single(self) -> None:
        data = np.random.default_rng(0).normal(size=(3, 4, 5))
        lamb = 0.7
        for dim in range(data.ndim):
            with self.subTest(dim=dim):
                expected = (
                    (2 + lamb**2) * data
                    - np.roll(data, 1, axis=dim)
                    - np.roll(data, -1, axis=dim)
                )
                np.testing.assert_allclose(latan.lfilter(data, lamb, dim), expected)

    def test_laplace_filter_multi(self) -> None:
        data = np.random.default_rng(1).normal(size=(3, 4, 5))
        lambs = np.array([0.7, 0.2])
        dims = (0, 2)
        expected = data.copy()
        for lamb in lambs:
            for dim in dims:
                expected = (
                    (2 + lamb**2) * expected
                    - np.roll(expected, 1, axis=dim)
                    - np.roll(expected, -1, axis=dim)
                )
        out = np.empty_like(data)
        result = latan.lfilter(data, lambs, dims, out=out)
        self.assertIs(result, out)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(latan.lfilter(data, [], dims), data)
