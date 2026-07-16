import unittest

import numpy as np

import latan
from latan.statistics.correlated_data import CorrelatedData


class TestLaplaceFilter(unittest.TestCase):
    @staticmethod
    def _reference_t2(mean: np.ndarray, cov: np.ndarray) -> float:
        err = np.sqrt(cov.diagonal())
        corr = cov / np.outer(err, err)
        mean_norm = mean / err
        return (mean_norm @ np.linalg.solve(corr, mean_norm)).item()

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

    def test_laplace_filter_t2_single(self) -> None:
        rng = np.random.default_rng(2)
        mean = rng.normal(size=5)
        matrix = rng.normal(size=(5, 5))
        cov = matrix @ matrix.T + np.eye(5)
        lamb = np.array([0.4, 0.7])
        data = CorrelatedData([mean], [[cov]])
        t2 = latan.LaplaceFilteredT2(data, [(1, 5)])

        mean_filtered = latan.lfilter(mean, lamb)[1:5]
        cov_filtered = latan.lfilter(cov, lamb, dim=(0, 1))[1:5, 1:5]

        self.assertEqual(t2.ranges, ((1, 5),))
        self.assertAlmostEqual(
            t2(lamb), self._reference_t2(mean_filtered, cov_filtered)
        )

    def test_laplace_filter_t2_combined(self) -> None:
        rng = np.random.default_rng(3)
        mean_a = rng.normal(size=4)
        mean_b = rng.normal(size=3)
        matrix = rng.normal(size=(7, 7))
        cov_total = matrix @ matrix.T + np.eye(7)
        cov_aa = cov_total[:4, :4]
        cov_ab = cov_total[:4, 4:]
        cov_bb = cov_total[4:, 4:]
        data = CorrelatedData([mean_a, mean_b], [[cov_aa, cov_ab], [cov_bb]])
        ranges = [(1, 4), (0, 2)]
        lamb = np.array([0.3, 0.6])
        t2 = latan.LaplaceFilteredT2(data, ranges)

        mean_expected = np.concatenate([
            latan.lfilter(mean_a, lamb)[1:4],
            latan.lfilter(mean_b, lamb)[:2],
        ])
        cov_expected = np.block([
            [
                latan.lfilter(cov_aa, lamb, dim=(0, 1))[1:4, 1:4],
                latan.lfilter(cov_ab, lamb, dim=(0, 1))[1:4, :2],
            ],
            [
                latan.lfilter(cov_ab.T, lamb, dim=(0, 1))[:2, 1:4],
                latan.lfilter(cov_bb, lamb, dim=(0, 1))[:2, :2],
            ],
        ])

        selected_mean, selected_cov = data.total_mean_cov(ranges)
        self.assertEqual(selected_mean.shape, (5,))
        self.assertEqual(selected_cov.shape, (5, 5))
        np.testing.assert_allclose(data.cov(1, 0), cov_ab.T)
        np.testing.assert_allclose(selected_cov[:3, 3:], cov_ab[1:4, :2])
        self.assertAlmostEqual(
            t2(lamb), self._reference_t2(mean_expected, cov_expected)
        )
