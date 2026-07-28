import unittest

import numpy as np

import latan


class TestStatistics(unittest.TestCase):
    def test_ill_conditioned_covariance(self) -> None:
        # Badly conditioned covariance C and correlation R with exactly
        # known inverse through Sherman–Morrison formula.
        #
        # C = D * R * D
        # R = (1 - rho) * I + rho * 1 * 1^T, with D = diag(error)
        # C^-1 = D^-1 * R^-1 * D^-1
        # R^-1 = (I - rho * 1 * 1^T / (1 + (n - 1) * rho)) / (1 - rho)
        #
        # CDR(R) = 10 * log10((1 + (n - 1) * rho) / (1 - rho))
        #
        # 1 is a column vector with all components set to unity.
        #
        n = 4
        rho = 1.0 - 1e-8
        correlation = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
        error = np.array([1e-3, 1e-1, 1e1, 1e3])
        covariance = correlation * np.outer(error, error)
        residual = error * np.array([1.0, -0.5, 0.25, -0.75])
        inverse_correlation = (
            np.eye(n) - rho * np.ones((n, n)) / (1.0 + (n - 1) * rho)
        ) / (1.0 - rho)
        expected = inverse_correlation @ (residual / error) / error

        actual = latan.cov_inverse_multiply(residual, covariance)
        independent = latan.cov_independent_residuals(residual, covariance)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        np.testing.assert_allclose(
            independent @ independent, residual @ expected, rtol=1e-6
        )

    def test_covariance_algebra(self) -> None:
        rng = np.random.default_rng(456)
        matrix = rng.normal(size=(4, 4))
        covariance = matrix @ matrix.T + np.eye(4)
        residual = rng.normal(size=(3, 4))
        inverse_residual = np.linalg.solve(covariance, residual.T).T
        expected = np.sum(residual * inverse_residual, axis=-1)
        independent = latan.cov_independent_residuals(residual, covariance)
        np.testing.assert_allclose(
            latan.cov_inverse_multiply(residual, covariance), inverse_residual
        )
        np.testing.assert_allclose(np.sum(independent**2, axis=-1), expected)
        np.testing.assert_allclose(
            latan.cov_quadratic_form(residual, covariance), expected
        )

    def test_make_correlated_data(self) -> None:
        primary = [
            np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]]),
            np.array([[0.0], [2.0], [5.0]]),
        ]
        primary_data = latan.make_correlated_data(primary)
        primary_joint = np.concatenate(primary, axis=1)
        primary_covariance = np.cov(primary_joint, rowvar=False) / len(primary_joint)
        np.testing.assert_allclose(primary_data.mean(0), primary[0].mean(axis=0))
        np.testing.assert_allclose(primary_data.mean(1), primary[1].mean(axis=0))
        np.testing.assert_allclose(primary_data.total_mean_cov()[1], primary_covariance)
        bootstrap = [
            latan.BootstrapArray(
                np.array([[2.0, 3.0], [1.0, 3.0], [3.0, 1.0], [4.0, 5.0]])
            ),
            latan.BootstrapArray(np.array([[2.0], [0.0], [2.0], [5.0]])),
        ]
        bootstrap_data = latan.make_correlated_data(bootstrap)
        bootstrap_joint = np.concatenate([item.samples for item in bootstrap], axis=1)
        np.testing.assert_allclose(bootstrap_data.mean(0), bootstrap[0].central)
        np.testing.assert_allclose(bootstrap_data.mean(1), bootstrap[1].central)
        np.testing.assert_allclose(
            bootstrap_data.total_mean_cov()[1], np.cov(bootstrap_joint, rowvar=False)
        )
