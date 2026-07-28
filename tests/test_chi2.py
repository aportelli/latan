import unittest

import numpy as np

import latan


class TestChi2(unittest.TestCase):
    def test_chi2(self) -> None:
        t = np.array([0.0, 1.0, 2.0, 3.0])
        x = np.array([-1.0, -0.2, 0.4, 1.1])
        y = np.array([-0.8, 0.3, 1.7, 2.9])
        matrix = np.random.default_rng(123).normal(size=(2 * x.size, 2 * x.size))
        covariance = matrix @ matrix.T + np.eye(2 * x.size)
        data = latan.XYData(
            latan.CorrelatedData(
                [x, y],
                [[covariance[:4, :4], covariance[:4, 4:]], [covariance[4:, 4:]]],
            ),
            x=[t, 0],
            y_indices=[1],
        )
        model = latan.Model(
            lambda values, parameters: (
                parameters[..., 0]
                + parameters[..., 1] * values[..., 0]
                + parameters[..., 2] * values[..., 1]
            ),
            n_var=2,
            n_par=3,
        )
        chi2 = latan.Chi2(data, model)
        parameters = chi2.full_parameters(np.array([0.5, 2.0, -1.0]))
        parameters[3:] += np.array([0.1, -0.1, 0.2, -0.2])
        latent_x = parameters[3:]
        residual = np.concatenate([
            x - latent_x,
            y - model(np.column_stack((t, latent_x)), parameters[:3]),
        ])
        expected = float(residual @ np.linalg.solve(covariance, residual))
        self.assertAlmostEqual(chi2(parameters), expected)
        independent_residuals = chi2.residual(parameters)
        self.assertAlmostEqual(
            float(independent_residuals @ independent_residuals), expected
        )
        uncorr = chi2.uncorrelated(exact_x=True)
        self.assertEqual(uncorr.n_parameters, model.n_par)
        uncorr_residual = y - model(np.column_stack((t, x)), parameters[:3])
        uncorr_expected = float(
            np.sum(uncorr_residual**2 / np.diag(covariance[4:, 4:]))
        )
        self.assertAlmostEqual(uncorr(parameters[:3]), uncorr_expected)

    def test_linear_fit(self) -> None:
        x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
        y = np.array([-1.1, 0.4, 0.9, 2.8, 5.4])
        matrix = np.random.default_rng(321).normal(size=(x.size, x.size))
        covariance = matrix @ matrix.T + np.eye(x.size)
        data = latan.XYData(latan.CorrelatedData(y, covariance), x=[x], y_indices=[0])
        model = latan.Model(
            lambda x, p: p[..., 0] + p[..., 1] * x[..., 0],
            n_var=1,
            n_par=2,
        )
        # exact solution of linear regression
        # p = (D^T * C^-1 * D)^-1 * D^T * C^-1 * y
        # with D design matrix and C covariance matrix
        design = np.column_stack((np.ones(x.size), x))
        inverse_design = np.linalg.solve(covariance, design)
        expected = np.linalg.solve(
            design.T @ inverse_design,
            design.T @ np.linalg.solve(covariance, y),
        )
        result = latan.fit(data, model, np.zeros(2))
        residual = y - design @ expected
        expected_chi2 = float(residual @ np.linalg.solve(covariance, residual))
        np.testing.assert_allclose(result.model_parameters, expected, atol=1e-8)
        self.assertAlmostEqual(result.chi2, expected_chi2)

    def test_fit_inexact_x(self) -> None:
        x = np.linspace(-2.0, 2.0, 6)
        parameters = np.array([1.2, -0.7])
        y = parameters[0] + parameters[1] * x
        covariance = 0.01 * np.eye(2 * x.size)
        data = latan.XYData(
            latan.CorrelatedData(
                [x, y],
                [[covariance[:6, :6], covariance[:6, 6:]], [covariance[6:, 6:]]],
            ),
            x=[0],
            y_indices=[1],
        )
        model = latan.Model(
            lambda values, parameters: (
                parameters[..., 0] + parameters[..., 1] * values[..., 0]
            ),
            n_var=1,
            n_par=2,
        )
        result = latan.fit(data, model, np.zeros(2))
        np.testing.assert_allclose(result.model_parameters, parameters, atol=1e-8)
        np.testing.assert_allclose(result.latent_parameters, x, atol=1e-8)
        self.assertAlmostEqual(result.chi2, 0.0)
