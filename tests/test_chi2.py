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
