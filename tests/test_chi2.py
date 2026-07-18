import unittest

import numpy as np

import latan


class TestChi2(unittest.TestCase):
    def test_chi2(self) -> None:
        x = np.array([-1.0, -0.2, 0.4, 1.1])
        y = np.array([-0.8, 0.3, 1.7, 2.9])
        matrix = np.random.default_rng(123).normal(size=(2 * x.size, 2 * x.size))
        covariance = matrix @ matrix.T + np.eye(2 * x.size)
        data = latan.XYData(
            latan.CorrelatedData(
                [x, y],
                [[covariance[:4, :4], covariance[:4, 4:]], [covariance[4:, 4:]]],
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
        chi2 = latan.Chi2(data, model)

        parameters = chi2.full_parameters(np.array([0.5, 2.0]))
        parameters[2:] += np.array([0.1, -0.1, 0.2, -0.2])

        latent_x = parameters[2:]
        residual = np.concatenate([
            x - latent_x,
            y - model(latent_x[:, None], parameters[:2]),
        ])
        expected = float(residual @ np.linalg.solve(covariance, residual))

        self.assertAlmostEqual(chi2(parameters), expected)
        independent_residuals = chi2.residual(parameters)
        self.assertAlmostEqual(float(independent_residuals @ independent_residuals), expected)
