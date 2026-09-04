from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from latan.display._common import (
    bootstrap_normality,
    bootstrap_value_html,
    non_gaussian_html,
    normality_css,
    p_value_colour,
)
from latan.statistics.bootstrap import BootstrapArray

if TYPE_CHECKING:
    from latan.physics.laplace_filter.amplitudes import LaplaceFilterAmplitudes
    from latan.physics.laplace_filter.spectrum import (
        LaplaceFilterEnergies,
    )


def _spectrum_normality_html(
    energy: float,
    energy_error: float,
    energy_lower: float,
    energy_upper: float,
    energy_non_gaussian: bool,
    energy_p_value: float,
    lamb: float,
    lambda_error: float,
    lambda_lower: float,
    lambda_upper: float,
    lambda_non_gaussian: bool,
    lambda_p_value: float,
) -> tuple[str, str]:
    errors = ()
    if energy_non_gaussian:
        errors += (("err_E", energy_error, energy_upper - energy, energy - energy_lower),)
    if lambda_non_gaussian:
        errors += (("err_λ", lambda_error, lambda_upper - lamb, lamb - lambda_lower),)
    return non_gaussian_html((energy_p_value, lambda_p_value), errors)


def render_laplace_filter_energies_html[T: npt.NDArray](
    result: "LaplaceFilterEnergies[T]",
) -> str:
    """Render a compact spectrum summary in Jupyter notebooks."""
    p_value = float(result.p_value)
    significance, colour = p_value_colour(p_value)
    ranges = ", ".join(f"[{start}, {stop})" for start, stop in result.ranges)

    if isinstance(result.energies, BootstrapArray):
        assert isinstance(result.lambdas, BootstrapArray)
        energy_lower, energy_upper, energy_ng, energy_normality_p = bootstrap_normality(
            result.energies
        )
        lambda_lower, lambda_upper, lambda_ng, lambda_normality_p = bootstrap_normality(
            result.lambdas
        )
        rows = "".join(
            "<tr>"
            f"<td>{i}</td>"
            f"{bootstrap_value_html(energy, error)}"
            f"{bootstrap_value_html(lamb, lamb_error, *_spectrum_normality_html(energy, error, energy_lo, energy_hi, e_ng, energy_p, lamb, lamb_error, lambda_lo, lambda_hi, l_ng, lambda_p))}"
            "</tr>"
            for i, (
                energy,
                error,
                lamb,
                lamb_error,
                energy_lo,
                energy_hi,
                e_ng,
                energy_p,
                lambda_lo,
                lambda_hi,
                l_ng,
                lambda_p,
            ) in enumerate(
                zip(
                    result.energies.central,
                    result.energies.error(),
                    result.lambdas.central,
                    result.lambdas.error(),
                    energy_lower,
                    energy_upper,
                    energy_ng,
                    energy_normality_p,
                    lambda_lower,
                    lambda_upper,
                    lambda_ng,
                    lambda_normality_p,
                )
            )
        )
        header = "<th>State</th><th>Energy</th><th>Lambda</th>"
    else:
        rows = "".join(
            f"<tr><td>{i}</td><td>{energy:.4g}</td><td>{lamb:.4g}</td></tr>"
            for i, (energy, lamb) in enumerate(zip(result.energies, result.lambdas))
        )
        header = "<th>State</th><th>Energy</th><th>Lambda</th>"

    return f"""
        {normality_css()}
        <table>
          <tr><th colspan=\"2\" style=\"text-align:center\">Laplace-filter spectrum</th></tr>
          <tr>
            <td colspan=\"2\">T<sup>2</sup> / dof = {result.t2:.4g} / {result.dof} = {result.t2 / result.dof:.2g}</td>
          </tr>
          <tr>
            <td colspan=\"2\">Time range = {ranges}</td>
          </tr>
          <tr>
            <td colspan=\"2\" style=\"text-align:center;background-color:{colour};color:black\">
              p = {p_value:.2g} ({significance:.2g}σ)
            </td>
          </tr>
          <tr>
            <td colspan=\"2\" style=\"text-align:left\">CDR at minimum = {result.cdr:.2g} dB</td>
          </tr>
        </table>
        <table style="margin-right:16em">
          <tr>{header}</tr>
          {rows}
        </table>
    """


def render_laplace_filter_amplitudes_html[T: npt.NDArray](
    result: "LaplaceFilterAmplitudes[T]",
) -> str:
    """Render a compact amplitude summary in Jupyter notebooks."""
    p_value = float(result.p_value)
    significance, colour = p_value_colour(p_value)
    ranges = ", ".join(f"[{start}, {stop})" for start, stop in result.ranges)

    if isinstance(result.amplitudes, BootstrapArray):
        amplitudes = result.amplitudes.central
        errors = result.amplitudes.error()
        lower, upper, non_gaussian, normality_p = bootstrap_normality(result.amplitudes)
        rows = "".join(
            "<tr>"
            f"<td>A<sub>{','.join(str(i) for i in index)}</sub></td>"
            + bootstrap_value_html(
                float(amplitudes[index]),
                float(errors[index]),
                *non_gaussian_html(
                    normality_p[index],
                    (
                        (
                            "err",
                            float(errors[index]),
                            float(upper[index] - amplitudes[index]),
                            float(amplitudes[index] - lower[index]),
                        ),
                    )
                    if non_gaussian[index]
                    else (),
                ),
            )
            + "</tr>"
            for index in np.ndindex(amplitudes.shape)
        )
        header = "<th>Amplitude</th><th>Value</th>"
    else:
        amplitudes = result.amplitudes
        rows = "".join(
            "<tr>"
            f"<td>A<sub>{','.join(str(i) for i in index)}</sub></td>"
            f"<td>{float(amplitudes[index]):.4g}</td>"
            "</tr>"
            for index in np.ndindex(amplitudes.shape)
        )
        header = "<th>Amplitude</th><th>Value</th>"

    return f"""
        {normality_css()}
        <table>
          <tr><th colspan=\"2\" style=\"text-align:center\">Laplace-filter amplitudes</th></tr>
          <tr>
            <td colspan=\"2\">χ<sup>2</sup> / dof = {result.chi2:.4g} / {result.dof} = {result.chi2 / result.dof:.2g}</td>
          </tr>
          <tr>
            <td colspan=\"2\">Time range = {ranges}</td>
          </tr>
          <tr>
            <td colspan=\"2\" style=\"text-align:center;background-color:{colour};color:black\">
              p = {p_value:.2g} ({significance:.2g}σ)
            </td>
          </tr>
          <tr>
            <td colspan=\"2\" style=\"text-align:left\">CDR at minimum = {result.cdr:.2g} dB</td>
          </tr>
        </table>
        <table style="margin-right:16em">
          <tr>{header}</tr>
          {rows}
        </table>
    """
