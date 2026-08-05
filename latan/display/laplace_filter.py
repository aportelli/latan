from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from latan.display._common import (
    bootstrap_error_html,
    bootstrap_normality,
    non_gaussian_attributes,
    normality_css,
    p_value_colour,
)
from latan.statistics.bootstrap import BootstrapArray

if TYPE_CHECKING:
    from latan.physics.laplace_filter.amplitudes import LaplaceFilterAmplitudes
    from latan.physics.laplace_filter.spectrum import (
        LaplaceFilterEnergies,
    )


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
            f"<td>{i}</td><td>{energy:.4g}</td>"
            f"{bootstrap_error_html(energy, error, energy_lo, energy_hi, e_ng)}"
            f"<td>{lamb:.4g}</td>"
            f"{bootstrap_error_html(lamb, lamb_error, lambda_lo, lambda_hi, l_ng, non_gaussian_attributes((energy_p, lambda_p)))}"
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
        header = (
            "<th>State</th><th>Energy</th><th>Error</th><th>Lambda</th><th>Error</th>"
        )
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
        <table style="margin-right:3em">
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
            f"<td>{float(amplitudes[index]):.4g}</td>"
            + bootstrap_error_html(
                float(amplitudes[index]),
                float(errors[index]),
                float(lower[index]),
                float(upper[index]),
                bool(non_gaussian[index]),
                non_gaussian_attributes(normality_p[index]),
            )
            + "</tr>"
            for index in np.ndindex(amplitudes.shape)
        )
        header = "<th>Amplitude</th><th>Value</th><th>Error</th>"
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
        <table style="margin-right:3em">
          <tr>{header}</tr>
          {rows}
        </table>
    """
