from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from latan.display._common import p_value_colour
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

    if isinstance(result.energies, BootstrapArray):
        assert isinstance(result.lambdas, BootstrapArray)
        rows = "".join(
            "<tr>"
            f"<td>{i}</td><td>{energy:.4g}</td><td>{error:.4g}</td>"
            f"<td>{lamb:.4g}</td><td>{lamb_error:.4g}</td>"
            "</tr>"
            for i, (energy, error, lamb, lamb_error) in enumerate(
                zip(
                    result.energies.central,
                    result.energies.error(),
                    result.lambdas.central,
                    result.lambdas.error(),
                )
            )
        )
        header = (
            "<th>State</th><th>Energy</th><th>Error</th>"
            "<th>Lambda</th><th>Error</th>"
        )
    else:
        rows = "".join(
            f"<tr><td>{i}</td><td>{energy:.4g}</td><td>{lamb:.4g}</td></tr>"
            for i, (energy, lamb) in enumerate(zip(result.energies, result.lambdas))
        )
        header = "<th>State</th><th>Energy</th><th>Lambda</th>"

    return f"""
        <table>
          <tr><th colspan=\"2\" style=\"text-align:center\">Laplace-filter spectrum</th></tr>
          <tr>
            <td colspan=\"2\">T<sup>2</sup> / dof = {result.t2:.4g} / {result.dof} = {result.t2 / result.dof:.2g}</td>
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
        <table>
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

    amplitudes = (
        result.amplitudes.central
        if isinstance(result.amplitudes, BootstrapArray)
        else result.amplitudes
    )
    errors = (
        result.amplitudes.error()
        if isinstance(result.amplitudes, BootstrapArray)
        else None
    )
    rows = "".join(
        "<tr>"
        f"<td>A<sub>{','.join(str(i) for i in index)}</sub></td>"
        f"<td>{amplitudes[index]:.4g}</td>"
        + (f"<td>{errors[index]:.4g}</td>" if errors is not None else "")
        + "</tr>"
        for index in np.ndindex(amplitudes.shape)
    )
    header = "<th>Amplitude</th><th>Value</th>"
    if errors is not None:
        header += "<th>Error</th>"

    return f"""
        <table>
          <tr><th colspan=\"2\" style=\"text-align:center\">Laplace-filter amplitudes</th></tr>
          <tr>
            <td colspan=\"2\">χ<sup>2</sup> / dof = {result.chi2:.4g} / {result.dof} = {result.chi2 / result.dof:.2g}</td>
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
        <table>
          <tr>{header}</tr>
          {rows}
        </table>
    """
