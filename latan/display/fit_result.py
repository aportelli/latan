from html import escape
from typing import TYPE_CHECKING

import numpy.typing as npt

from latan.display._common import p_value_colour
from latan.statistics.bootstrap import BootstrapArray

if TYPE_CHECKING:
    from latan.statistics.fit import FitResult


def render_fit_result_html[T: npt.NDArray](result: "FitResult[T]") -> str:
    """Render a compact fit summary in Jupyter notebooks."""
    p_value = float(result.p_value)
    significance, colour = p_value_colour(p_value)

    if isinstance(result.parameters, BootstrapArray):
        mean = result.parameters.central
        err = result.parameters.std()
        parameter_rows = "".join(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{value:.4g}</td>"
            f"<td>{error:.4g}</td>"
            "</tr>"
            for name, value, error in zip(result._display_parameter_names, mean, err)
        )
        parameter_header = "<th>Name</th><th>Value</th><th>Error</th>"
    else:
        parameter_rows = "".join(
            f"<tr><td>{escape(name)}</td><td>{value:.4g}</td></tr>"
            for name, value in zip(result._display_parameter_names, result.parameters)
        )
        parameter_header = "<th>Name</th><th>Value</th>"

    n_latent = result.parameters.shape[-1] - result.n_model_parameters
    latent = (
        f'<p class="fitresult-latent">{n_latent} latent parameters hidden</p>'
        if n_latent
        else ""
    )
    return f"""
        <table>
          <tr><th colspan=\"2\" style=\"text-align:center\">Fit summary</th></tr>
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
          <tr>{parameter_header}</tr>
          {parameter_rows}
        </table>
        {latent}
    """
