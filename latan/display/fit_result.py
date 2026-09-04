from html import escape
from typing import TYPE_CHECKING

import numpy.typing as npt

from latan.display._common import (
    bootstrap_error_html,
    bootstrap_normality,
    non_gaussian_html,
    normality_css,
    p_value_colour,
)
from latan.statistics.bootstrap import BootstrapArray

if TYPE_CHECKING:
    from latan.statistics.fit import FitResult


def _parameter_normality_html(
    value: float,
    lower: float,
    upper: float,
    non_gaussian: bool,
    p_value: float,
) -> tuple[str, str]:
    errors = (("err", upper - value, value - lower),) if non_gaussian else ()
    return non_gaussian_html(p_value, errors)


def render_fit_result_html[T: npt.NDArray](result: "FitResult[T]") -> str:
    """Render a compact fit summary in Jupyter notebooks."""
    p_value = float(result.p_value)
    significance, colour = p_value_colour(p_value)

    if isinstance(result.parameters, BootstrapArray):
        mean = result.parameters.central
        err = result.parameters.error()
        lower, upper, non_gaussian, normality_p = bootstrap_normality(result.parameters)
        parameter_rows = "".join(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{value:.4g}</td>"
            f"{bootstrap_error_html(error, *_parameter_normality_html(value, lo, hi, ng, normal_p))}"
            "</tr>"
            for name, value, error, lo, hi, ng, normal_p in zip(
                result._display_parameter_names,
                mean,
                err,
                lower,
                upper,
                non_gaussian,
                normality_p,
            )
        )
        parameter_header = "<th>Name</th><th>Value</th><th>Std</th>"
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
        {normality_css()}
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
        <table style="margin-right:16em">
          <tr>{parameter_header}</tr>
          {parameter_rows}
        </table>
        {latent}
    """
