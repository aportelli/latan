from base64 import b64encode
from html import escape
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from latan.display._common import _component_labels, p_value_colour

if TYPE_CHECKING:
    from latan.statistics.normality import NormalityTest

def _qq_plot(result: "NormalityTest", labels: list[str], width: int) -> str:
    observed = result.qq_observed.reshape(result.n_samples, -1)
    n_columns = 4
    n_rows = int(np.ceil(result.n_components / n_columns))
    figure_height = 2.6 * n_rows
    left, right, bottom, title_height = 0.02, 0.99, 0.02, 0.34
    panel_height = (figure_height - title_height - bottom * figure_height) / n_rows
    figure_width = n_columns * panel_height / (right - left)
    font_size = 13
    figure = Figure(figsize=(figure_width, figure_height))
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(n_rows, n_columns)).ravel()
    figure.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=1.0 - title_height / figure_height,
        wspace=0.0,
        hspace=0.0,
    )
    figure.suptitle(
        "Sample quantiles vs normal quantiles",
        x=0.02,
        y=1.0 - 0.1 / figure_height,
        ha="left",
        fontsize=font_size + 2,
    )
    for axes_item, values, label in zip(axes, observed.T, labels):
        low = float(min(result.qq_theoretical.min(), values.min()))
        high = float(max(result.qq_theoretical.max(), values.max()))
        padding = (high - low) * 0.05 or 1.0
        low -= padding
        high += padding
        axes_item.scatter(result.qq_theoretical, values, s=8)
        for sigma in (1, 2, 3):
            colour = {1: "0.35", 2: "0.6", 3: "0.8"}[sigma]
            axes_item.axhline(sigma, color=colour, linestyle="--", linewidth=0.8)
            axes_item.axhline(-sigma, color=colour, linestyle="--", linewidth=0.8)
            axes_item.axvline(sigma, color=colour, linestyle="--", linewidth=0.8)
            axes_item.axvline(-sigma, color=colour, linestyle="--", linewidth=0.8)
        axes_item.plot(
            (low, high), (low, high), color="black", linestyle="--", linewidth=1.5
        )
        axes_item.scatter(
            0.0, 0.0, color="black", marker="+", s=180, linewidths=1.2, zorder=4
        )
        axes_item.set(
            xlim=(low, high),
            ylim=(low, high),
        )
        axes_item.set_aspect("equal", adjustable="box")
        axes_item.tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False
        )
        axes_item.text(
            0.04,
            0.96,
            label,
            transform=axes_item.transAxes,
            va="top",
            fontsize=font_size,
        )
    for axes_item in axes[result.n_components :]:
        axes_item.set_visible(False)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=200)
    image = b64encode(buffer.getvalue()).decode()
    return (
        f'<img style="display:block;width:{width}px;max-width:100%;height:auto" '
        f'src="data:image/png;base64,{image}" '
        'alt="Componentwise normal Q-Q plots" />'
    )


def _p_value_cell(p_value: float) -> str:
    significance, colour = p_value_colour(p_value)
    return (
        f'<td style="background-color:{colour};color:black">'
        f"{p_value:.2g} ({significance:.2g}σ)</td>"
    )


def render_normality_html(result: "NormalityTest") -> str:
    """Render componentwise normality diagnostics and Q-Q plots."""
    labels = _component_labels(result.observable_shape)
    table_width = 600
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{skewness:.4g}</td>"
        f"<td>{kurtosis:.4g}</td>"
        f"<td>{reduced_statistic:.4g}</td>"
        f"{_p_value_cell(float(p_value))}"
        "</tr>"
        for label, skewness, kurtosis, reduced_statistic, p_value in zip(
            labels,
            result.skewness.ravel(),
            result.kurtosis_excess.ravel(),
            result.reduced_statistic.ravel(),
            result.p_value.ravel(),
        )
    )
    title = "Normality diagnostic" if result.is_scalar else "Componentwise normality diagnostic"
    return f"""
        <div style="width:fit-content;max-width:100%">
          <div style="overflow-x:auto">
            <table style="width:{table_width}px">
              <tr><th colspan="5" style="text-align:center">{title}: {result.n_samples} samples</th></tr>
              <tr>
                <th>Component</th><th>Skewness</th><th>Excess kurtosis</th>
                <th>K<sup>2</sup> / 2</th><th>Normality p-value</th>
              </tr>
              {rows}
            </table>
          </div>
          {_qq_plot(result, labels, table_width)}
        </div>
    """
