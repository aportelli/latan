from html import escape
from typing import Any, cast

import numpy as np

from latan.display._common import (
    _component_labels,
    asymmetric_error_text,
    bootstrap_normality,
    bootstrap_value_html,
    bootstrap_value_text,
    non_gaussian_html,
    non_gaussian_text,
    normality_css,
)
from latan.statistics.bootstrap import BootstrapArray


def _normality(
    data: BootstrapArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if data.samples.shape[0] < 8 or not np.isrealobj(data):
        return None
    return bootstrap_normality(data)


def render_bootstrap_array_compact_text(data: BootstrapArray) -> str:
    """Render central values and standard errors on one line."""
    if data.central.size == 1:
        return f"{data.central.item():.4g} ± {data.error().item():.4g}"
    formatter = cast(
        Any,
        {
            "float_kind": lambda value: f"{value:.4g}",
            "complex_kind": lambda value: f"{value:.4g}",
        },
    )
    central = np.array2string(data.central, formatter=formatter)
    error = np.array2string(data.error(), formatter=formatter)
    return f"{central} ± {error}"


def render_bootstrap_array_paper_text(data: BootstrapArray) -> str:
    """Render values using parenthesized standard uncertainties."""
    central = data.central
    error = data.error()
    if central.size == 1:
        return bootstrap_value_text(central.item(), error.item())
    values = np.empty(central.shape, dtype=object)
    for index in np.ndindex(central.shape):
        values[index] = bootstrap_value_text(central[index], error[index])
    return np.array2string(values, formatter={"all": str})


def render_bootstrap_array_text(data: BootstrapArray) -> str:
    """Render a plain-text BootstrapArray summary."""
    central = np.asarray(data.central)
    error = np.asarray(data.error())
    normality = _normality(data)
    rows = []
    values = central.ravel()
    errors = error.ravel()
    labels = _component_labels(central.shape)
    if normality is None:
        annotations = ("",) * values.size
    else:
        lower, upper, non_gaussian, p_value = normality
        annotations = tuple(
            non_gaussian_text(
                p,
                errors=(asymmetric_error_text(value, lo, hi),) if ng else (),
            )
            for value, lo, hi, ng, p in zip(
                values,
                lower.ravel(),
                upper.ravel(),
                non_gaussian.ravel(),
                p_value.ravel(),
            )
        )
    for label, value, std, annotation in zip(labels, values, errors, annotations):
        rows.append((label, f"{value:.4g}", f"{std:.4g}{annotation}"))

    headers = ("Component", "Value", "Std")
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header = "  ".join(
        (
            f"{headers[0]:<{widths[0]}}",
            *(f"{item:>{width}}" for item, width in zip(headers[1:], widths[1:])),
        )
    )
    body = "\n".join(
        "  ".join(
            (
                f"{row[0]:<{widths[0]}}",
                *(f"{item:>{width}}" for item, width in zip(row[1:], widths[1:])),
            )
        )
        for row in rows
    )
    return f"Bootstrap array: {data.samples.shape[0]} samples\n{header}\n{body}"


def render_bootstrap_array_html(data: BootstrapArray) -> str:
    """Render a compact BootstrapArray summary in Jupyter notebooks."""
    central = np.asarray(data.central)
    error = np.asarray(data.error())
    normality = _normality(data)
    values = central.ravel()
    errors = error.ravel()
    labels = _component_labels(central.shape)
    if normality is None:
        normality_values = (None,) * values.size
    else:
        lower, upper, non_gaussian, p_value = normality
        normality_values = zip(
            lower.ravel(), upper.ravel(), non_gaussian.ravel(), p_value.ravel()
        )

    rows = ""
    for label, value, std, diagnostic in zip(labels, values, errors, normality_values):
        if diagnostic is None:
            attributes, annotation = "", ""
        else:
            lower, upper, non_gaussian, p_value = diagnostic
            asymmetry = (
                (("err", float(std), float(upper - value), float(value - lower)),)
                if non_gaussian
                else ()
            )
            attributes, annotation = non_gaussian_html(p_value, asymmetry)
        rows += (
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"{bootstrap_value_html(float(value), float(std), attributes, annotation)}"
            "</tr>"
        )
    return f"""
        {normality_css()}
        <table style="margin-right:16em">
          <tr><th colspan="2" style="text-align:center">Bootstrap array: {data.samples.shape[0]} samples</th></tr>
          <tr><th>Component</th><th>Value</th></tr>
          {rows}
        </table>
    """
