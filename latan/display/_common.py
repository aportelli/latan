from math import floor

import numpy as np
import numpy.typing as npt
from scipy import stats

from latan.statistics.bootstrap import BootstrapArray

NORMALITY_P_THRESHOLD = 2.0 * stats.norm.sf(2.0)


def _component_labels(shape: tuple[int, ...]) -> list[str]:
    if not shape:
        return ["Value"]
    return [
        "[" + ", ".join(str(index) for index in np.unravel_index(i, shape)) + "]"
        for i in range(int(np.prod(shape)))
    ]


def p_value_colour(p_value: float) -> tuple[float, str]:
    significance = normality_significance(p_value)
    if significance < 2:
        colour = "#2e7d32"
    elif significance < 3:
        colour = "#ed6c02"
    else:
        colour = "#c62828"
    return significance, colour


def normality_significance(p_value: float) -> float:
    return float(stats.norm.isf(p_value / 2) if p_value > 0 else np.inf)


def bootstrap_normality(
    data: BootstrapArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from latan.statistics.normality import normality_test

    test = normality_test(data)
    lower, upper = np.quantile(
        data.samples, (stats.norm.cdf(-1.0), stats.norm.cdf(1.0)), axis=0
    )
    lower, upper = np.asarray(lower), np.asarray(upper)
    p_value = np.asarray(test.p_value)
    non_gaussian = p_value < NORMALITY_P_THRESHOLD
    return lower, upper, non_gaussian, p_value


def bootstrap_value_text(value: float, error: float) -> str:
    """Format a value with its standard uncertainty in parentheses."""
    if not np.isfinite(value) or not np.isfinite(error) or error <= 0:
        return f"{value:.4g} ± {error:.4g}"
    precision = _bootstrap_error_precision(error)
    assert precision is not None
    place, n_digits = precision
    uncertainty = round(error / 10**place)
    if uncertainty == 10**n_digits:
        place += 1
        uncertainty = round(error / 10**place)
    return f"{value:.{max(0, -place)}f}({uncertainty})"


def _bootstrap_error_precision(error: float) -> tuple[int, int] | None:
    if not np.isfinite(error) or error <= 0:
        return None
    exponent = floor(np.log10(error))
    n_digits = 2 if error / 10**exponent < 3 else 1
    return exponent - n_digits + 1, n_digits


def bootstrap_asymmetric_error_text(
    error: float, upper: float, lower: float
) -> tuple[str, str]:
    """Format asymmetric errors at the precision of a standard uncertainty."""
    precision = _bootstrap_error_precision(error)
    if precision is None:
        return f"+{upper:.4g}", f"−{lower:.4g}"
    place, _ = precision
    return f"+{round(upper / 10**place)}", f"−{round(lower / 10**place)}"


def bootstrap_value_html(
    value: float,
    error: float,
    attributes: str = "",
    annotation: str = "",
) -> str:
    return f"<td{attributes}>{bootstrap_value_text(value, error)}{annotation}</td>"


def bootstrap_error_text(
    value: float,
    error: float,
) -> str:
    return f"{value:.4g} ± {error:.4g}"


def asymmetric_error_text(value: float, lower: float, upper: float, label: str = "") -> str:
    prefix = f"err_{label}" if label else "err"
    return f"{prefix} = +{upper - value:.4g} −{value - lower:.4g}"


def non_gaussian_html(
    p_values: npt.ArrayLike,
    errors: tuple[tuple[str, float, float, float], ...] = (),
) -> tuple[str, str]:
    p_values = np.asarray(p_values)
    flagged = p_values[p_values < NORMALITY_P_THRESHOLD]
    if flagged.size == 0:
        return "", ""
    _, colour = p_value_colour(float(flagged.min()))
    items = "&nbsp;&amp;&nbsp;".join(
        f'<span class="latan-ng-item">{label}&nbsp;('
        f'<small style="display:inline-block;vertical-align:middle;'
        'line-height:0.85;text-align:left">'
        f'<span style="display:block">{bootstrap_asymmetric_error_text(error, upper, lower)[0]}</span>'
        f'<span style="display:block">{bootstrap_asymmetric_error_text(error, upper, lower)[1]}</span>'
        "</small>)</span>"
        for label, error, upper, lower in errors
    )
    annotation = f'<span class="latan-ng" style="--latan-ng-colour:{colour}">'
    annotation += f"<b>NN</b> {items}</span>"
    return ' class="latan-ng-holder"', annotation


def non_gaussian_text(
    p_values: npt.ArrayLike,
    *,
    sigma: bool = False,
    errors: tuple[str, ...] = (),
) -> str:
    p_values = np.asarray(p_values)
    flagged = p_values[p_values < NORMALITY_P_THRESHOLD]
    if flagged.size == 0:
        return ""
    if sigma:
        return f" NN ({normality_significance(float(flagged.min())):.2g}σ)"
    if errors:
        return f" NN ({' & '.join(errors)})"
    return " NN"


def normality_css() -> str:
    return """
        <style>
          td.latan-ng-holder { position:relative; overflow:visible; }
          span.latan-ng {
            position:absolute;
            left:calc(100% + 0.35em);
            top:50%;
            transform:translateY(-50%);
            color:var(--latan-ng-colour);
            display:inline-flex;
            align-items:center;
            white-space:nowrap;
          }
          span.latan-ng-item {
            display:inline-flex;
            align-items:center;
          }
          span.latan-ng b { margin-right:0.25em; }
        </style>
    """
